"""Oracle history vs oracle endpoint posteriors under the synthetic mixture (fixed design).

The endpoint density m_y(s) = sum_j pi_j (f_{yj})^{*T}(s) is computed by FFT convolution on a
fine grid, so r* = P(Y=1 | S) is exact up to grid error. The history posterior q* uses the
exact finite-prior likelihood. Brier and log decompositions:
  R_B(rhat) - R_B(q*) = E(q*-r*)^2 + E(rhat-r*)^2,
  R_log(rhat) - R_log(q*) = E KL(q*||r*) + E KL(r*||rhat).
Part B: aligned Gaussian with a centred Student-t5 contamination of weight eps.
"""
import json
import numpy as np
from scipy.special import expit, gammaln, logsumexp
from scipy.optimize import minimize

STATES = np.array([(w, m) for w in (.15, .4, .65) for m in (.25, .5, .75)])

def gate_weights(v, omega):
    raw = np.stack(np.broadcast_arrays(omega*(1+v)**.5, (1-omega)*2/3 + 0*v, (1-omega)/3*(1+v)**.3), -1)
    return raw/raw.sum(-1, keepdims=True)

def log_density(x, v, label, omega, mu):
    sign = 2*label-1
    mi = sign*mu*(-np.expm1(-.1*v)); si = .3/np.sqrt(1+.05*v); mo = -sign*.3*expit((v-1.5)/.3)
    li = -.5*((x-mi)/si)**2-np.log(si)-.5*np.log(2*np.pi)
    ln = -.5*(x/.5)**2-np.log(.5)-.5*np.log(2*np.pi)
    z = (x-mo)/.4
    lo = gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-np.log(.4)-3*np.log1p(z*z/5)
    w = gate_weights(v, omega)
    return logsumexp(np.stack(np.broadcast_arrays(li, ln, lo), -1)+np.log(w), -1)

def sample_step(rng, n, v, label, omega, mu):
    w = gate_weights(v, omega); k = (rng.random(n)[:, None] > np.cumsum(w)[None, :]).sum(1)
    sign = 2*label-1
    mi = sign*mu*(-np.expm1(-.1*v)); si = .3/np.sqrt(1+.05*v); mo = -sign*.3*expit((v-1.5)/.3)
    out = np.where(k == 0, mi+si*rng.standard_normal(n), np.where(k == 1, .5*rng.standard_normal(n), mo+.4*rng.standard_t(5, n)))
    return out

def conv_logpdf(step_logpdf, T, L, h=.002):
    n = int(2**np.ceil(np.log2(2*L/h)))
    x = (np.arange(n)-n//2)*h
    f = np.exp(step_logpdf(x)); f /= f.sum()*h
    F = np.fft.fft(np.fft.ifftshift(f*h))
    g = np.real(np.fft.fftshift(np.fft.ifft(F**T)))/h
    return x, np.log(np.maximum(g, 1e-300))

def kl(p, q):
    p = np.clip(p, 1e-12, 1-1e-12); q = np.clip(q, 1e-12, 1-1e-12)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))

def fit_slope(s, y):
    a = np.column_stack([np.ones_like(s), s])
    obj = lambda b: (np.logaddexp(0, a@b).sum()-y@(a@b), a.T@(expit(a@b)-y))
    return minimize(obj, np.zeros(2), jac=True, method='L-BFGS-B').x

def part_a(Ts=(5, 10, 25, 50), v=1., n=40000, ntrain=2000, seed=7, grids=((.002, 1.), (.001, 1.5))):
    rows = []
    for T in Ts:
        rng = np.random.default_rng(seed+T)
        L0 = .15*T+5*np.sqrt(T)+25
        def endpoint(h, mult):
            dens = {}
            for y in (0, 1):
                comps = []
                for w, mu in STATES:
                    x, lg = conv_logpdf(lambda xx: log_density(xx, v, y, w, mu), T, L0*mult, h)
                    comps.append(lg)
                dens[y] = (x, logsumexp(np.array(comps), 0)-np.log(len(STATES)))
            return dens
        dens = endpoint(*grids[0]); dens_fine = endpoint(*grids[1])
        def sim(m):
            y = rng.integers(0, 2, m); st = rng.integers(0, 9, m)
            Z = np.empty((m, T))
            for j, (w, mu) in enumerate(STATES):
                for lab in (0, 1):
                    ix = (st == j) & (y == lab)
                    for t in range(T):
                        Z[ix, t] = sample_step(rng, ix.sum(), v, lab, w, mu)
            return y, Z
        y, Z = sim(n)
        lh = {lab: logsumexp(np.stack([log_density(Z, v, lab, w, mu).sum(1) for w, mu in STATES]), 0) for lab in (0, 1)}
        q = expit(lh[1]-lh[0]); S = Z.sum(1)
        le = {lab: np.interp(S, dens[lab][0], dens[lab][1]) for lab in (0, 1)}
        r = expit(le[1]-le[0])
        rf = expit(np.interp(S, dens_fine[1][0], dens_fine[1][1])-np.interp(S, dens_fine[0][0], dens_fine[0][1]))
        ytr, Ztr = sim(ntrain); b = fit_slope(Ztr.sum(1), ytr.astype(float)); rh = expit(b[0]+b[1]*S)
        rows.append(dict(T=T, brier_q=float(((q-y)**2).mean()), brier_r=float(((r-y)**2).mean()), brier_rhat=float(((rh-y)**2).mean()),
                         intrinsic_brier=float(((q-r)**2).mean()), calib_brier=float(((rh-r)**2).mean()),
                         intrinsic_log=float(kl(q, r).mean()), calib_log=float(kl(r, rh).mean()),
                         # consistency: r* should equal E[q*|S]; report mean(q-r) and corr of residual with S
                         se_intrinsic_brier=float(((q-r)**2).std()/np.sqrt(n)), se_intrinsic_log=float(kl(q, r).std()/np.sqrt(n)),
                         fine_intrinsic_brier=float(((q-rf)**2).mean()), fine_intrinsic_log=float(kl(q, rf).mean()),
                         max_abs_r_change=float(np.abs(r-rf).max()), grid=L0, n=n))
        print(rows[-1], flush=True)
    return rows

def part_b(T=10, mu=.1, sigma=.5, epss=(0, .001, .01, .03, .1, .2, .4), n=400000, seed=11):
    rows = []
    for eps in epss:
        rng = np.random.default_rng(seed)
        ts = sigma*np.sqrt(3/5)
        def lp(x, y):
            g = -.5*((x-(2*y-1)*mu)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)
            if eps == 0: return g
            z = x/ts
            s = gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-np.log(ts)-3*np.log1p(z*z/5)
            return np.logaddexp(np.log1p(-eps)+g, np.log(eps)+s)
        L = 20+6*sigma*np.sqrt(T)
        dens = {y: conv_logpdf(lambda xx: lp(xx, y), T, L) for y in (0, 1)}
        densf = {y: conv_logpdf(lambda xx: lp(xx, y), T, 1.5*L, .001) for y in (0, 1)}
        y = rng.integers(0, 2, n)
        cont = rng.random((n, T)) < eps
        Z = np.where(cont, ts*rng.standard_t(5, (n, T)), (2*y[:, None]-1)*mu+sigma*rng.standard_normal((n, T)))
        q = expit(lp(Z, 1).sum(1)-lp(Z, 0).sum(1)); S = Z.sum(1)
        r = expit(np.interp(S, *dens[1])-np.interp(S, *dens[0]))
        rf = expit(np.interp(S, *densf[1])-np.interp(S, *densf[0]))
        ih = float(np.log(2)-np.mean([-(qq*np.log(qq)+(1-qq)*np.log(1-qq)) for qq in [np.clip(q, 1e-12, 1-1e-12)]]))
        ie = float(np.log(2)-np.mean([-(rr*np.log(rr)+(1-rr)*np.log(1-rr)) for rr in [np.clip(r, 1e-12, 1-1e-12)]]))
        k = kl(q, r); b2 = (q-r)**2
        rows.append(dict(eps=eps, n=n, I_history=ih, I_endpoint=ie, gain_log=float(k.mean()), se_log=float(k.std()/np.sqrt(n)),
                         gain_brier=float(b2.mean()), se_brier=float(b2.std()/np.sqrt(n)),
                         fine_gain_log=float(kl(q, rf).mean()), fine_gain_brier=float(((q-rf)**2).mean()), max_abs_r_change=float(np.abs(r-rf).max()), L=L))
        print(rows[-1], flush=True)
    return rows

if __name__ == '__main__':
    out = dict(part_a=part_a(), part_b=part_b())
    from pathlib import Path
    json.dump(out, open(Path(__file__).resolve().parents[1] / 'code/feedback_results/oracle.json', 'w'), indent=1)
