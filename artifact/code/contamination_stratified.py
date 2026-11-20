"""Stratified Monte Carlo for the contamination gain I(Y;Z|S) at small weights.

Stratify on K, the number of contaminated increments among T: draw n_k histories with exactly
k contaminated positions (uniform positions), weight by Binomial(T, eps) probabilities, and
report the weighted mean with its standard error. Strata k > kmax are omitted; their total
probability is reported.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, gammaln
from scipy.stats import binom
from oracle_endpoint import conv_logpdf, kl

def run(eps, T=10, mu=.1, sigma=.5, kmax=5, n_k=100000, seed=23):
    rng = np.random.default_rng(seed)
    ts = sigma*np.sqrt(3/5)
    def lp(x, y):
        g = -.5*((x-(2*y-1)*mu)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)
        z = x/ts
        s = gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-np.log(ts)-3*np.log1p(z*z/5)
        return np.logaddexp(np.log1p(-eps)+g, np.log(eps)+s)
    L = 1.5*(20+6*sigma*np.sqrt(T))
    dens = {y: conv_logpdf(lambda xx: lp(xx, y), T, L, .001) for y in (0, 1)}
    mean, var, strata = 0., 0., []
    for k in range(kmax+1):
        w = binom.pmf(k, T, eps)
        y = rng.integers(0, 2, n_k)
        cont = np.zeros((n_k, T), bool)
        idx = np.argsort(rng.random((n_k, T)), 1)[:, :k]
        np.put_along_axis(cont, idx, True, 1)
        Z = np.where(cont, ts*rng.standard_t(5, (n_k, T)), (2*y[:, None]-1)*mu+sigma*rng.standard_normal((n_k, T)))
        q = expit(lp(Z, 1).sum(1)-lp(Z, 0).sum(1))
        r = expit(np.interp(Z.sum(1), *dens[1])-np.interp(Z.sum(1), *dens[0]))
        g = kl(q, r)
        mean += w*g.mean(); var += w**2*g.var()/n_k
        strata.append(dict(k=k, weight=float(w), mean=float(g.mean())))
    return dict(eps=eps, gain_log=float(mean), se_log=float(np.sqrt(var)), omitted_mass=float(binom.sf(kmax, T, eps)), strata=strata)

if __name__ == "__main__":
    out = [run(e) for e in (.001, .01, .03, .1)]
    for o in out:
        print(o["eps"], f'{o["gain_log"]:.3e} se {o["se_log"]:.1e} omitted {o["omitted_mass"]:.1e}', [round(s["mean"], 7) for s in o["strata"]])
    p = Path(__file__).resolve().parents[1] / "code/feedback_results/contamination_stratified.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
