"""Power and null checks for the confidence test, using its exact fitting and decision code.

Unknown-scale model (Sect. 3.2) with Sigma_0 = I, a = mu*1, lambda ~ Gamma(alpha, rate alpha/lambda0),
lambda0 = 1/0.49^2 (the empirical robust increment scale), mu chosen so the known-scale endpoint
Brier is near the empirical five-minute level. Two price constructions:
  naive:     x_t = sum_{j<=t} z_j  (price moves by the raw signal; history informs reliability)
  efficient: x_t = log BF_H(z_{1:t}) (price is the Bayesian public posterior; no population gain)
Each replication draws 1,322 training, 664 validation and 664 test contracts (14 test dates), fits
the matched pair with confidence_test.select on the same features, and reads the test gain with
the pre-declared rules.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from confidence_test import baseline, phi, path_stats, select, date_bootstrap, classify, DELTA
from empirical_evaluation import logistic_predict, losses

LAM0 = 1 / .49 ** 2
MU = .16
EPS = .001


def log_bf_hist(Z, alpha, beta, mu):
    T = Z.shape[1]
    U = mu * Z.sum(1); R = (Z ** 2).sum(1); J = T * mu ** 2
    return (alpha + T / 2) * (np.log(beta + (R + J + 2 * U) / 2) - np.log(beta + (R + J - 2 * U) / 2))


def log_bf_end(S, T, alpha, beta, mu):
    M, C = T * mu, T
    return (alpha + .5) * (np.log(beta + (S + M) ** 2 / (2 * C)) - np.log(beta + (S - M) ** 2 / (2 * C)))


def simulate(rng, n, T, alpha, regime):
    beta = alpha / LAM0
    y = rng.integers(0, 2, n).astype(float)
    lam = rng.gamma(alpha, 1 / beta, n)
    Z = (2 * y[:, None] - 1) * MU + rng.standard_normal((n, T)) / np.sqrt(lam)[:, None]
    if regime == "naive":
        x = np.column_stack([np.zeros(n), np.cumsum(Z, 1)])
    else:
        x = np.column_stack([np.zeros(n)] + [log_bf_hist(Z[:, :t], alpha, beta, MU) for t in range(1, T + 1)])
    prices = expit(x)
    vol = rng.gamma(2., 1e4, (n, T + 1))
    asset = rng.integers(0, 2, n)
    return prices, vol, asset, y, Z


def oracle(T, alpha, regime, n=400000, seed=1):
    if regime == "efficient":
        return 0., 0.
    rng = np.random.default_rng(seed)
    beta = alpha / LAM0
    _, _, _, y, Z = simulate(rng, n, T, alpha, "naive")
    q = expit(log_bf_hist(Z, alpha, beta, MU)); r = expit(log_bf_end(Z.sum(1), T, alpha, beta, MU))
    g = (q - r) ** 2
    return float(g.mean()), float(g.std() / np.sqrt(n))


def replicate(rng, T, alpha, regime, n_test=664, n_dates=14):
    P, V, A, Y, _ = simulate(rng, 1322, T, alpha, regime)
    Pv, Vv, Av, Yv, _ = simulate(rng, 664, T, alpha, regime)
    Pt, Vt, At, Yt, _ = simulate(rng, n_test, T, alpha, regime)
    _, _, E = path_stats(P, EPS); c = float(np.median(E))
    Bt, Bv, Bs = baseline(P, V, A, EPS), baseline(Pv, Vv, Av, EPS), baseline(Pt, Vt, At, EPS)
    _, _, fr = select(Bt, Y, Bv, Yv, EPS)
    _, _, fq = select(np.column_stack([Bt, phi(P, EPS, c)]), Y, np.column_stack([Bv, phi(Pv, EPS, c)]), Yv, EPS)
    r_hat = logistic_predict(fr, Bs); q_hat = logistic_predict(fq, np.column_stack([Bs, phi(Pt, EPS, c)]))
    g = losses(Yt, r_hat, EPS)["brier"] - losses(Yt, q_hat, EPS)["brier"]
    dates = np.repeat(np.arange(n_dates), int(np.ceil(n_test / n_dates)))[:n_test]
    mean, lo, hi = date_bootstrap(g, dates, reps=2000, seed=int(rng.integers(1 << 31)))
    return {"gain": mean, "lo": lo, "hi": hi, "reading": classify(lo, hi), "gamma": fq["beta"][-1] / fq["sd"][-1],
            "endpoint_brier": float(((r_hat - Yt) ** 2).mean())}


def run(settings, reps, T=8, seed=2026, n_test=664, n_dates=14):
    out = []
    for regime, alpha in settings:
        og, ose = oracle(T, alpha, regime)
        rng = np.random.default_rng(seed + int(alpha * 100) + (0 if regime == "naive" else 7))
        rs = [replicate(rng, T, alpha, regime, n_test, n_dates) for _ in range(reps)]
        g = np.array([r["gain"] for r in rs])
        row = {"regime": regime, "alpha": alpha, "T": T, "n_test": n_test, "n_dates": n_dates, "oracle_gain": og, "oracle_se": ose, "reps": reps,
               "achieved_mean": float(g.mean()), "achieved_sd": float(g.std()),
               "p_lo_above_0": float(np.mean([r["lo"] > 0 for r in rs])),
               "p_hi_below_delta": float(np.mean([r["hi"] < DELTA for r in rs])),
               "p_hi_below_0": float(np.mean([r["hi"] < 0 for r in rs])),
               "p_gamma_negative": float(np.mean([r["gamma"] < 0 for r in rs])),
               "endpoint_brier": float(np.mean([r["endpoint_brier"] for r in rs])),
               "readings": {k: int(sum(r["reading"] == k for r in rs)) for k in set(r["reading"] for r in rs)}}
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "oracle":
        for a in (.5, 1, 1.5, 2, 3, 4, 8, 16, 64):
            print(a, oracle(8, a, "naive", n=200000))
    else:
        reps = int(sys.argv[2])
        settings = [("efficient", 1.), ("efficient", 3.), ("efficient", 8.)] + [("naive", a) for a in json.loads(sys.argv[3])]
        res = run(settings, reps)
        res += run(settings, reps, n_test=2350, n_dates=49)
        Path(__file__).resolve().parents[1].joinpath("code/feedback_results/confidence_power.json").write_text(json.dumps(res, indent=1) + "\n")
