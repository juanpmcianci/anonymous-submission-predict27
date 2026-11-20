"""Matched test of the confidence mechanism suggested by the unknown-scale result.

Both forecasts get the same baseline b: x_0, an asset-specific cubic in x_T (x_T, x_T^2, x_T^3,
asset, asset*x_T, asset*x_T^2, asset*x_T^3) and log(1+v_k) for every retained candle. The
history forecast adds one feature,
    phi = S * log(1 + E/c),   S = x_T - x_0,   E = sum_t (z_t - S/T)^2,
with c the median of E over training paths. Both are penalized logistic regressions fitted on
the training window, with the penalty chosen from {0.01, 0.1, 1, 10} by validation log loss
(the procedure of the other training-fitted comparators). The coefficient on phi is unconstrained.

    python3 code/confidence_test.py freeze      # fit on development data, write frozen models
    python3 code/confidence_test.py evaluate --data DIR --start-date D --stop-date D --out DIR [--label NAME]

Gains are G = R(endpoint model) - R(history model), so positive values favour the history
feature. Decision rules and delta are fixed in revision/transfer_protocol.md (amendment 2).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

import empirical_evaluation as ev
from empirical_evaluation import calibration_design, design, fit_logistic, logistic_predict, losses
from transfer_evaluation import load_cohort

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "code/empirical_results/confidence_models.json"
MODELS = ROOT / "code/empirical_results/fitted_models.json"
PENALTIES = (.01, .1, 1., 10.)
DELTA = .001
SEED = 20260929
REPS = 5000


def path_stats(prices, eps):
    x = logit(np.clip(prices, eps, 1 - eps))
    z = np.diff(x, axis=1)
    S = x[:, -1] - x[:, 0]
    E = ((z - S[:, None] / z.shape[1]) ** 2).sum(1)
    return x, S, E


def baseline(prices, volumes, asset, eps):
    x, _, _ = path_stats(prices, eps)
    return np.column_stack([x[:, 0], calibration_design(prices, asset, eps), np.log1p(volumes)])


def phi(prices, eps, c):
    _, S, E = path_stats(prices, eps)
    return S * np.log1p(E / c)


def arrays(rr):
    return (np.stack([r["prices"] for r in rr]), np.stack([r["volume"] for r in rr]), np.array([r["asset"] for r in rr]),
            np.array([r["y"] for r in rr], float), np.array([r["date"] for r in rr]))


def select(Xt, yt, Xv, yv, eps):
    cands = []
    for pen in PENALTIES:
        f = fit_logistic(Xt, yt, pen)
        cands.append((float(losses(yv, logistic_predict(f, Xv), eps)["log_loss"].mean()), pen, f))
    return min(cands, key=lambda c: c[0])


def fit_pair(train, val, lead, eps):
    """Fit the matched endpoint and history models for one lead and clipping level."""
    P, V, A, Y, _ = arrays([r for r in train if r["lead"] == lead])
    Pv, Vv, Av, Yv, _ = arrays([r for r in val if r["lead"] == lead])
    _, _, E = path_stats(P, eps)
    c = float(np.median(E))
    cut = np.quantile(E, [1 / 3, 2 / 3]).tolist()
    Bt, Bv = baseline(P, V, A, eps), baseline(Pv, Vv, Av, eps)
    _, pen_r, fr = select(Bt, Y, Bv, Yv, eps)
    Ht, Hv = np.column_stack([Bt, phi(P, eps, c)]), np.column_stack([Bv, phi(Pv, eps, c)])
    _, pen_q, fq = select(Ht, Y, Hv, Yv, eps)
    gamma_std = fq["beta"][-1]
    return {"c": c, "E_tertiles": cut, "endpoint": fr, "history": fq, "penalty_endpoint": pen_r, "penalty_history": pen_q,
            "gamma_standardized": gamma_std, "gamma_raw": gamma_std / fq["sd"][-1]}


def freeze():
    data = ROOT / "code/empirical_data"
    train, _ = load_cohort(data, "2026-08-03", "2026-08-31")
    val, _ = load_cohort(data, "2026-08-31", "2026-09-14")
    out = {ev.tag(lead, eps): fit_pair(train, val, lead, eps) for lead in ev.LEADS for eps in ev.EPSILONS}
    # Development-data reference for delta: raw last price vs training-fitted price calibration on validation.
    models = json.loads(MODELS.read_text())
    Pv, Vv, Av, Yv, _ = arrays([r for r in val if r["lead"] == 5])
    plin = logistic_predict(models[ev.tag(5, .001)]["calibrated_price"]["fit"], design(Pv, Vv, Av, .001))
    out["_meta"] = {"frozen": "2026-11-18", "delta_brier": DELTA,
                    "validation_brier_last_price_5m": float(((Pv[:, -1] - Yv) ** 2).mean()),
                    "validation_brier_plin_5m": float(((plin - Yv) ** 2).mean())}
    FROZEN.write_text(json.dumps(out, indent=1) + "\n")
    print("sha256", hashlib.sha256(FROZEN.read_bytes()).hexdigest())
    for k, v in out.items():
        if k != "_meta":
            print(k, "c=%.4f pen_r=%g pen_q=%g gamma_raw=%+.4f" % (v["c"], v["penalty_endpoint"], v["penalty_history"], v["gamma_raw"]))
    print(out["_meta"])


def date_bootstrap(diff, dates, block=1, reps=REPS, seed=SEED):
    """Resample whole UTC dates; block>1 resamples circular blocks of consecutive dates."""
    u = np.unique(dates)
    tot = np.array([diff[dates == d].sum() for d in u])
    cnt = np.array([(dates == d).sum() for d in u])
    n = len(u)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n] % n
    means = tot[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(diff.mean()), float(lo), float(hi)


def classify(lo, hi, delta=DELTA):
    """Pre-declared reading of a gain interval (positive favours the history feature)."""
    if lo > delta:
        return "improvement exceeding delta"
    if lo > 0:
        return "improvement"
    if hi < 0:
        return "history feature worsens prediction"
    if -delta < lo and hi < delta:
        return "practically equivalent within delta"
    if hi < delta:
        return "gains of delta or more excluded"
    return "inconclusive"


def evaluate(records, frozen, label):
    models = json.loads(MODELS.read_text())
    result = {"label": label, "contrasts": [], "strata": [], "per_date": [], "absolute": []}
    for lead in ev.LEADS:
        for eps in ev.EPSILONS:
            m = frozen[ev.tag(lead, eps)]
            rr = [r for r in records if r["lead"] == lead]
            P, V, A, Y, D = arrays(rr)
            B = baseline(P, V, A, eps)
            r_hat = logistic_predict(m["endpoint"], B)
            q_hat = logistic_predict(m["history"], np.column_stack([B, phi(P, eps, m["c"])]))
            mm = models[ev.tag(lead, eps)]
            plin = logistic_predict(mm["calibrated_price"]["fit"], design(P, V, A, eps))
            pcub = logistic_predict(mm["recent_calibrated_price"], calibration_design(P, A, eps))
            L = {k: losses(Y, p, eps) for k, p in (("endpoint", r_hat), ("history", q_hat), ("P_lin", plin), ("P_cub", pcub))}
            result["absolute"].append({"lead": lead, "epsilon": eps, "n": len(Y), "dates": len(np.unique(D)),
                                       **{f"{k}_{met}": float(L[k][met].mean()) for k in L for met in ("brier", "log_loss")}})
            for met in ("brier", "log_loss"):
                g = L["endpoint"][met] - L["history"][met]
                mean, lo, hi = date_bootstrap(g, D)
                _, blo, bhi = date_bootstrap(g, D, block=3)
                row = {"lead": lead, "epsilon": eps, "metric": met, "gain": mean, "lo": lo, "hi": hi, "block3_lo": blo, "block3_hi": bhi}
                if met == "brier":
                    row["reading"] = classify(lo, hi)
                    row["reading_block3"] = classify(blo, bhi)
                    u = np.unique(D)
                    daily = [float(g[D == d].mean()) for d in u]
                    row["dates_positive"] = int(sum(x > 0 for x in daily))
                    row["lodo_min"] = float(min(g[D != d].mean() for d in u))
                    row["lodo_max"] = float(max(g[D != d].mean() for d in u))
                    if eps == .001:
                        result["per_date"] += [{"lead": lead, "date": str(d), "gain": x} for d, x in zip(u, daily)]
                        _, _, E = path_stats(P, eps)
                        tert = np.digitize(E, m["E_tertiles"])
                        for name, ix in [("BTC", A == 0), ("ETH", A == 1)] + [(f"E tertile {k + 1}", tert == k) for k in range(3)]:
                            result["strata"].append({"lead": lead, "stratum": name, "n": int(ix.sum()), "gain": float(g[ix].mean())})
                result["contrasts"].append(row)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--start-date", required=True)
    e.add_argument("--stop-date", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--label", default="evaluation")
    e.add_argument("--exclude", nargs="*", default=[], help="tickers excluded by protocol amendment")
    args = ap.parse_args()
    if args.cmd == "freeze":
        if FROZEN.exists():
            raise SystemExit(f"{FROZEN} exists; refusing to refit a frozen model")
        freeze()
        return
    frozen = json.loads(FROZEN.read_text())
    records, rejected = load_cohort(args.data, args.start_date, args.stop_date)
    records = [r for r in records if r["ticker"] not in set(args.exclude)]
    res = evaluate(records, frozen, args.label)
    res["frozen_sha256"] = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    res["n_contracts"] = len({r["ticker"] for r in records})
    res["exclusions"] = len(rejected)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "confidence_results.json").write_text(json.dumps(res, indent=1) + "\n")
    for row in res["contrasts"]:
        if row["epsilon"] == .001:
            print(row["lead"], row["metric"], "%+.5f [%+.5f, %+.5f] block3 [%+.5f, %+.5f]" % (row["gain"], row["lo"], row["hi"], row["block3_lo"], row["block3_hi"]), row.get("reading", ""))


if __name__ == "__main__":
    main()
