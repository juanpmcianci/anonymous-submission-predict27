"""Post-review diagnostics on the frozen Kalshi study. No frozen model is refitted.

1. Usable path lengths per lead.
2. Per-date paired Brier differences and leave-one-date-out (LODO) means on the test period.
3. Matched endpoint comparator: logistic on (x_0, x_last, mean log volume, log-volume slope, asset),
   i.e. the history-logistic inputs minus the three path features. Fitted with the same
   train/validation procedure, then frozen to code/empirical_results/endpoint_matched.json.
   Its original-test score is post hoc and labelled exploratory.
4. Validation-period error of the mixture and price calibrator by initial-price bin.
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.special import logit
import empirical_evaluation as ev
from empirical_evaluation import fit_logistic, logistic_predict, losses, paired_date_bootstrap, design
from transfer_evaluation import load_cohort, forecasts_for
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code/feedback_results"; OUT.mkdir(exist_ok=True)
DATA = ROOT / "code/empirical_data"
models = json.loads((ROOT / "code/empirical_results/fitted_models.json").read_text())
prior = float(np.mean(list(json.loads((ROOT / "code/empirical_results/labels_train.json").read_text()).values())))

from feedback_diagnostics_lib import endpoint_design


def arrays(rr):
    return (np.stack([r["prices"] for r in rr]), np.stack([r["volume"] for r in rr]),
            np.array([r["asset"] for r in rr]), np.array([r["y"] for r in rr]), np.array([r["date"] for r in rr]))

train, _ = load_cohort(DATA, "2026-08-03", "2026-08-31")
val, _ = load_cohort(DATA, "2026-08-31", "2026-09-14")
test, _ = load_cohort(DATA, "2026-09-14", "2026-09-28")
report = {"lengths": {}, "endpoint_matched": {}, "lodo": {}, "per_date": []}
matched = {}
for lead in ev.LEADS:
    tr = [r for r in train if r["lead"] == lead]; va = [r for r in val if r["lead"] == lead]
    report["lengths"][lead] = {"prices": int(tr[0]["prices"].size), "increments": int(tr[0]["prices"].size - 1)}
    for eps in ev.EPSILONS:
        P, V, A, Y, _ = arrays(tr); Pv, Vv, Av, Yv, _ = arrays(va)
        Xt, Xv = endpoint_design(P, V, A, eps), endpoint_design(Pv, Vv, Av, eps)
        cands = []
        for pen in (.01, .1, 1., 10.):
            f = fit_logistic(Xt, Y, pen); cands.append((float(losses(Yv, logistic_predict(f, Xv), eps)["log_loss"].mean()), f))
        matched[ev.tag(lead, eps)] = min(cands, key=lambda c: c[0])[1]
path = ROOT / "code/empirical_results/endpoint_matched.json"
path.write_text(json.dumps(matched, indent=1) + "\n")
report["endpoint_matched_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

rows = []
for lead in ev.LEADS:
    for eps in ev.EPSILONS:
        rr, f = forecasts_for(test, lead, eps, models, prior)
        P, V, A, Y, D = arrays(rr)
        f["endpoint_matched"] = logistic_predict(matched[ev.tag(lead, eps)], endpoint_design(P, V, A, eps))
        L = {k: losses(Y, q, eps) for k, q in f.items()}
        for k in f:
            report["endpoint_matched"].setdefault(f"{lead}_{eps}", {})[k] = {m: float(L[k][m].mean()) for m in ("brier", "log_loss")}
        contrasts = {"combined-recent": ("combined", "recent_calibrated_price"),
                     "combined_no_volume-recent": ("combined_no_volume", "recent_calibrated_price"),
                     "gaussian_combined-recent": ("gaussian_combined", "recent_calibrated_price"),
                     "history_logistic-endpoint_matched": ("history_logistic", "endpoint_matched"),
                     "endpoint_matched-calibrated": ("endpoint_matched", "calibrated_price"),
                     "history_logistic-recent": ("history_logistic", "recent_calibrated_price"),
                     "mixture-recent": ("mixture", "recent_calibrated_price"),
                     "mixture_no_volume-recent": ("mixture_no_volume", "recent_calibrated_price"),
                     "gaussian_history-recent": ("gaussian_history", "recent_calibrated_price"),
                     "calibrated-recent": ("calibrated_price", "recent_calibrated_price"),
                     "latest-recent": ("latest_price", "recent_calibrated_price")}
        for name, (a, b) in contrasts.items():
            for metric in ("brier", "log_loss"):
                d = L[a][metric] - L[b][metric]
                mean, lo, hi = paired_date_bootstrap(d, D)
                dates = np.unique(D)
                daily = [float(d[D == x].mean()) for x in dates]
                lodo = [float(d[D != x].mean()) for x in dates]
                rows.append({"lead": lead, "epsilon": eps, "contrast": name, "metric": metric, "mean": mean, "lo": lo, "hi": hi,
                             "lodo_min": min(lodo), "lodo_max": max(lodo), "dates_negative": int(sum(x < 0 for x in daily)), "n_dates": len(dates)})
                if eps == .001 and metric == "brier":
                    for x, v in zip(dates, daily):
                        report["per_date"].append({"lead": lead, "contrast": name, "date": str(x), "diff": v})
with (OUT / "contrasts.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# Validation diagnostic by initial-price bin (no test labels).
diag = []
for lead in ev.LEADS:
    rr, f = forecasts_for(val, lead, .001, models, prior)
    P, V, A, Y, D = arrays(rr)
    bins = np.digitize(P[:, 0], [.3, .45, .55, .7])
    for b in range(5):
        ix = bins == b
        diag.append({"lead": lead, "bin": b, "n": int(ix.sum()), "p0_mean": float(P[ix, 0].mean()), "outcome_rate": float(Y[ix].mean()),
                     "mixture_mean": float(f["mixture"][ix].mean()), "calibrated_mean": float(f["calibrated_price"][ix].mean()),
                     "mixture_brier": float(((f["mixture"] - Y) ** 2)[ix].mean()), "calibrated_brier": float(((f["calibrated_price"] - Y) ** 2)[ix].mean())})
report["validation_by_initial_price"] = diag
(OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report["lengths"]))
for r in rows:
    if r["epsilon"] == .001 and r["metric"] == "brier":
        print(r["lead"], f'{r["contrast"]:36s} {r["mean"]:+.5f} [{r["lo"]:+.5f},{r["hi"]:+.5f}] lodo[{r["lodo_min"]:+.5f},{r["lodo_max"]:+.5f}] neg {r["dates_negative"]}/{r["n_dates"]}')
for d in diag: print(d)
