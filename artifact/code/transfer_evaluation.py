"""Apply the frozen fitted models to a later Kalshi cohort and score them.

Nothing is refitted. The fitted models come from code/empirical_results/fitted_models.json,
which was produced from the 3-30 August training window and the 31 Aug-13 Sep validation
window and locked before any score was computed. Protocol addendum:
revision/transfer_protocol.md.

Validation mode: run on the original data restricted to the original test dates and compare
with code/empirical_results/scores.csv:

    python3 code/transfer_evaluation.py --data code/empirical_data \
        --start-date 2026-09-14 --stop-date 2026-09-28 --out code/transfer_check --check

Transfer run:

    python3 code/transfer_evaluation.py --data code/empirical_data_transfer \
        --start-date 2026-09-28 --stop-date 2026-11-17 --out code/transfer_results
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

import empirical_evaluation as ev
from empirical_evaluation import (EPSILONS, LEADS, METHODS, calibration_design, design, extract_prefix,
                                  logistic_predict, losses, paired_date_bootstrap, timestamp)
from empirical_mixture import EmpiricalMixture, GaussianHistory

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "code/empirical_results/fitted_models.json"
MATCHED = ROOT / "code/empirical_results/endpoint_matched.json"  # E_log, frozen 18 Nov 2026
TRAIN_LABELS = ROOT / "code/empirical_results/labels_train.json"


def load_cohort(data, start_date, stop_date):
    markets = json.loads((data / "markets.json").read_text())
    if isinstance(markets, dict):
        markets = markets["markets"]
    candles = json.loads((data / "candles.json").read_text())
    records, rejected, seen = [], [], set()
    for market in sorted(markets, key=lambda m: (m["close_time"], m["ticker"])):
        ticker = market["ticker"]
        if ticker in seen:
            raise ValueError(f"duplicate ticker: {ticker}")
        seen.add(ticker)
        closed = timestamp(market["close_time"])
        date = datetime.fromtimestamp(closed, timezone.utc).strftime("%Y-%m-%d")
        if not start_date <= date < stop_date:
            continue
        try:
            if closed % 3600:
                raise ValueError("not_exact_utc_hour")
            if ticker.startswith("KXBTC15M-"):
                asset = 0
            elif ticker.startswith("KXETH15M-"):
                asset = 1
            else:
                raise ValueError("outside_series")
            if market.get("result") not in ("yes", "no") or market.get("status") not in ("finalized", "settled") or market.get("is_provisional", False):
                raise ValueError("not_final_binary_outcome")
            if not market.get("settlement_ts") or timestamp(market["settlement_ts"]) < closed:
                raise ValueError("invalid_settlement_time")
        except ValueError as e:
            rejected.append({"ticker": ticker, "lead": "all", "reason": str(e)})
            continue
        for lead in LEADS:
            try:
                prefix = extract_prefix(market, candles.get(ticker, []), lead, .001)
            except ValueError as e:
                rejected.append({"ticker": ticker, "lead": lead, "reason": str(e)})
                continue
            records.append({"ticker": ticker, "date": date, "asset": asset, "lead": lead,
                            "y": int(market["result"] == "yes"), "prices": prefix["prices"], "volume": prefix["volume"]})
    return records, rejected


def forecasts_for(records, lead, epsilon, models, prior):
    rr = [r for r in records if r["lead"] == lead]
    prices = np.stack([r["prices"] for r in rr])
    volume = np.stack([r["volume"] for r in rr])
    assets = np.array([r["asset"] for r in rr])
    m = models[ev.tag(lead, epsilon)]
    f = {"prior": np.full(len(rr), prior), "initial_price": prices[:, 0], "latest_price": prices[:, -1]}
    f["calibrated_price"] = logistic_predict(m["calibrated_price"]["fit"], design(prices, volume, assets, epsilon))
    f["history_logistic"] = logistic_predict(m["history_logistic"]["fit"], design(prices, volume, assets, epsilon, history=True))
    price_x = calibration_design(prices, assets, epsilon)
    f["recent_calibrated_price"] = logistic_predict(m["recent_calibrated_price"], price_x)
    z = np.diff(logit(np.clip(prices, epsilon, 1 - epsilon)), axis=1)
    for suffix in ("", "_no_volume"):
        model = EmpiricalMixture.from_dict(m["mixture" + suffix])
        bf = model.log_bayes_factor(z, volume[:, 1:])
        f["mixture" + suffix] = model.predict_proba(z, volume[:, 1:])
        f["combined" + suffix] = logistic_predict(m["combined" + suffix], np.column_stack([price_x, bf]))
    gaussian = GaussianHistory.from_dict(m["gaussian_history"])
    f["gaussian_history"] = gaussian.predict_proba(z, assets)
    f["gaussian_combined"] = logistic_predict(m["gaussian_combined"], np.column_stack([price_x, gaussian.log_bayes_factor(z, assets)]))
    if MATCHED.exists():
        from feedback_diagnostics_lib import endpoint_design
        matched = json.loads(MATCHED.read_text())[ev.tag(lead, epsilon)]
        f["endpoint_matched"] = logistic_predict(matched, endpoint_design(prices, volume, assets, epsilon))
    for name, q in f.items():
        if not np.all(np.isfinite(q)) or q.min() < 0 or q.max() > 1:
            raise RuntimeError(f"invalid forecast for {name}")
    return rr, f


def score(records, models, prior, out):
    out.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, daily, by_asset = [], [], [], []
    for lead in LEADS:
        for epsilon in EPSILONS:
            rr, f = forecasts_for(records, lead, epsilon, models, prior)
            dates = np.array([r["date"] for r in rr])
            y = np.array([r["y"] for r in rr])
            assets = np.array([r["asset"] for r in rr])
            ref = losses(y, f["calibrated_price"], epsilon)
            recent = losses(y, f["recent_calibrated_price"], epsilon)
            for method in [m for m in METHODS + ("endpoint_matched",) if m in f]:
                metrics = losses(y, f[method], epsilon)
                row = {"lead": lead, "epsilon": epsilon, "method": method, "n": len(y), "dates": len(np.unique(dates))}
                for metric, values in metrics.items():
                    mean, lo, hi = paired_date_bootstrap(values, dates)
                    row.update({metric: mean, metric + "_lo": lo, metric + "_hi": hi})
                    if metric == "accuracy":
                        continue
                    refs = [("calibrated_price", ref), ("recent_calibrated_price", recent)]
                    if method == "history_logistic" and "endpoint_matched" in f:
                        refs.append(("endpoint_matched", losses(y, f["endpoint_matched"], epsilon)))
                    for name, reference in refs:
                        d, l, h = paired_date_bootstrap(values - reference[metric], dates)
                        comparisons.append({"lead": lead, "epsilon": epsilon, "method": method, "reference": name,
                                            "metric": metric, "difference": d, "lo": l, "hi": h})
                        for date in np.unique(dates):
                            ix = dates == date
                            daily.append({"lead": lead, "epsilon": epsilon, "method": method, "reference": name, "metric": metric,
                                          "date": date, "n": int(ix.sum()), "difference": float((values - reference[metric])[ix].mean())})
                summaries.append(row)
                for a in np.unique(assets):
                    ix = assets == a
                    by_asset.append({"lead": lead, "epsilon": epsilon, "method": method, "asset": "BTC" if a == 0 else "ETH",
                                     "n": int(ix.sum()), **{k: float(v[ix].mean()) for k, v in metrics.items()}})
    for name, table in (("scores.csv", summaries), ("paired_differences.csv", comparisons),
                        ("daily_differences.csv", daily), ("asset_scores.csv", by_asset)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    return summaries, comparisons


def check_against_frozen(summaries, comparisons, out):
    """Validation mode: the original test period must reproduce the frozen scores."""
    stored = {(int(r["lead"]), float(r["epsilon"]), r["method"]): r
              for r in csv.DictReader((ROOT / "code/empirical_results/scores.csv").open())}
    worst = 0.0
    for s in summaries:
        if s["method"] == "endpoint_matched":
            continue
        ref = stored[(s["lead"], s["epsilon"], s["method"])]
        for metric in ("brier", "log_loss", "accuracy", "brier_lo", "brier_hi"):
            worst = max(worst, abs(float(ref[metric]) - s[metric]))
        if int(ref["n"]) != s["n"]:
            raise RuntimeError("cohort size differs from frozen test set")
    stored_cmp = {(int(r["lead"]), float(r["epsilon"]), r["method"], r["reference"], r["metric"]): r
                  for r in csv.DictReader((ROOT / "code/empirical_results/paired_differences.csv").open())}
    for c in comparisons:
        if c["method"] == "endpoint_matched" or c["reference"] == "endpoint_matched":
            continue
        ref = stored_cmp[(c["lead"], c["epsilon"], c["method"], c["reference"], c["metric"])]
        worst = max(worst, abs(float(ref["difference"]) - c["difference"]), abs(float(ref["lo"]) - c["lo"]), abs(float(ref["hi"]) - c["hi"]))
    print(f"max absolute deviation from frozen scores and intervals: {worst:.3e}")
    if worst > 1e-9:
        raise RuntimeError("transfer pipeline does not reproduce the frozen test scores")
    (out / "check_passed.txt").write_text(f"max abs deviation {worst:.3e}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--stop-date", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="reproduce frozen test scores (original data only)")
    args = parser.parse_args()
    models = json.loads(MODELS.read_text())
    prior = float(np.mean(list(json.loads(TRAIN_LABELS.read_text()).values())))
    records, rejected = load_cohort(args.data, args.start_date, args.stop_date)
    n_contracts = len({r["ticker"] for r in records})
    print(f"{n_contracts} contracts, {len(records)} forecast records, {len(rejected)} exclusions")
    out = args.out if args.out.is_absolute() else ROOT / args.out
    summaries, comparisons = score(records, models, prior, out)
    (out / "exclusions.json").write_text(json.dumps(rejected, indent=1) + "\n")
    if args.check:
        check_against_frozen(summaries, comparisons, out)
    for s in summaries:
        if s["epsilon"] == .001 and s["method"] in ("calibrated_price", "recent_calibrated_price", "mixture", "combined"):
            print(s["lead"], s["method"], round(s["brier"], 4), round(s["log_loss"], 4))


if __name__ == "__main__":
    main()
