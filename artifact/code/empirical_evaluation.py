"""Retrospective, chronological Kalshi evaluation with separate predict/score stages.

Protocol: revision/empirical_protocol.md. This script never uses test outcomes
while fitting or generating forecasts. Public raw metadata necessarily contains
already resolved labels; this separation is an analysis safeguard, not blinding.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code/empirical_data"
OUT = ROOT / "code/empirical_results"
GENERATED = ROOT / "manuscript/generated"
EPSILONS = (.001, .005)
LEADS = (5, 1)
METHODS = ("prior", "initial_price", "latest_price", "calibrated_price", "recent_calibrated_price", "history_logistic", "mixture", "mixture_no_volume", "combined", "combined_no_volume", "gaussian_history", "gaussian_combined")


def timestamp(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def split_date(date):
    if "2026-08-03" <= date < "2026-08-31":
        return "train"
    if "2026-08-31" <= date < "2026-09-14":
        return "validation"
    if "2026-09-14" <= date < "2026-09-28":
        return "test"
    raise ValueError("outside_cohort")


def extract_prefix(market, candles, lead_minutes, epsilon):
    """Only read transaction closes and volumes at the allowed timestamps."""
    opened, closed = timestamp(market["open_time"]), timestamp(market["close_time"])
    if closed - opened != 900:
        raise ValueError("not_15_minute_window")
    cutoff = closed - lead_minutes * 60 - 60
    expected = np.arange(opened + 60, cutoff + 1, 60, dtype=np.int64)
    selected = sorted((c for c in candles if opened < c["end_period_ts"] <= cutoff), key=lambda c: c["end_period_ts"])
    times = np.array([c["end_period_ts"] for c in selected], dtype=np.int64)
    if not np.array_equal(times, expected):
        raise ValueError("missing_or_duplicate_candle")
    try:
        prices = np.array([float(c["price"]["close_dollars"]) for c in selected])
        volume = np.array([float(c["volume_fp"]) for c in selected])
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing_price_or_volume") from None
    if not np.all(np.isfinite(prices) & (prices >= 0) & (prices <= 1)):
        raise ValueError("invalid_price")
    if not np.all(np.isfinite(volume) & (volume > 0)):
        raise ValueError("missing_or_untraded_candle")
    x = logit(np.clip(prices, epsilon, 1 - epsilon))
    return {"prices": prices, "volume": volume, "x": x, "z": np.diff(x), "cutoff": cutoff, "times": times}


def design(prices, volumes, asset, epsilon, history=False):
    x = logit(np.clip(prices, epsilon, 1 - epsilon))
    if not history:
        return np.column_stack([x[:, -1], asset])
    z = np.diff(x, axis=1)
    lv = np.log1p(volumes)
    t = np.arange(lv.shape[1], dtype=float)
    t -= t.mean()
    slope = lv @ t / (t @ t)
    vw = (z * volumes[:, 1:]).sum(axis=1) / volumes[:, 1:].sum(axis=1)
    return np.column_stack([x[:, 0], x[:, -1], z[:, -3:].sum(axis=1), (z*z).sum(axis=1), lv.mean(axis=1), slope, vw, asset])


def calibration_design(prices, asset, epsilon):
    """Same asset-specific cubic endpoint basis for recent control/combinations."""
    x = logit(np.clip(prices[:, -1], epsilon, 1-epsilon))
    return np.column_stack([x, x*x, x*x*x, asset, asset*x, asset*x*x, asset*x*x*x])


def audit_label_availability(records, markets_by_ticker):
    audit = []
    for source, destination in (("train", "validation"), ("validation", "test")):
        tickers = {r["ticker"] for r in records if r["split"] == source}
        first_forecast = min(r["cutoff"] + 60 for r in records if r["split"] == destination)
        last_settlement = max(timestamp(markets_by_ticker[t]["settlement_ts"]) for t in tickers)
        if last_settlement >= first_forecast:
            raise ValueError(f"Late {source} outcome unavailable before first {destination} forecast")
        audit.append({"source": source, "destination": destination, "labels": len(tickers), "last_settlement_ts": last_settlement, "first_forecast_ts": first_forecast, "all_labels_available": True})
    return audit


def fit_logistic(x, y, penalty):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.)
    a = np.column_stack([np.ones(len(x)), (x - mean) / sd])
    def objective(beta):
        u = a @ beta
        value = np.logaddexp(0, u).sum() - y @ u + penalty * np.sum(beta[1:]**2) / 2
        gradient = a.T @ (expit(u) - y)
        gradient[1:] += penalty * beta[1:]
        return value, gradient
    fit = minimize(objective, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8})
    gradient_per_observation = float(np.max(np.abs(fit.jac)) / len(y))
    # Line-search termination can occur at floating-point objective precision.
    # Accept it only after independently checking the first-order condition.
    if not fit.success and gradient_per_observation > 1e-8:
        raise RuntimeError(f"Logistic fit failed: {fit.message}")
    return {"mean": mean.tolist(), "sd": sd.tolist(), "beta": fit.x.tolist(), "penalty": penalty, "success": True, "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message), "gradient_max_per_observation": gradient_per_observation, "iterations": int(fit.nit)}


def logistic_predict(fit, x):
    a = (np.asarray(x) - np.asarray(fit["mean"])) / np.asarray(fit["sd"])
    return expit(fit["beta"][0] + a @ np.asarray(fit["beta"])[1:])


def losses(y, p, epsilon):
    q = np.clip(p, epsilon, 1 - epsilon)
    return {"brier": (np.asarray(p) - y)**2, "log_loss": -(y*np.log(q) + (1-y)*np.log1p(-q)), "accuracy": ((p >= .5) == y).astype(float)}


def paired_date_bootstrap(diff, dates, repetitions=5000, seed=20260929):
    unique = np.unique(dates)
    totals = np.array([np.sum(diff[dates == d]) for d in unique])
    counts = np.array([np.sum(dates == d) for d in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(repetitions, len(unique)))
    means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(np.mean(diff)), float(lo), float(hi)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(lead, epsilon):
    return f"lead{lead}_clip{epsilon:g}"


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    markets = json.loads((DATA / "markets.json").read_text())
    if isinstance(markets, dict):
        markets = markets["markets"]
    candles = json.loads((DATA / "candles.json").read_text())
    rows, arrays, rejected = [], {}, []
    labels = {s: {} for s in ("train", "validation", "test")}
    seen = set()
    for market in sorted(markets, key=lambda m: (m["close_time"], m["ticker"])):
        ticker = market["ticker"]
        if ticker in seen:
            raise ValueError(f"duplicate ticker: {ticker}")
        seen.add(ticker)
        closed = timestamp(market["close_time"])
        date = datetime.fromtimestamp(closed, timezone.utc).strftime("%Y-%m-%d")
        try:
            split = split_date(date)
            if closed % 3600 != 0:
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
        labels[split][ticker] = int(market["result"] == "yes")
        for lead in LEADS:
            try:
                prefix = extract_prefix(market, candles.get(ticker, []), lead, .001)
            except ValueError as e:
                rejected.append({"ticker": ticker, "lead": lead, "reason": str(e)})
                continue
            rows.append({"ticker": ticker, "event_ticker": market["event_ticker"], "date": date, "split": split, "asset": asset, "lead": lead, "cutoff": prefix["cutoff"]})
            arrays[f"{ticker}_lead{lead}_prices"] = prefix["prices"]
            arrays[f"{ticker}_lead{lead}_volume"] = prefix["volume"]
    if not rows:
        raise RuntimeError("No eligible forecast records")
    availability = audit_label_availability(rows, {m["ticker"]: m for m in markets})
    write_json(OUT / "label_availability.json", availability)
    np.savez_compressed(OUT / "features.npz", **arrays)
    write_json(OUT / "records.json", rows)
    for split in labels:
        write_json(OUT / f"labels_{split}.json", labels[split])
    write_json(OUT / "exclusions.json", rejected)
    counts = []
    for lead in LEADS:
        for split in labels:
            for asset in (0, 1):
                subset = [r for r in rows if r["lead"] == lead and r["split"] == split and r["asset"] == asset]
                counts.append({"lead": lead, "split": split, "asset": "BTC" if asset == 0 else "ETH", "n": len(subset), "dates": len({r["date"] for r in subset})})
    write_csv(OUT / "coverage.csv", counts)
    print(json.dumps({"candidates": len(markets), "forecast_records": len(rows), "exclusions": len(rejected), "coverage": counts}, indent=2))


def predict():
    from empirical_mixture import EmpiricalMixture, GaussianHistory
    if (OUT / "scores.csv").exists():
        raise RuntimeError("Scores already exist. Do not refit after opening held-out results.")
    records = json.loads((OUT / "records.json").read_text())
    data = np.load(OUT / "features.npz", allow_pickle=False)
    # Deliberately no read of labels_test.json in this stage.
    known_labels = {}
    for split in ("train", "validation"):
        known_labels.update(json.loads((OUT / f"labels_{split}.json").read_text()))
    predictions, artifacts = [], {}
    for lead in LEADS:
        rr = [r for r in records if r["lead"] == lead]
        prices = np.stack([data[f'{r["ticker"]}_lead{lead}_prices'] for r in rr])
        volume = np.stack([data[f'{r["ticker"]}_lead{lead}_volume'] for r in rr])
        assets = np.array([r["asset"] for r in rr])
        train, val, test = (np.array([r["split"] == s for r in rr]) for s in ("train", "validation", "test"))
        ytrain = np.array([known_labels[r["ticker"]] for r in rr if r["split"] == "train"])
        yval = np.array([known_labels[r["ticker"]] for r in rr if r["split"] == "validation"])
        if min(sum(train), sum(val), sum(test)) < 20:
            raise RuntimeError("Insufficient cohort in at least one split")
        for epsilon in EPSILONS:
            key = tag(lead, epsilon)
            forecasts = {"prior": np.full(len(rr), ytrain.mean()), "initial_price": prices[:, 0], "latest_price": prices[:, -1]}
            models = {}
            for method, history in (("calibrated_price", False), ("history_logistic", True)):
                x = design(prices, volume, assets, epsilon, history=history)
                candidates = []
                for penalty in (.01, .1, 1., 10.):
                    fit = fit_logistic(x[train], ytrain, penalty)
                    q = logistic_predict(fit, x[val])
                    candidates.append((float(losses(yval, q, epsilon)["log_loss"].mean()), fit))
                _, fit = min(candidates, key=lambda pair: pair[0])
                forecasts[method] = logistic_predict(fit, x)
                models[method] = {"fit": fit, "validation_losses": [{"penalty": f["penalty"], "log_loss": v} for v, f in candidates]}
            price_x = calibration_design(prices, assets, epsilon)
            recent_fit = fit_logistic(price_x[val], yval, 1.)
            forecasts["recent_calibrated_price"] = logistic_predict(recent_fit, price_x)
            models["recent_calibrated_price"] = recent_fit
            z = np.diff(logit(np.clip(prices, epsilon, 1-epsilon)), axis=1)
            for use_volume in (True, False):
                suffix = "" if use_volume else "_no_volume"
                model = EmpiricalMixture().fit(z[train], volume[train, 1:], ytrain, use_volume=use_volume)
                bf = model.log_bayes_factor(z, volume[:, 1:])
                forecasts["mixture" + suffix] = model.predict_proba(z, volume[:, 1:])
                models["mixture" + suffix] = model.to_dict()
                # A validation-fitted classifier combination, not a second Bayesian update.
                combo_x = np.column_stack([price_x, bf])
                combo_fit = fit_logistic(combo_x[val], yval, 1.)
                forecasts["combined" + suffix] = logistic_predict(combo_fit, combo_x)
                models["combined" + suffix] = combo_fit
            gaussian = GaussianHistory().fit(z[train], assets[train], ytrain)
            gaussian_bf = gaussian.log_bayes_factor(z, assets)
            forecasts["gaussian_history"] = gaussian.predict_proba(z, assets)
            models["gaussian_history"] = gaussian.to_dict()
            gx = np.column_stack([price_x, gaussian_bf])
            gf = fit_logistic(gx[val], yval, 1.)
            forecasts["gaussian_combined"] = logistic_predict(gf, gx)
            models["gaussian_combined"] = gf
            artifacts[key] = models
            for i in np.flatnonzero(test):
                record = {"ticker": rr[i]["ticker"], "date": rr[i]["date"], "asset": "BTC" if assets[i] == 0 else "ETH", "lead": lead, "epsilon": epsilon}
                for method in METHODS:
                    q = float(forecasts[method][i])
                    if not np.isfinite(q) or not 0 <= q <= 1:
                        raise RuntimeError(f"Invalid prediction for {method}")
                    record[method] = q
                predictions.append(record)
    write_csv(OUT / "predictions.csv", predictions)
    write_json(OUT / "fitted_models.json", artifacts)
    paths = [ROOT / "revision/empirical_protocol.md", ROOT / "revision/empirical_model_design.md", Path(__file__), ROOT / "code/empirical_mixture.py", ROOT / "code/collect_kalshi.py"]
    paths += [OUT / name for name in ("features.npz", "records.json", "label_availability.json", "labels_train.json", "labels_validation.json", "labels_test.json", "predictions.csv", "fitted_models.json")]
    write_json(OUT / "prediction_lock.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "numpy": np.__version__, "files": {str(p.relative_to(ROOT)): sha(p) for p in paths}, "test_scores_computed": False})
    print(f"Frozen {len(predictions)} test prediction records; no test scores computed.")


def score():
    lock = json.loads((OUT / "prediction_lock.json").read_text())
    for name, expected in lock["files"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Frozen artifact changed: {name}")
    labels = json.loads((OUT / "labels_test.json").read_text())
    rows = list(csv.DictReader((OUT / "predictions.csv").open()))
    summaries, comparisons, calibration, daily, by_asset, scored_rows = [], [], [], [], [], []
    for lead in LEADS:
        for epsilon in EPSILONS:
            rr = [r for r in rows if int(r["lead"]) == lead and float(r["epsilon"]) == epsilon]
            dates = np.array([r["date"] for r in rr])
            y = np.array([labels[r["ticker"]] for r in rr])
            assets = np.array([r["asset"] for r in rr])
            ref = losses(y, np.array([float(r["calibrated_price"]) for r in rr]), epsilon)
            recent_ref = losses(y, np.array([float(r["recent_calibrated_price"]) for r in rr]), epsilon)
            for method in METHODS:
                q = np.array([float(r[method]) for r in rr])
                metrics = losses(y, q, epsilon)
                summary = {"lead": lead, "epsilon": epsilon, "method": method, "n": len(y), "dates": len(np.unique(dates))}
                for metric, values in metrics.items():
                    mean, lo, hi = paired_date_bootstrap(values, dates)
                    summary.update({metric: mean, metric + "_lo": lo, metric + "_hi": hi})
                    if metric != "accuracy":
                        mean, lo, hi = paired_date_bootstrap(values - ref[metric], dates)
                        comparisons.append({"lead": lead, "epsilon": epsilon, "method": method, "reference": "calibrated_price", "metric": metric, "difference": mean, "lo": lo, "hi": hi})
                        mean_recent, lo_recent, hi_recent = paired_date_bootstrap(values - recent_ref[metric], dates)
                        comparisons.append({"lead": lead, "epsilon": epsilon, "method": method, "reference": "recent_calibrated_price", "metric": metric, "difference": mean_recent, "lo": lo_recent, "hi": hi_recent})
                        if method in ("mixture", "combined"):
                            ablation = losses(y, np.array([float(r[method + "_no_volume"]) for r in rr]), epsilon)
                            md, ld, hd = paired_date_bootstrap(values - ablation[metric], dates)
                            comparisons.append({"lead": lead, "epsilon": epsilon, "method": method, "reference": method + "_no_volume", "metric": metric, "difference": md, "lo": ld, "hi": hd})
                        for date in np.unique(dates):
                            ix = dates == date
                            for reference, reference_values in (("calibrated_price", ref), ("recent_calibrated_price", recent_ref)):
                                daily.append({"lead": lead, "epsilon": epsilon, "method": method, "reference": reference, "metric": metric, "date": date, "n": int(sum(ix)), "difference": float((values - reference_values[metric])[ix].mean())})
                summaries.append(summary)
                for asset in np.unique(assets):
                    ix = assets == asset
                    by_asset.append({"lead": lead, "epsilon": epsilon, "method": method, "asset": asset, "n": int(sum(ix)), **{metric: float(values[ix].mean()) for metric, values in metrics.items()}})
                bins = np.minimum((q * 10).astype(int), 9)
                for b in range(10):
                    ix = bins == b
                    calibration.append({"lead": lead, "epsilon": epsilon, "method": method, "bin": b, "n": int(sum(ix)), "mean_forecast": float(q[ix].mean()) if sum(ix) else "", "outcome_frequency": float(y[ix].mean()) if sum(ix) else ""})
            for r, yy in zip(rr, y):
                scored_rows.append({**r, "outcome": int(yy)})
    for filename, table in (("scores.csv", summaries), ("paired_differences.csv", comparisons), ("calibration.csv", calibration), ("daily_differences.csv", daily), ("asset_scores.csv", by_asset), ("scored_predictions.csv", scored_rows)):
        write_csv(OUT / filename, table)
    write_json(OUT / "scoring_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "prediction_lock_sha256": sha(OUT / "prediction_lock.json"), "bootstrap_seed": 20260929, "bootstrap_repetitions": 5000, "outputs": {p.name: sha(p) for p in OUT.glob("*.csv")}})
    print(json.dumps([s for s in summaries if s["epsilon"] == .001], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "predict", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "predict": predict, "score": score}[args.stage]()
