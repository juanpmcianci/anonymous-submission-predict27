# Transfer evaluation protocol (later cohort)

Written 17 November 2026, before any data for the transfer cohort were collected. Not an external preregistration.

## Purpose

The original retrospective study used 14 test dates from one eight-week window. This addendum tests the same fitted models on a later, untouched period.

## Frozen models

No model is refitted, retuned or recalibrated. The fitted models are those in `code/empirical_results/fitted_models.json` (SHA-256 `3dac814693ea32fbc1dbd530578b8b02ed4a830875651b0d6f5fd0778382d078`), fitted on the 3-30 August training window and the 31 August-13 September validation window and locked on 28 September 2026. The training-frequency prior is the mean of `code/empirical_results/labels_train.json`.

## Cohort

KXBTC15M and KXETH15M contracts with scheduled close at an exact UTC hour in [2026-09-28 00:00 UTC, 2026-11-17 00:00 UTC) (start moved to 01:00 UTC by amendment 2), collected with `code/collect_kalshi_transfer.py` under the original eligibility rules (900-second window, binary finalized outcome, one contract per series and hour). The nominal maximum is 2,400 contracts on 50 UTC dates. All eligible contracts are used. No contract is selected, dropped or reweighted using volume, outcome or forecast error.

## Forecasts and metrics

Unchanged from `empirical_protocol.md`: five-minute and one-minute leads, a 60-second availability buffer, clipping at 0.001 (primary) and 0.005 (sensitivity), all twelve methods, Brier loss primary and log loss secondary, paired differences with 95% percentile intervals from 5,000 bootstrap resamples of whole UTC dates (seed 20260929).

## Amendment, 18 November 2026 (before collection)

Adds one frozen comparator, the matched endpoint logistic E_log in `code/empirical_results/endpoint_matched.json` (SHA-256 `ecb93150bedd50c3d98baa4c3331ac179b8ef927a2b2bdaa17369b0762509253`). Its inputs are the history-logistic inputs without the three path features: initial and latest logit, mean log volume, log-volume slope, asset. It was fitted on the original training window with the penalty chosen on the validation window, after the original test scores had been seen. No transfer data existed when it was frozen. Contrast 4 below is added.

## Pre-specified contrasts

1. Five-minute Brier loss, mixture minus training-fitted price calibration.
2. Five-minute Brier loss, price plus mixture minus validation-fitted cubic price calibration.
3. The same two contrasts at the one-minute lead.
4. Brier loss, history logistic minus matched endpoint logistic, at both leads.

All other comparisons are descriptive. Results are reported whatever their sign. Intervals are unadjusted for multiple comparisons.

## Checks

`code/transfer_evaluation.py --check` on the original data restricted to the original test dates must reproduce `code/empirical_results/scores.csv`, `paired_differences.csv` and their intervals. It did so to a maximum absolute deviation of 1.7e-16 before collection.

## Decision rules

Proceed regardless of direction. No change to models, features, dates or clipping after seeing transfer scores. Any later model change needs a further untouched cohort. Collection failures are documented as feasibility amendments before scoring, and the original eligibility rules stay in force.

## Amendment 2, 18 November 2026 (before collection, before any evaluation of the new comparison)

### Exposure
No transfer-period contract has been collected or scored. One overlap exists: the original candidate listing (`code/empirical_data/candidate_markets.json`) contains `KXBTC15M-26SEP272000-00` and `KXETH15M-26SEP272000-00`, which close at 2026-09-28 00:00 UTC, the first hour of this window. Their outcome fields were downloaded with the listing, and one was displayed during the exposure check on 18 November. Both are excluded: the cohort now starts at 2026-09-28 01:00 UTC. No other transfer-period data exist in the project.

### Unchanged frozen replication
All models, preprocessing, contrasts and metrics of the original protocol and amendment 1 are applied unchanged.

### New comparison: the confidence mechanism
Specified in `code/confidence_test.py` and frozen in `code/empirical_results/confidence_models.json` (SHA-256 `486b1df544a858a4364bdda621e99a82b46f59c407e770239a3b7106491493a5`), fitted on the training window with penalties chosen on the validation window, before the comparison was evaluated on any test or transfer data.

- Endpoint model: penalized logistic regression on b = (x_0, asset-specific cubic in x_T, log(1+v_k) for every retained candle).
- History model: the same, plus phi = S log(1 + E/c), S = x_T - x_0, E = sum_t (z_t - S/T)^2, c the training median of E. The coefficient is unconstrained.
- The unknown-scale result predicts a negative coefficient. Training fits: -0.0455 (five minutes), +0.514 (one minute), clipping 0.001. (Rounding of the first value corrected from -0.046 on 18 November; no substantive change.)

Gain G = R(endpoint model) - R(history model); positive favours the history feature.

- Primary: five-minute Brier gain, clipping 0.001.
- Secondary: five-minute log-loss gain.
- Robustness: the same at the one-minute lead and at clipping 0.005.
- Descriptive: gains by asset, by UTC date, and by training-defined tertiles of E.
- Absolute losses of both models are reported next to the frozen P_lin and P_cub.

Smallest worthwhile gain: delta = 0.001 Brier, about 0.8% of the five-minute baseline loss and twice the validation-window gain from recalibrating the raw last price (0.1318 to 0.1314).

Intervals: 95% percentile intervals from 5,000 resamples of whole UTC dates, both assets together, seed 20260929. Sensitivity, fixed now: circular blocks of 3 consecutive dates.

Reading of the primary interval [lo, hi]:
- lo > delta: improvement exceeding delta.
- 0 < lo <= delta: improvement.
- hi < 0: the history feature worsens prediction.
- -delta < lo and hi < delta: practically equivalent within delta.
- otherwise hi < delta: gains of delta or more excluded.
- otherwise: inconclusive.

The original test window (14-27 September) was used for earlier comparisons. The new comparison is also evaluated there, after this amendment, and is reported as exploratory. Only the transfer cohort is confirmatory.
