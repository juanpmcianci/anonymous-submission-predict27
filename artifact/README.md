# Artifact: endpoint sufficiency in prediction markets (anonymized)

Code, data and frozen artifacts for the PREDICT'27 submission.

- code/collect_kalshi.py: collection of the original cohort from Kalshi's public API (checksummed raw cache not included; metadata and candles are in code/empirical_data).
- code/empirical_mixture.py, code/empirical_evaluation.py: models and the prepare/predict/score pipeline. revision/ holds the protocol and model design written before collection.
- code/empirical_results: frozen fitted models, prediction lock, held-out predictions and scores.
- code/transfer_evaluation.py: scores the frozen models on any later cohort; --check reproduces the frozen test scores.
- code/feedback_diagnostics.py, code/oracle_endpoint.py, code/feedback_figures.py: date-level diagnostics, matched endpoint comparator, oracle endpoint study, stratified contamination study (code/contamination_stratified.py) and figures.
- code/revised_experiments.py and code/check_*.py: synthetic experiments and numerical checks of the theory.

Requirements: Python 3.10+, numpy, scipy, matplotlib.
- code/confidence_test.py: frozen matched test of the confidence mechanism (Sect. 5.4); code/confidence_power.py: its power and null simulation; code/run_transfer.sh: one-command later-cohort run.
- code/empirical_data_transfer, code/transfer_results: later cohort (2,370 contracts) and its frozen evaluation, produced by code/run_transfer.sh.
