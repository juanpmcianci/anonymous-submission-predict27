# When Does Price History Add Information? Endpoint Sufficiency in Prediction Markets

Anonymous submission to PREDICT'27. This repository contains the paper source and the code, data and frozen artifacts needed to reproduce its results.

## Summary

A prediction-market price is a free forecast, so any method that uses the market's history must beat it. The paper asks when the price history adds outcome information beyond the latest observed price. It gives sufficiency criteria under Gaussian increments and finite regime mixtures, a Brier and log-loss decomposition of intrinsic value versus approximation and estimation error, and a frozen retrospective study of Kalshi BTC and ETH 15-minute contracts (2,650 original and 2,370 later-cohort contracts).

## Layout

| Path | Contents |
|---|---|
| `main.tex`, `*_appendix.tex`, `supporting_bounds.tex` | Paper source (LLNCS format: `llncs.cls`, `splncs04.bst`, `references.bib`) |
| `main.pdf` | Compiled paper |
| `generated/` | Tables, macros and figures produced by the code and included by the paper |
| `artifact/` | Code, data and frozen results (see `artifact/README.md` for a file-by-file description) |
| `artifact/code/` | Data collection, models, evaluation, synthetic experiments, numerical checks |
| `artifact/revision/` | Protocols and model design written before collection and before the later cohort |

## Requirements

Python 3.10+ with `numpy`, `scipy`, `matplotlib`. A LaTeX distribution is needed to rebuild the paper.

## Reproducing

Build the paper:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Re-score the frozen models on the held-out test set (reproduces the reported scores):

```bash
cd artifact/code
python transfer_evaluation.py --check
```

Run the later-cohort evaluation end to end:

```bash
cd artifact/code
bash run_transfer.sh
```

Synthetic experiments and numerical checks of the theory are in `revised_experiments.py` and `check_*.py`; the matched confidence test is `confidence_test.py` (power and null simulation in `confidence_power.py`).

Raw API caches are not included. Contract metadata and candles are in `artifact/code/empirical_data` and `artifact/code/empirical_data_transfer`.
