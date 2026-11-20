#!/usr/bin/env bash
# Later-cohort evaluation (transfer protocol, amendments 1 and 2). Run from the project root
# in a terminal that can reach api.elections.kalshi.com / external-api.kalshi.com.
# Nothing is refitted. Results land in code/transfer_results/.
set -euo pipefail
PY=${PY:-python3}
$PY code/collect_kalshi_transfer.py --start 2026-09-28T01:00:00Z --stop 2026-11-17T00:00:00Z --output code/empirical_data_transfer
$PY code/transfer_evaluation.py --data code/empirical_data_transfer --start-date 2026-09-28 --stop-date 2026-11-17 --out code/transfer_results
$PY code/confidence_test.py evaluate --data code/empirical_data_transfer --start-date 2026-09-28 --stop-date 2026-11-17 --out code/transfer_results --label "later cohort (confirmatory)"
echo "Done. Send code/transfer_results/ back for the paper."
