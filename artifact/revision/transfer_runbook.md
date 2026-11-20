# Running the later cohort

Kalshi's API is not reachable from Claude's environments. Run this once in a normal terminal on your Mac, from the project root (`BIP`), with the Python that has numpy and scipy:

```sh
PY=/opt/anaconda3/bin/python3 bash code/run_transfer.sh
```

It collects contracts closing from 2026-09-28 01:00 UTC to 2026-11-17 00:00 UTC (amendment 2 moves the start), scores every frozen model of the original protocol and amendment 1, and runs the frozen confidence test of amendment 2. Nothing is refitted. Raw responses are cached, so rerunning resumes.

Do not open or plot the results before handing them back, so the paper can report them as confirmatory. If the collector stops with "historical cutoff moved into cohort", send the message to Claude.

Check that the scoring code is faithful (passed, 1.7e-16):

```sh
python3 code/transfer_evaluation.py --data code/empirical_data --start-date 2026-09-14 --stop-date 2026-09-28 --out code/transfer_check --check
```
