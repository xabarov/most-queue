# Resource group runtime calibration pilot

[EPIC-050](../../docs/epics/EPIC-050-msj-group-runtime-calibration.md),
[API](../../docs/msj_runtime_prediction.md#calibration-by-resource-group),
[protocol and results](../../docs/research/msj-group-calibration-results-2026-10.md).

Run from the repository root:

```bash
.venv/bin/python -m examples.msj_group_calibration_experiment --shape erlang2 --jobs 4000 --replications 12 --output works/msj_group_calibration/erlang2.json
.venv/bin/python -m examples.msj_group_calibration_experiment --shape lognormal_hetero --jobs 4000 --replications 12 --output works/msj_group_calibration/lognormal.json
```

Each file contains 336 scheduler runs, 28 queue summaries, 72 prediction runs,
six prediction summaries, 12 calibration records, and 48 calibration-scarcity
records. Seeds 50000–50011 are new relative to EPIC-049. The data law is reused,
not its observations. Independent child streams generate train/calibration/test
and arrival gaps. Train has 1500 completed jobs, calibration 1000, replay 400
warm-up and 4000 measured jobs. All historical labels are available before t=0.

Pooled and group calibration share one fitted model and calibration cohort;
FCFS/EASY/conservative see identical actual work. Group labels are requested
server counts K=1,3,4, known at submission. Slowdown doubles test S and halves
lambda, without retraining/recalibration. Both modes use the same target 95%.

The scarcity sweep uses nested calibration prefixes of size 60/100/300/1000
and unchanged train/test. It is iid prediction-only, not an experiment that
discards unsupported jobs from a queue. A null group coverage/estimate ratio
means no finite prediction is available; it is neither zero nor successful
coverage. Availability is recorded separately, including unseen groups.

Environment: Python 3.12.3, NumPy 2.2.5, SciPy 1.13.0. Array-byte SHA-256
fingerprints support reproduction in this environment, not a leakage proof.
Intervals are unadjusted Student 95% over independent seed bundles; paired
contrasts compare modes within one discipline/seed. They are not population-p99
bounds, simultaneous coverage, stationarity evidence or reservation guarantees.
The earlier EPIC-049 artifacts remain unchanged.
