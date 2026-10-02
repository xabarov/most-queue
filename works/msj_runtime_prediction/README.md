# Feature based runtime prediction pilot

[EPIC-049](../../docs/epics/EPIC-049-msj-runtime-prediction-calibration.md),
[API and assumptions](../../docs/msj_runtime_prediction.md),
[protocol and interpretation](../../docs/research/msj-runtime-prediction-results-2026-10.md).

Run from the repository root:

```bash
.venv/bin/python -m examples.msj_runtime_prediction_experiment --shape erlang2 --jobs 4000 --replications 6 --output works/msj_runtime_prediction/erlang2.json
.venv/bin/python -m examples.msj_runtime_prediction_experiment --shape lognormal_hetero --jobs 4000 --replications 6 --output works/msj_runtime_prediction/lognormal.json
```

Each artifact contains 216 scheduler runs, 36 queue summaries, 48 prediction
evaluations, eight prediction summaries and six fit/calibration records with
cohort fingerprints. Seeds 49000–49005 spawn separate train/calibration/test/arrival
streams. Each historical train cohort has 1500 jobs, calibration 1000; replay
has 400 warm-up and 4000 measured jobs, all drained. Historical cohorts are
assumed fully completed before time zero. No current-job duration is used by
the non-oracle predictor. These are synthetic data, not measured cluster traces.

Environment: Python 3.12.3, NumPy 2.2.5, SciPy 1.13.0. SHA-256 fingerprints of
array bytes are for reproduction in this environment, not a data-leakage proof.
All policies within a scenario receive common jobs/forecasts. Slowdown doubles
test service and halves its arrival rate, preserving the offered resource load;
it does not refit or recalibrate. Intervals are unadjusted Student 95% intervals
over independent seed bundles, not population p99 bounds or scheduling guarantees.
