# EPIC-051: censored history and age-aware MSJ forecasts

[Protocol and results](../../docs/research/msj-age-runtime-results-2026-10.md) ·
[API and mathematical contract](../../docs/msj_age_runtime.md)

From the repository root:

```bash
.venv/bin/python -m examples.msj_age_runtime_experiment --shape erlang2 --jobs 4000 --replications 8 --output works/msj_age_runtime/erlang2.json
.venv/bin/python -m examples.msj_age_runtime_experiment --shape lognormal_hetero --jobs 4000 --replications 8 --output works/msj_age_runtime/lognormal.json
```

Environment: Python 3.12.3, NumPy 2.2.5, SciPy 1.13.0. New seeds 51000–51007;
independent streams for history/test/arrival gaps/censoring. Historical cohort
4000, replay 400 warmup + 4000 measured arrivals, quantile 0.90. Censoring exposure
is independent exponential with mean five class service means. The synthetic
service law includes the EPIC-049 log-size mixture; curves condition on class only.
Older EPIC-049/050 artifacts are not overwritten or used as paired baselines.

Each JSON contains 176 complete scheduler runs, 144 landmark records, eight
history diagnostics with fingerprints, 22 queue summaries and 18 landmark
summaries. Across both files: 352 runs and 288 landmark records. Queue differences
are paired by seed, with age minus the same estimator's fixed prediction;
completed-fixed/oracle use KM-fixed as their reference. All 4400 jobs are drained.
Latency and landmarks exclude warmup, promise/update counters include it.

Intervals are Student 95% across eight independent bundles, without multiplicity
correction. Landmark intervals condition on available forecasts; their available
seed counts are explicit. `null` is unavailable, never zero or successful
coverage. A p99 summary is an average of finite-run empirical p99s, not a certified
population percentile. Finite replay does not establish stability or an SLO.

Promise sums/max use best historical finite promises and are zero if none were
made. Unavailable update counts include both unknown statistical tails and
positive residuals below floating-point timestamp resolution. The resource load
is matched from the data law, not rescaled to equalize realized work.
