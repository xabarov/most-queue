# Feature based runtime prediction for MSJ backfilling

[MSJ API](models/msj.md) · [EPIC-049](epics/EPIC-049-msj-runtime-prediction-calibration.md) ·
[EPIC-050](epics/EPIC-050-msj-group-runtime-calibration.md)

`LogLinearRuntimePredictor` learns positive runtime estimates from completed
historical jobs. Its prediction interface accepts only submission-time features,
not actual service times. Use its estimates with the existing EASY/conservative
trace replay. The model is deliberately small and uses NumPy/SciPy already
required by the project; it is not a production-trained cluster predictor.

## Fit calibrate and replay

```python
from dataclasses import replace
import numpy as np

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor

rng = np.random.default_rng(49)
# Historical training jobs, held-out calibration jobs, then future test jobs.
# All features below are observable BEFORE service starts.
x_train = rng.normal(size=(1000, 1))
s_train = rng.lognormal(0.5 * x_train[:, 0], 0.5)
x_cal = rng.normal(size=(500, 1))
s_cal = rng.lognormal(0.5 * x_cal[:, 0], 0.5)
x_test = rng.normal(size=(200, 1))

predictor = LogLinearRuntimePredictor().fit(x_train, s_train)
diagnostics = predictor.calibrate(x_cal, s_cal, coverage=0.95)
point = predictor.predict(x_test)
upper = predictor.predict(x_test, upper=True)
print(diagnostics.rank, diagnostics.samples)  # 476, 500

# Only the event engine/evaluation stage gets to see these actual runtimes.
actual = rng.lognormal(0.5 * x_test[:, 0], 0.5)
arrivals = np.cumsum(rng.exponential(3.0, size=len(actual)))
trace = tuple(MsjTraceJob(float(a), 0, float(s)) for a, s in zip(arrivals, actual))
estimated_trace = tuple(replace(job, estimate=float(u)) for job, u in zip(trace, upper))
sim = MsjGeneralSim(2, discipline="conservative")
sim.set_servers([1])
result = sim.run_trace(estimated_trace, warmup_jobs=20)
print(result.v[0], result.reservation_violations)
```

Use the same column meanings/encoding at training, calibration and prediction.
The feature matrix must be finite, real, two-dimensional and have at least one
row. Zero columns allow an intercept-only model. Fit requires at least two jobs;
constant or collinear columns are accepted through least squares. Durations must
be a finite positive vector matching the row count. A successful refit clears
both pooled and group calibrations; a failed fit/calibration leaves the previous valid model unchanged.
`calibration` exposes a frozen `RuntimeCalibration` record with coverage, sample
count, rank and log margin. Empty prediction batches are rejected.

## Regression and calibration

Fit standardizes columns using training data only, adds an intercept and solves
OLS for log S. With fitted log predictor g(x), the point estimate is
`exp(g(x)) * mean_train(exp(log(S_i) - g(X_i)))`. The smearing correction is also
training-only. It is a conditional-mean approximation when the residual law is
appropriate, not a general guarantee, especially under heteroscedasticity.

For an independent calibration set, compute signed scores
`r_i = log(S_i) - log(point(X_i))`. If there are n calibration jobs, let
`k = ceil((n+1)*coverage)` and select the k-th smallest score, without interpolation.
Return `upper(x) = exp(log(point(x)) + r_(k))`. This is a one-sided specialization
of the split-conformal rank construction; it does not use absolute residuals.
The margin may be negative: “upper” refers to a probability bound, not necessarily
a value above the point estimate. Floating-point upper outputs are rounded
outwards to avoid artificial misses at ties; nonrepresentable outputs raise.

When k>n, the corresponding conformal bound is infinite. The predictor raises
`ValueError` because MSJ replay requires finite estimates; it does not substitute
the sample maximum or silently lower the requested coverage. For example, 95%
needs at least 19 calibration jobs, and 99% needs at least 99. More data are
usually needed for useful precision. Exponential overflow/underflow and invalid
feature extrapolation are also explicit errors, not clipping policies.

The statistical basis is [Lei et al., JASA 2018](https://doi.org/10.1080/01621459.2017.1307116)
and the arbitrary-score construction in
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).
Our particular log-runtime score and smearing regression are baseline design
choices, not a reproduction of a published runtime-prediction algorithm.

## Calibration by resource group

**In plain words:** a single overall coverage rate can hide systematic misses for
jobs that need many servers. Keep the same runtime regression but estimate a
separate correction from completed jobs in each resource group. The group must
be known at submission; it is not a category inferred from the future duration.

```python
import numpy as np
from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor

rng = np.random.default_rng(50)
# Independent historical train and calibration, then submission features only.
g_train = rng.choice([1, 4], size=1000)
x_train = np.column_stack((g_train == 4, rng.normal(size=1000)))
s_train = rng.lognormal(x_train[:, 0] + 0.5 * x_train[:, 1], 0.5)
g_cal = rng.choice([1, 4], size=500)
x_cal = np.column_stack((g_cal == 4, rng.normal(size=500)))
s_cal = rng.lognormal(x_cal[:, 0] + 0.5 * x_cal[:, 1], 0.5)
g_test = np.array([4, 1, 4])
x_test = np.column_stack((g_test == 4, [0.2, -0.1, 0.8]))

model = LogLinearRuntimePredictor().fit(x_train, s_train)
model.calibrate(x_cal, s_cal, coverage=0.95)
info = model.calibrate_by_group(x_cal, s_cal, g_cal, coverage=0.95)
pooled = model.predict(x_test, upper=True)
grouped = model.predict(x_test, upper=True, groups=g_test)
assert all(record.is_finite for record in info.values())
# Use grouped[i] as MsjTraceJob.estimate; never pass actual test S to predict.
print({group: (record.samples, record.rank) for group, record in info.items()})
```

Group labels are a one-dimensional vector of nonnegative integers, one per feature
row; floats, strings and booleans are rejected. Labels need not be contiguous or
start at zero. For MSJ, either stable class IDs or the requested server counts
can be used, provided the meaning is consistent at calibration and prediction.
Fix the partition before examining calibration/test outcomes. Do not choose the
grouping or target coverage by searching for favorable test results.

For group g, use only its n_g signed log residuals and the rank
`k_g = ceil((n_g+1)*coverage)`. The forecast uses that group's score threshold,
with the same outward rounding and overflow checks as pooled prediction.
This is the known group-balanced/Mondrian split-conformal construction:
[Angelopoulos and Bates, §4.1](https://arxiv.org/html/2107.07511v6#S4.SS1).
Resource classes are observed X-groups, not unknown classification targets Y.

`calibrate_by_group` returns a read-only snapshot mapping observed labels to
frozen `RuntimeGroupCalibration` records: `coverage`, `samples`, `rank`,
`log_margin`, and the `is_finite` property. It replaces the previous group map
but does not refit the regression or change pooled calibration. Likewise,
`calibrate` changes only the pooled record. A successful `fit` clears both.
The initial `group_calibrations` map is empty; invalid updates preserve valid state.

| Request or data condition | Behavior |
|---|---|
| `predict(X)` | Point forecast; no calibration needed |
| `predict(X, upper=True)` | Explicit pooled mode; pooled calibration required |
| `predict(X, upper=True, groups=g)` | Explicit grouped mode; matching group calibration required |
| `groups` supplied without `upper=True` | `ValueError`; no silently ignored labels |
| Observed group with k_g>n_g | Diagnostics: `log_margin=None`, `is_finite=False`; predicting it raises |
| Group not observed in calibration | No record; predicting it raises, even if a pooled bound exists |
| Mixed prediction batch containing unsupported groups | Whole call raises; no partial result or hidden fallback |

At 95% each group needs at least 19 calibration jobs (99% needs 99), not 19 across
the full dataset. A finite rank does not ensure useful precision or prevent
numerical overflow. `None` denotes a theoretically infinite bound that this finite
runtime API does not return. Other adequately represented groups remain usable.
An application must explicitly decide what to do with unsupported jobs: gather
data or choose another scheduling policy with its own stated information limits.
This library does not drop such jobs, merge groups or substitute a pooled claim.

Coverage is group-conditional only under exchangeability within the fixed group
and an independently fitted score. It still averages over the calibration sample
and future observation within that group, not over arbitrary scheduler-selected
subsets. It does not imply coverage at every feature value, for a fixed realized
calibration set, or simultaneously for every job. Small-group statistical
limitations are also discussed by [Dewolf et al., §3.3](https://arxiv.org/html/2309.08313v2#S3.SS3).
Group calibration alone does not repair within-group workload drift or turn
estimated runtimes into hard limits on service or reservation violations.

## Information and guarantee boundaries

The pooled calibration statement concerns a new exchangeable observation and a model
trained independently of calibration/test data. Coverage is marginal over the
calibration sample and new observation. It does **not** promise any of the following:

- Coverage at the target within every resource class or feature subgroup.
- That a particular fixed calibration set covers exactly that test fraction.
- That every job of a trace stays below its estimate.
- That reservations, stationary p99 or queueing SLOs are satisfied.
- Validity after workload drift or with nonexchangeable/censored historical data.

Callers are responsible for disjoint splits, historical availability of labels,
consistent feature encoding and absence of target leakage. The API cannot detect
a true runtime disguised as a feature or the same observations copied into two
arrays. Select completed historical cohorts carefully: taking only jobs that
finished recently can bias the duration distribution. Hyperparameter selection
must not use calibration/test labels; recalibrate on fresh data after refitting.

The existing stochastic `ExpNoiseSimPredictor`/`LognormalNoiseSimPredictor` for
size-based queues sample a prediction conditional on true size. They remain
unchanged and model noisy-oracle information; this feature-only model is separate.
EPIC-049 added pooled calibration; EPIC-050 adds resource-group calibration.
No online age correction, adaptive drift handling or real-trace importer is included.

## Validation and experiment

Tests cover exact log-linear recovery, intercept-only smearing, signed order
statistics, exhaustive exchangeable ranks, insufficient sample sizes, ties,
outward rounding, lifecycle/overflow and information separation. EASY and
conservative with trained upper estimates are checked against M/E2/1 when every
job occupies the whole resource pool, using the project's shared tolerances.

The [experiment](../examples/msj_runtime_prediction_experiment.py) uses independent
training/calibration/test streams, two service-noise laws, two offered resource
loads and an explicit unseen slowdown. Class means are trained too. All schedulers
within a scenario see identical actual jobs and the same estimates. Read the
[protocol and results](research/msj-runtime-prediction-results-2026-10.md) before
interpreting its point accuracy, coverage or queueing comparisons.

The [grouped experiment](../examples/msj_group_calibration_experiment.py) uses new
seed bundles and the same service/feature laws to compare pooled95/grouped95/oracle
on common traces. It records all-class latency/coverage, paired differences and
a separate nested calibration-size sweep. Unsupported groups are reported, not
removed from a queue comparison. See its
[protocol and results](research/msj-group-calibration-results-2026-10.md).
