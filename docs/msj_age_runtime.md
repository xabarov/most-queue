# Age-aware runtime forecasts from censored history

[MSJ API](models/msj.md) · [EPIC-051](epics/EPIC-051-msj-age-residual-runtime.md) ·
[Experiment and limitations](research/msj-age-runtime-results-2026-10.md)

In plain words: a job that has already run for a long time is not a fresh job.
Estimate its remaining duration conditional on having survived to its current
age. Historical jobs still running at the observation cutoff are censored, not
short completed jobs and not rows to discard.

## Model and data contract

`KaplanMeierRuntimeEstimator` in `most_queue.sim.utils.residual_runtime` fits one
population. Use one instance per class if censoring is independent only within
classes. Supply positive finite `observed_times = min(S, C)` and a matching boolean
`event_observed = (S <= C)`. The unobserved S of censored jobs is never an input.
All observations must be available before the model is used for scheduling.
This API does not verify independence, provenance or the historical cutoff.

The Kaplan–Meier product-limit estimator is
`G_hat(t) = product_{u <= t} (1 - d_u / n_u)`, where n is the risk set immediately
before u and d counts completions at u. Completions precede censoring at ties;
survival is right-continuous and means `P(S > t)`.
[Kaplan and Meier, JASA 1958](https://doi.org/10.1080/01621459.1958.10501452).

For age a with positive `G_hat(a)`, the residual survival is
`G_hat(a+r) / G_hat(a)`. The p-quantile is the first observed future time t with
`G_hat(t) <= (1-p)*G_hat(a)`, minus a. It is **not** the initial quantile minus a.
These are plug-in estimates, not split-conformal bounds or a finite-sample
conditional-coverage guarantee. The feature/conformal model in
[EPIC-049/050](msj_runtime_prediction.md) remains separate and unchanged.

| Method | Result and unavailable cases |
|---|---|
| `fit(times, completed)` | Fits/replaces the curve; invalid refits leave the previous fit unchanged. |
| `curve` | Tuple of immutable `KaplanMeierPoint(time, at_risk, events, censored, survival)` diagnostics. |
| `survival(age)` | KM survival; a flat terminal extension is an estimator convention, not evidence about the unseen tail. |
| `remaining_quantile(age, probability=0.9)` | Positive residual, or `None` if survival at age is zero or the required crossing is unobserved. |
| `remaining_mean(age)` | Integral of conditional survival; `None` for positive terminal survival or zero survival at age. |
| `remaining_mean(age, horizon=H)` | `E[min(S-a, H-a) | S>a]` estimate for H>a. No extrapolation beyond the last observation unless terminal survival is zero. |

Unrestricted and restricted means are different quantities. A zero terminal KM
survival makes the sample estimator's full mean computable, not the population
tail known exactly. Even an available estimate can be unstable in a sparse tail.

## Runnable example

```python
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.residual_runtime import KaplanMeierRuntimeEstimator

# Already available at time zero. False marks a still-running historical job.
small = KaplanMeierRuntimeEstimator().fit([1, 2, 3, 5], [True, False, True, True])
wide = KaplanMeierRuntimeEstimator().fit([2, 4, 6, 8], [True, False, True, True])
models = [small, wide]
initial = [model.remaining_quantile(0, 0.9) for model in models]
assert all(value is not None for value in initial)

def remaining(cls, age):
    return models[cls].remaining_quantile(age, 0.9)

# Actual services belong only to the event engine and subsequent evaluation.
trace = [
    MsjTraceJob(0, 0, 7, initial[0]),
    MsjTraceJob(1, 1, 3, initial[1]),
    MsjTraceJob(2, 0, 1, initial[0]),
]
sim = MsjGeneralSim(2, "conservative")
sim.set_servers([1, 2])
result = sim.run_trace(trace, remaining_predictor=remaining)
print(result.v[0], result.runtime_updates, result.reservation_violations)
```

`run(..., estimates=..., remaining_predictor=...)` also forwards the callback.
Initial positive estimates remain mandatory for backfilling. If a class has no
initial quantile, the caller must choose an explicit separate policy or decline
that comparison; the experiment declines it, never deleting unsupported jobs.

## Scheduling and promise semantics

`remaining_predictor(cls, age)` is opt-in for EASY/conservative and is rejected
for FCFS. With no callback, the previous static behavior is unchanged. The
callback receives only class and elapsed **service** time, not waiting time,
actual duration or future completion. It must return a finite positive residual
or `None`; malformed values raise `ValueError`. A callback can of course leak
data through its own closure: this interface is not a security boundary.

Updates occur for active jobs after all observed completions and arrivals at an
existing arrival/completion/conservative-reservation event, before dispatch.
Completed jobs are not predicted. Newly starting jobs use their initial estimate;
waiting jobs are not re-estimated. There is no independent forecast-expiry timer,
heartbeat, online history update or refitting. EASY and conservative may therefore
refresh at different event times. This is an event-driven policy choice, not a
reproduction of the correction heuristics in
[Tsafrir, Etsion and Feitelson, TPDS 2007](https://doi.org/10.1109/TPDS.2007.70606).

A finite updated end is `now + residual`. Any extension invalidates the current
conservative reservation calendar before it is rebuilt. A shorter forecast allows
normal compression. **Historical best promises are never erased.** A later actual
start can still violate them even after rebuilding the calendar.
The static upper-bound reservation guarantee does not extend to this reset
policy: even a conservative upper forecast can permit new backfills that cross
an old promise after its extension. Revisions are audited, not guaranteed safe
with respect to every historical promise.

`None` marks an unknown release, suspends new backfills and permits FCFS-prefix
starts with physically available resources. No jobs are killed or censored by
the replay. A positive residual too small to advance the floating-point timestamp
uses the same safe unknown-release path. This can occur at a discrete KM atom
after subtraction of large timestamps; inventing a later release is avoided.
Overflow is an error. Full traces are drained even when every forecast is unknown.

New counters in `MsjSimulationResults`, including warmup:

- `runtime_updates`: callback evaluations, not distinct jobs or changed endpoints.
- `unavailable_runtime_updates`: `None` **or sub-timestamp-resolution** results.
- `forecast_calendar_resets`: nonempty calendars invalidated by refresh; not all
  overrun recovery operations in the static scheduler.

`reserved_start_times` retains each job's best historical finite promise, not the
whole revision sequence. `reservation_violations` counts jobs starting after that
promise, with the existing relative tolerance. Promise sums and maximum lateness
in the pilot are additional diagnostics, not proof of harm caused by backfill.
Reservations under EASY and conservative apply to different sets of jobs.

## Validation and scope

Exact risk tables, ties, complete/all-censored samples, residual integrals,
unsupported tails, lifecycle and malformed inputs are tested. Random tied curves
are compared against the independent
[SciPy 1.13 `ecdf(CensoredData(...))` implementation](https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.stats.ecdf.html).
Scheduler tests cover forecast extensions, unknown tails, old promises, capacity,
observed-prefix causality and callback arguments. Both disciplines match the
M/E2/1 limit with all-resource jobs, shared test tolerances and exact FCFS starts.

No informative censoring, left truncation, competing risks, drift adaptation,
survival regression on features, certified queueing tails or real-cluster trace
validation is claimed. See the [352-run pilot](research/msj-age-runtime-results-2026-10.md)
for gains, class-level tradeoffs and explicit tail unavailability.
