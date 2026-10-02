# Multiserver-job systems (MSJ)

[🇷🇺 Русская версия](msj.ru.md) · [← Model catalog](../models.md)

![Multiserver-job diagram](../figures/msj.png)

**In plain words:** in modern datacenters and GPU clusters a single job often needs **several
servers at once** (cores, GPUs) for its whole run — unlike the classic M/M/c where one job uses one
server. A job waits until enough servers are simultaneously free. Service distributions,
resource fragmentation and the scheduling discipline jointly determine its delay.

### FCFS MSJ — exact response time (small systems)

**Description:** k servers; each job class has a **server need** (servers held simultaneously) and an
exponential rate. FCFS with head-of-line blocking: the oldest jobs fill the servers greedily and the
first job that does not fit stops the scan. Exact mean response time (overall and per class) via a
CTMC on the arrival-ordered job sequence — for small k / few classes / moderate load.

**Calculator class:** `MsjExactCalc` (`most_queue.theory.msj`) ·
**Simulator:** `MsjSim` (`most_queue.sim.msj`)

```python
from most_queue.theory.msj import MsjExactCalc, MsjClass

calc = MsjExactCalc(k=2, classes=[MsjClass(0.4, 1, 1.0), MsjClass(0.2, 2, 1.0)])
r = calc.run()          # r.v[0] overall mean sojourn; r.v_per_class per class
```

### Saturated MSJ — throughput and stability threshold

**Description:** The stability region of an MSJ system is characterised by its *saturated*
(always-backlogged) version. `MsjSaturatedCalc` solves the saturated CTMC exactly and returns the
throughput `X_sat`, which is the **maximum total arrival rate** the open system can sustain — it is
stable iff Λ < X_sat. Reproduces the product-form stability results of Grosof, Harchol-Balter &
Scheller-Wolf.

**Calculator class:** `MsjSaturatedCalc` (`most_queue.theory.msj`)

```python
from most_queue.theory.msj import MsjSaturatedCalc, MsjClass

sat = MsjSaturatedCalc(k=2, classes=[MsjClass(1.0, 1, 1.0), MsjClass(1.0, 2, 1.0)])
x_sat = sat.run()       # max sustainable total arrival rate (class mix from arrival ratios)
```

### General phase-type service — small FCFS systems

**In plain words:** a job still holds all its servers until it finishes, but its duration
can have several stages or a mixture of short and long modes. The first waiting job
blocks later jobs that might otherwise fit.

**Calculator:** `MsjPHCalc` (`most_queue.theory.msj`). This new API does not change
`MsjClass`, `MsjExactCalc`, `MsjSaturatedCalc` or `MsjSim`.

```python
from most_queue.random.map_ph import PHDistribution
from most_queue.random.utils.params import Cox2Params, ErlangParams
from most_queue.theory.msj import MsjPHCalc

services = [
    PHDistribution.from_erlang(ErlangParams(r=2, mu=2.0)),
    PHDistribution.from_cox(Cox2Params(mu1=2.0, mu2=1.5, p1=0.75)),
]
calc = MsjPHCalc(k=2, truncation=10, max_states=50000)
calc.set_sources([0.25, 0.15])
calc.set_servers([1, 2], services)
result = calc.run()
print(result.w_per_class, result.v_per_class)
print(result.boundary_mass, result.state_count, result.stationary_residual)
print(result.stability_threshold)  # saturated FCFS throughput for this class mix
```

The numerical solution is for a finite system admitting at most `truncation` jobs
in total. At the boundary arrivals are rejected. `throughput` is the **admitted**
rate; Little's law uses that rate. Increase the truncation and compare means and
boundary mass before using it as an open-queue approximation. Boundary mass alone
is not an error bound. Only first moments are returned in `w`/`v` (lists of length 1).

The calculator checks `sum(rates) < saturated_throughput()` for the open FCFS queue.
Resource load `sum(lambda_i * need_i * E[S_i]) / k < 1` alone is insufficient.
The threshold is not an EASY stability result. `max_states` bounds enumeration;
large server counts, phase counts or truncations can be impractical. PH must be
real, proper and transient; complex moment fits are rejected. General G is not
automatically fitted to PH by this calculator.

### General-service simulation and backfilling

**In plain words:** EASY allows a later job to use idle resources only when it does
not delay the **predicted** start of the first waiting job. It protects only that
head job, not every waiting job. Conservative backfilling reserves a start for
every waiting job; new jobs and calendar compression must respect all existing
reservations. Jobs remain nonpreemptive.

```python
from most_queue.sim.msj_general import MsjGeneralSim

source = MsjGeneralSim(k=2, seed=47)
source.set_sources([0.25, 0.15])
source.set_servers([1, 2], [(service, "PH") for service in services])
trace = source.make_trace(11000, estimates="oracle")  # explicit perfect information
for policy in ("fcfs", "easy", "conservative"):
    sim = MsjGeneralSim(k=2, discipline=policy)
    sim.set_servers([1, 2])  # replay does not need a service generator
    measured = sim.run_trace(trace, warmup_jobs=1000)
    print(policy, measured.v[0], measured.v_quantiles[0.99], measured.idle_with_queue)
```

`set_servers` accepts library `(params, notation)` pairs or a `sampler(rng)` callable,
e.g. `lambda rng: rng.lognormal(0, 0.8)`. `make_trace` accepts explicit `"oracle"`,
one positive duration estimate per class, or `None` for FCFS without predictions.
For measured/noisy estimates use immutable `MsjTraceJob(arrival, cls, service, estimate)`
records and `run_trace`. Arrivals must be nonnegative and ordered; durations must
be positive. A trace can represent dependent jobs and non-Poisson arrivals.

Actual service is sampled before scheduling and is used only by the completion
event engine. Forecast overruns do not kill jobs: new backfilling is suspended
while any running job is overdue. Conservative invalidates its infeasible current
calendar and temporarily uses FCFS, rebuilding reservations after overruns end.
The historical promises are retained. `reservation_violations` counts jobs that
start after their best recorded finite reservation (head jobs for EASY, all
reserved waiting jobs for conservative). `reserved_start_times` maps their
zero-based trace IDs to those best promises, including warm-up jobs. It is not a
history of all schedule revisions. Underestimated durations can violate promises;
oracle or upper-bound estimates preserve them. The guarantee concerns promised
start times, not dominance over FCFS or EASY in response time.

Conservative keeps a resource calendar and compresses one reservation at a time,
in planned-start order, without moving any other reservation later. It schedules
explicit wake-ups at reserved starts, even without an arrival/completion then.
This reference implementation uses repeated scans/sorts of the waiting calendar;
large backlogs can be expensive. The separate FCFS PH calculator does not provide
an analytical solution or stability threshold for either backfilling discipline.

`run(num_of_jobs, warmup_fraction=0.05, estimates=...)` generates an extra warm-up
cohort and measures exactly `num_of_jobs` arrivals. Replay drains all measured jobs.
Response moments and empirical p95/p99 cover the post-warmup arrival cohort;
time averages use the interval from its first arrival to the last input arrival,
excluding final draining. `utilization` is the occupied-resource fraction, and
`idle_with_queue` the idle-resource fraction while somebody waits. Quantiles are
empirical, not certified stationary SLOs. Missing class samples yield NaN; a
zero-length time window yields `None` time averages. Backfill/reservation counters
cover the whole input, including warm-up. Finite replay does not prove stability.

Methods, limitations and validation: [PH-MSJ methods](../msj_ph_methods.md).
Reproducible comparison: [experiment](../../examples/msj_backfilling_experiment.py).
Controlled-load FCFS/EASY/conservative comparison:
[experiment](../../examples/msj_conservative_experiment.py),
[protocol and results](../research/msj-conservative-results-2026-10.md).

### Runtime estimates from submission features

**In plain words:** learn from already completed historical jobs, predict a new
job's runtime from features available before service, and calibrate a statistical
upper estimate on a separate dataset. The scheduler uses that estimate, without
access to the actual duration. Statistical coverage does not protect every
reservation or every resource class, especially after a workload change.

`LogLinearRuntimePredictor` (`most_queue.sim.utils.runtime_prediction`) provides
`fit(X, service)`, `calibrate(X_cal, service_cal, coverage=0.95)` and
`predict(X, upper=False)`. Feed predictions into `MsjTraceJob.estimate`; the
existing schedulers do not change. Training, calibration and future evaluation
must be disjoint; feature provenance remains the caller's responsibility.
See the [complete example, formulas and limits](../msj_runtime_prediction.md),
[experiment](../../examples/msj_runtime_prediction_experiment.py), and
[results](../research/msj-runtime-prediction-results-2026-10.md).

Resource-group calibration is available through `calibrate_by_group(X_cal, S_cal, groups)`
and `predict(X, upper=True, groups=groups)`. The regression is shared; score
thresholds are separate for each observed, submission-time group. Insufficient
or unseen groups cause an explicit prediction error, never an automatic pooled
fallback. See the [group API and limits](../msj_runtime_prediction.md#calibration-by-resource-group)
and [EPIC-050 results](../research/msj-group-calibration-results-2026-10.md).
