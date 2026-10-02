# Initial state, cancellation and runtime-limit sensitivity

EPIC-057 tests three omissions of the completed-job studies: empty initial state,
cancelled resource use, and hard runtime limits. It keeps the same completed
target cohort and existing six policies. It does not reconstruct the original
cluster or infer unobserved completion demand from a cancelled job.

EPIC-058 adds [Acme/Kalos ingestion](modern_gpu_trace.md) and optional
failed/node_failed/recorded timed_out labels to the same lifecycle API. A recorded
timeout does not imply a known runtime budget. The EPIC-057 protocol and saved
artifacts remain tied to their original implementation at commit `99b8500`;
ordinary replay and existing completed/cancelled behavior are unchanged.

## Source evidence and missing information

The pinned SDSC SP2 file and reuse conditions are unchanged from
[EPIC-055](real_trace_calibration.md). Credit: SDSC / Victor Hazlewood; conversion
Dror Feitelson; DIR-LAB version-pinned mirror. Raw data is kept in the ignored cache,
not redistributed with code. See the [source and usage agreement](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html).

The [SWF specification](https://www.cs.huji.ac.il/labs/parallel/workload/swf.html)
defines runtime as wall-clock running duration and wait as delay until start.
Status=5 can represent cancellation before or during execution; status alone
does not reliably distinguish user cancellation, failure and system killing.
The [original JOBLOG field description](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/format.txt)
describes requested resource time, including wall-clock time for parallel jobs.
Neither source establishes exact hard-limit enforcement for every recorded job.

`parse_swf_lifecycle(lines, capacity)` in `most_queue.sim.utils.workload_lifecycle`
first applies the unchanged strict SWF validation and retains its completed-only
training trace. It additionally admits status=5 with positive runtime, known wait,
rigid requested=allocated K within capacity, and no predecessor. The input must
contain a completed reference cohort. It preserves stable source order at ties.

On the pinned file, 43 117 completed and 10 917 cancelled jobs meet these rules;
5681 cancelled records with unknown/nonpositive runtime are excluded. Their
consumption and cancellation time are not set to zero or reconstructed from wait.
Positive requested time is retained; nonpositive is unavailable. Audit fields
record mutually exclusive exclusions and request availability/overruns among
accepted jobs. A status=5 runtime is consumed occupation up to the recorded end,
not a right-censored observation with an established independent censoring law.
There is no Kaplan–Meier fit or assumption that full latent service equals runtime.

## Retrospective carry-in boundary

Each origin uses the completed-only training cutoff and first 1200 eligible later
completed jobs from EPIC-056. The replay boundary is the **first selected submit**,
which can be later than the training cutoff. Target IDs are the last 1000 of those
1200; extra cancelled arrivals never alter this warmup or target membership.

`observed_snapshot(source, at, include_cancelled=False)` selects jobs submitted
strictly before `at` and ending strictly after it. A historical start at or before
`at` is running; a later start is waiting. A terminal event exactly at `at` leaves
no carry-in, and arrivals exactly at `at` are new arrivals. Known initial resource
use above capacity is an error, not silently clipped. The snapshot is partial:
unknown/zero cancellation records and jobs removed by upstream cleaning are absent.

Running jobs retain observed K, elapsed service age and observed remaining
occupation. Waiting jobs retain full observed runtime and source submit order,
but their old wait is not added to target delay. Their future starts are decided
by the chosen policy. Historical priority, reservations and Adaptive Quickswap's
internal phase are not recovered. Policy memory starts fresh.

The running forecast is `max(historical p90 - age, 0)`. An unknown/expired estimate
marks the job overdue; the actual remaining duration is not substituted into
the forecast. Actual residuals and inclusion of jobs with future outcomes are
retrospective state inputs, not features available to a duration predictor.
They are held fixed across workload models and seeds. Thus model errors are
conditional on this supplied environment, not errors of predicting that environment.

## Four fixed scenarios

| Scenario | Initial state | New cancelled jobs | Hard runtime budgets |
|---|---|---|---|
| `empty_completed` | Empty | None | None |
| `carry_completed` | Observable status=1 running/waiting | None | None |
| `carry_cancelled` | Observable status=1 and 5 running/waiting | Positive-runtime status=5 within the selected 1200-job arrival span | None |
| `requested_limit` | Same as carry_cancelled | Same as carry_cancelled | Positive recorded request for new arrivals only |

New cancelled arrivals include both endpoints of `[first selected submit,
last target submit]`, including the 200-job warmup portion. The utilization window
instead begins at the first of the 1000 measured targets and ends at the last.

Initially waiting and newly arriving cancelled jobs consume their recorded
positive runtime **after their simulated start**, then release resources with a
cancelled label. Changing the scheduler
therefore moves the cancellation event with its service start. This is a specified
service-clock scenario, not an assertion that users cancel after a fixed amount
of execution or that an original absolute cancellation time survives rescheduling.
Already-running carry-in jobs are the exception: they retain their recorded
remaining occupation and therefore their original release relative to the boundary.
Unknown waiting cancellations are not simulated; they could affect queue order
and reservations even if they used no resources in the original log.

The limit scenario releases a new job at
`simulated_start + min(service, requested_time)`, not relative to submission.
If service exceeds the limit, its label is `timed_out`, including an earlier
timeout of a recorded cancelled job. Equality preserves completed/cancelled.
There are no retries, resumed attempts or invented successful completions.
Initial jobs are grandfathered: no retrospective cut of their historical service
or instantaneous removal is applied. Missing requested times remain unlimited.

The hard budget is a hypothetical wall-clock cap, not a claim about the original
cluster's enforcement. Backfilling forecasts stay unchanged even in this scenario;
they are not tightened to the known cap. This isolates the occupation/termination
intervention, not the best cap-aware scheduling algorithm.

## Replay API and invariants

`MsjLifecycleSim` in `most_queue.sim.msj_lifecycle` inherits server configuration
and the existing dispatchers from `MsjGeneralSim`. The opt-in `run_lifecycle` API
does not modify ordinary `run_trace`, its types, or previous artifact hashes.
Supported policies are FCFS, FirstFit, MSF, Adaptive Quickswap, EASY, conservative.
Preemptive policies and arbitrary cancellation deadlines are outside this API.

```python
from most_queue.sim.msj_lifecycle import MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim

sim = MsjLifecycleSim(2, "fcfs")
sim.set_servers([1, 2])
result = sim.run_lifecycle(
    [MsjLifecycleJob(0, 1, 10, runtime_limit=2)],
    initial_running=[MsjCarryIn(MsjLifecycleJob(0, 1, 5), age=3)],
)
assert result.start_times == [-3, 2]
assert result.release_times == [2, 4]
assert result.outcomes == ["completed", "timed_out"]
assert result.resource_time_by_outcome == {
    "completed": 4.0, "cancelled": 0.0, "timed_out": 4.0
}
```

Initial jobs require arrival=0 and no retroactive runtime_limit. A carry-in's
age must leave positive remaining service. All new services and explicit budgets
must be finite positive values. A running job may have no estimate; queued/new
jobs need estimates for EASY/conservative, as in ordinary replay.

Result arrays are ordered running, initial waiting, new arrivals. Carry-in starts
are negative elapsed ages; release times are relative to the boundary. Consumed
service and resource-time count **only time after the boundary**, including drain.
The event engine sees actual release durations; dispatch decisions use K, queue
order and forecasts, never terminal labels or hidden S. Releases at a timestamp
precede dispatch of all arrivals at that timestamp.

`observation_window=(begin,end)` controls utilization/idle-with-queue; defaults
to first/last new arrival. It must lie within [0,last arrival]. A zero window
gives `None` time averages. It does not truncate execution or exclude jobs from
the all-input resource ledger. The experiment applies its fixed target mask
separately rather than counting cancelled competitors as target completions.

## Protocol, metrics and interpretation

Four origins (.35, .50, .65, .80), eight seeds 56000–56007 and source-boundary
random streams match EPIC-056. Historical fits use only already completed
status=1 strictly before cutoff. Models are expanding-coarse and recent-coarse
empirical iid, the latter using the last 4000 known observations in submit order.
Observed is a separate control. No future service rescales generated work.

Only the 1200 selected status=1 services are generated; carry-in and status=5
occupation remain recorded and common to models. Arrival/K/expanding-p90 forecasts
are fixed. Across six policies all workloads are identical per scenario/model/seed.
Total: `4 × 4 × (1 + 2 × 8) × 6 = 1632` schedules, with full draining.

Primary latency is time to terminal release of the fixed 1000 target jobs,
reported alongside successful-completion and timeout fractions. Separate successful
mean/p99 use only successful target jobs; all-timeout cells have `null` successful
latency, not zero. Their denominator changes under limits. A lower terminal
latency under killing is not automatically better completed service.
For targets, `W = simulated_start - submit` and `T = terminal_release - submit`.

The report retains mean W, K-weighted terminal T, resource-group mean/p99, occupied
utilization and idle-with-queue on the target-arrival window, promise violations,
and whole-replay resource time partitioned by terminal outcome. This partition
does not establish useful versus wasted scientific computation.

Model error and policy-choice regret use observed replay **within the same
scenario**. Adjacent scenario comparisons instead report paired raw changes
carry/empty, cancelled/carry, limit/cancelled. They change the workload contract
and are sensitivity measurements, not a scheduler-improvement leaderboard.
Policy choices under limits minimize terminal latency, not successful SLOs.

Intervals are 95% Student t intervals over eight seeds, conditional on the history, target
arrivals/K and supplied historical environment. Observed scenario changes are
deterministic and have no MC interval. There is no population, causal,
multiple-comparison-adjusted or unobserved-cancellation uncertainty guarantee.
Origins share history and are not independent folds. Aggregates weight all
four origins and six policies equally; p99 means are averages of cohort quantiles.
Each model/scenario/policy/origin cell first averages its eight generated metrics.
Its absolute relative error against the matching observed cell then enters
aggregate MAPE. Scenario contrasts instead subtract paired same-seed metrics
within a policy/origin before computing a mean and interval over seeds. There is
no interval over pooled policies or origins, and no aggregate confidence interval.

## Reproduction

From the development environment in [infrastructure](INFRASTRUCTURE.md):

```bash
.venv/bin/python -m examples.real_trace_lifecycle_experiment \
  --output-dir works/real_trace_lifecycle
```

Add `--download` only if the verified EPIC-055 cache is absent and the source
usage conditions have been reviewed. The experiment checks the same SHA-256.
The manifest records source, environment, all origin artifact hashes and thirteen
implementation file hashes. Hashes detect mismatch, not archive dependencies;
retain the corresponding source revision and environment for reproduction.
Raw jobs and identifiers are not exported. Explicit smoke settings through
`--fractions`, `--jobs`, `--warmup`, `--replications` must not be mixed with full runs.

See [EPIC-057](epics/EPIC-057-real-trace-initial-state-cancellation.md),
[results](research/real-trace-lifecycle-results-2026-10.md) and
[roadmap](roadmaps/real-trace-calibration.md).
