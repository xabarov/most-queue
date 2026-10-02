# MSJ useful-service protection

`MsjCheckpointSim(..., min_service_time=q)` adds a minimum useful-service
interval between preemptions. A newly started or restored job can finish at any
time, but cannot be interrupted until it has received q units of useful service
in its current episode. This can amortize checkpoint/resume costs, at the price
of delaying a different job that needs those resources.

The default q=0 preserves the [EPIC-053 checkpoint model](msj_checkpoint.md).
Positive q is an experimental scheduling heuristic, not a throughput-optimality
claim, a new ServerFilling theorem or a Slurm emulation. Costs, the overhead gate
and the original power-of-two restrictions remain unchanged.

## Runnable example

```python
from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.sim.msj_general import MsjTraceJob

sim = MsjCheckpointSim(4, checkpoint_time=0.5, resume_time=0.25,
                       min_service_time=2)
sim.set_servers([1, 4])
trace = (
    MsjTraceJob(0, 0, 10),  # one server; first useful episode starts at 0
    MsjTraceJob(1, 1, 2),   # needs all four, but cannot interrupt before 2
    MsjTraceJob(5, 1, 1),   # encounters the small job's SECOND protected episode
)
result = sim.run_trace(trace)
assert result.start_times == [0, 2.5, 7.25]
assert result.checkpoint_segments == [(0, 2, 2.5), (0, 6.75, 7.25)]
assert result.resume_segments == [(0, 4.5, 4.75), (0, 8.25, 8.5)]
assert result.completion_times == [14.5, 4.5, 8.25]
assert result.preemptions_per_job == [2, 0, 0]
assert result.protection_expirations == 2
```

The second protected interval begins at 4.75, **after** resume, and ends at
6.75. Neither the earlier useful episode nor time spent checkpointing, waiting
or restoring counts toward it. The timer at 6.75 works even without another
arrival or completion at that instant.

`min_service_time` is a keyword-only, finite nonnegative duration, common to
all jobs. The four existing positional constructor arguments, including seed,
are preserved. It uses the same time units as c, r and S. Strings, booleans,
negative or nonfinite values are rejected. A positive duration that cannot
advance the floating timestamp raises an error; there is no implicit epsilon.
The inherited generator, trace replay, warm-up and deadline interfaces remain
available. Valid runtime estimates are ignored; `remaining_predictor` is still
rejected for packing.

## Dispatch and event rules

The selector still packs the shortest unfinished arrival-ordered prefix
reaching total K≥k, or all unfinished jobs if their total need is smaller,
by descending K and FIFO ties. It sees only IDs and K.
Useful episode start times and known q determine preemption eligibility;
future service S and its residual never enter this decision.

| Condition at dispatch | Action |
|---|---|
| Any checkpoint/resume active | No new preemption batch; existing phases continue |
| Gate open, useful job outside selected set, age below q | Continue that job's useful service |
| Gate open, useful job outside selected set, age at least q | Start its checkpoint, preserving progress |
| Selected waiting job fits actually free capacity | First start or resume, as in EPIC-053 |
| Protection expires for a job still selected | No forced preemption or periodic rotation |

Eligibility is per job: one protected job does not prohibit checkpointing
another eligible job in the same batch. Resources held by protected or overhead
phases remain unavailable. The algorithm does not invent a different prefix or
fill resulting holes with an additional FirstFit rule. It may therefore leave
servers idle while jobs wait.

After dispatch, if the overhead gate is open, schedule the earliest protection
deadline among active, still-protected useful jobs outside the selected set.
This is a real review event, not a speculative service completion. Recompute it
after every event; no stale timers survive a change of schedule. During overhead,
phase-ending events suffice to reconsider eligibility. No timers are needed
solely for jobs that remain selected.

At ties, finish all phases, add all arrivals, then dispatch once with protection
ending at that time already expired. A completion at its protection deadline
wins and incurs no checkpoint. A completed short job never receives artificial
extra service to fill q. Timer events do not split continuous useful-service
logs. Protection resets at each useful start after resume.

With c=r=q=0 the class delegates exactly to the original zero-cost SF. With
c=r=0 and q>0 protection still changes the schedule; free preemption does not
disable an explicitly requested minimum interval.

## Measurements and an accounting bound

All [checkpoint result fields and observation windows](msj_checkpoint.md#result-accounting)
retain their meaning. Protected time is **productive**, not an additional
overhead category. W still equals queue wait + checkpoint + resume, and also
first wait + all later interruptions.

Two additional counters include warm-up and drain:

- `protected_preemptions`: rejected job/dispatch preemption attempts because
  episode age is below q while the overhead gate is open. The same job can
  contribute at several events. This is **not** the number of preemptions saved
  relative to a different policy.
- `protection_expirations`: fired review timestamps. Multiple expiries at the
  same timestamp count once; the event can coincide with an arrival/completion,
  and does not imply an actual preemption.

For q>0, each actual preemption ends a distinct useful episode of at least q.
Thus, for a completed job with useful size S and P preemptions, **P·q≤S** in
exact arithmetic. Under the model's complete draining, every preemption pays
one c and one r. If q=h(c+r), h>0 and c+r>0, its cumulative overhead duration
is at most S/h, and overhead server-time at most K·S/h. This is a direct
accounting consequence, not a response-time, fairness or stability bound:
other jobs may wait longer for a protected resource allocation. Floating-point
interval audits allow only numerical rounding, not additional service.

## Choosing a protection level

The [registered EPIC-054 experiment](epics/EPIC-054-msj-protected-service.md)
tests q=h(c+r), h∈{0,1,4,16}. For each workload/load/cost point it selects h
by mean work-weighted T on four independent, fully completed historical replay
traces, breaking exact ties toward smaller h. It freezes the choice before
generating eight held-out test traces. The fully observed tuning traces are
distinct from the censored KM history used by the EASY comparison.

Every fixed candidate remains in the output. The `tuned` result aliases the
historically selected candidate, not the best candidate observed on test data.
Its paired intervals describe test uncertainty **conditional on this one tuning
history**; they do not include repeated-training variability. No production
cost calibration or drift-adaptation claim is made.
See [reproduction commands](../works/msj_protected_service/README.md) and
[results and limitations](research/msj-protected-service-results-2026-10.md).

## Sources and limits

The prefix rule is from Grosof and Harchol-Balter, *ServerFilling: A better
approach to packing multiserver jobs*, ApPLIED 2023,
[DOI](https://doi.org/10.1145/3584684.3597264). Episode protection and its timer
semantics here are our explicit extension, not attributed to that paper.

Holding a schedule to amortize switching costs is a broader scheduling idea:
Celik, Borst, Whiting and Modiano, *Dynamic scheduling with reconfiguration
delays*, Queueing Systems 83 (2016), 87–129,
[DOI](https://doi.org/10.1007/s11134-016-9471-4),
[author abstract](https://dspace.mit.edu/entities/publication/8a3b7a31-a826-41d1-ba21-56272d36cd3b).
Their network/MaxWeight policies and conditions are not this MSJ rule, and their
throughput results are not transferred here.

Ramakrishna, Peng and Scully, *Priority Scheduling in the M/G/1 with Preemption
Overhead*, [preprint v1, 2 May 2026](https://arxiv.org/abs/2605.01522v1), analyzes
class priorities with stochastic pause/resume overhead in one server. It is
related analytical work, not an analytical solution of our multi-resource model.

[Slurm PreemptExemptTime](https://slurm.schedmd.com/slurm.conf.html#OPT_PreemptExemptTime)
documents a minimum runtime before preemption eligibility (consulted 2026-10-02).
This motivates an operational constraint, but does not establish our
reset-after-every-resume semantics or give measured q/c/r values.

No memory or checkpoint I/O contention, restart/lost work, periodic checkpoints,
random costs or online tuning are modelled. Power-of-two resources still matter.
Finite drained experiments and reduced preemption counts do not prove
stationarity, starvation freedom, population p99 or SLO guarantees.
