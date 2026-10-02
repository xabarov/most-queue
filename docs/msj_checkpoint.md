# MSJ checkpoint and resume costs

`MsjCheckpointSim` is an explicit, experimental extension of prediction-free
ServerFilling. It asks whether the benefit of repacking survives the time spent
saving and restoring interrupted work. Costs affect the event schedule itself;
they are not added to latency after a zero-cost run.

Use `MsjGeneralSim(..., "server_filling")` for the original zero-cost model.
Use this separate class when checkpoint and resume consume time on the job's
allocated servers. Neither model assumes exponential useful service times.
This page describes the default `min_service_time=0`. Optional
[useful-service protection](msj_protected_service.md) delays preemption until a
minimum episode age, without changing the default model or charging extra work.

## Runnable example

```python
from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.sim.msj_general import MsjTraceJob

trace = (
    MsjTraceJob(0, 1, 1),   # need 2, useful service 1
    MsjTraceJob(0, 0, 10),  # need 1, useful service 10
    MsjTraceJob(0, 1, 3),   # need 2, useful service 3
    MsjTraceJob(0, 2, 2),   # need 4, useful service 2
)
sim = MsjCheckpointSim(4, checkpoint_time=0.5, resume_time=0.25)
sim.set_servers([1, 2, 4])
result = sim.run_trace(trace)
assert result.start_times == [0, 3.5, 0, 1.5]
assert result.completion_times == [1, 13.5, 5.75, 3.5]
assert result.wait_samples == [0, 3.5, 2.75, 1.5]
assert result.queue_wait_samples == [0, 3.5, 2, 1.5]
assert result.checkpoint_segments == [(2, 1, 1.5)]
assert result.resume_segments == [(2, 3.5, 3.75)]
assert result.checkpoint_resource_time == 1.0  # K=2 times c=0.5
assert result.resume_resource_time == 0.5
assert result.utilization is None  # all arrivals simultaneous: zero window

# Generator API is inherited; costs are in the same time units as service.
generated = MsjCheckpointSim(4, 0.05, 0.02, seed=53)
generated.set_servers([1, 4], [lambda rng: rng.gamma(2, 0.5)] * 2)
generated.set_sources([0.2, 0.1])
sample = generated.run(1000)
assert len(sample.wait_samples) == 1000
```

`k` and each class need K must be powers of two, with K≤k. Costs c and r must
be finite, nonnegative numbers. They are deterministic durations **per job per
interruption**, not fractions of its hidden service S. First starts have no
resume cost. Unsupported needs, invalid costs and `remaining_predictor` raise
`ValueError`. Valid supplied `estimate` fields are ignored, as for zero-cost
packing. Trace validation, deadline thresholds and arrival-index warm-up follow
the [general MSJ API](models/msj.md).

## State and scheduling contract

| Phase | Holds K servers? | Useful work advances? | Next phase |
|---|---|---|---|
| Initial queue | No | No | Service on first admission |
| Service | Yes | Yes | Completion or checkpoint on preemption |
| Checkpoint, duration c | Yes | No | Paused queue |
| Paused queue | No | No | Resume on readmission |
| Resume, duration r | Yes | No | Service with the preserved remainder |

At every arrival, completion or overhead-phase ending, select the shortest
arrival-ordered prefix of **all unfinished jobs** reaching total need k (or all
jobs if smaller). Greedily pack this prefix by descending K, FIFO ties, as in
[ServerFilling](msj_packing.md). Only order and K reach the selector; actual S
is private to the event engine.

The positive-cost extension uses these additional rules:

1. While **any** checkpoint or resume is active, do not start a new preemption
   batch. Existing useful service continues; overheads cannot be cancelled.
2. Otherwise, simultaneously checkpoint every active useful job outside the
   selected set. Progress is preserved, with no restart or redrawn service.
3. Admit selected waiting jobs that fit the **actually free** servers, even
   while overhead is active. A resumed job first pays r. Selection is recomputed
   at the next event; it does not reserve a future start or freeze a target.

This gate is a modelling choice to prevent overlapping preemption batches;
it is not a published ServerFilling theorem or a cost-aware optimization rule.
It can leave capacity idle while jobs wait. A just-restored job can immediately
be preempted again if the selected set changed during its restoration.

For simultaneous events, finish **all** phases, enqueue **all** arrivals, then
dispatch once. Zero overhead resolves immediately, with no artificial epsilon.
Positive phases that cannot advance the floating-point timestamp raise an
error: rescale the trace instead of silently losing elapsed time. Zero-length
useful episodes after restoration are not logged.

For c=r=0 and default `min_service_time=0`, the class delegates to the original ServerFilling replay, preserving
every original result field apart from runtime measurement. This is an exact
reference case, not a claim about convergence of every positive-cost path.

## Result accounting

`MsjCheckpointResults` extends `MsjSimulationResults`. For each measured job:

```text
W = T - S = queue_wait + checkpoint + resume
W = first_wait + interruption
```

`queue_wait_samples` includes the initial queue and all paused queues, but
neither overhead phase. `first_wait_samples` ends at first useful service;
`interruption_samples` includes every later paused queue and overhead phase.
`checkpoint_samples` and `resume_samples` are cumulative elapsed overhead
times. `wait_samples`, W moments and waiting-deadline statistics use total W.
The equality to T−S is conceptual; W is accumulated from intervals to avoid
cancellation when subtracting large timestamps.

| Scope | Fields |
|---|---|
| Jobs after `warmup_jobs` | Latency samples/moments/quantiles, class delays, the five new per-job delay arrays |
| Entire trace, including warm-up and drain | First `start_times`, final `completion_times`, preemption counters, `service_segments`, `checkpoint_segments`, `resume_segments`, overhead resource-time |
| First measured arrival to last input arrival, excluding drain | `utilization`, `productive_utilization`, `checkpoint_utilization`, `resume_utilization`, `p`, `idle_with_queue`, `throughput` |

Every segment is `(trace_id, start, end)`; the ID is the zero-based input
position. Useful segments sum to the original S for each job. Overhead resource
times sum K times segment duration over the **whole trace**, not just the
utilization window. With a positive observation window:

```text
utilization = productive_utilization + checkpoint_utilization + resume_utilization
```

`utilization` therefore means allocated capacity, not useful throughput.
Time averages are `None` for a zero-length window. The inherited throughput
counts all completions in `(first measured arrival, last input arrival]`,
including warm-up jobs that finish in that window, divided by its duration.
It excludes drain and is an observed finite-window rate, not stationary capacity.
The model makes no reservation promises. Zero promise violations cannot be
compared to a nonzero-denominator backfilling guarantee.

## Validation and evidence

Exact traces cover costs on either side, event ties, repeat replay and the
zero-cost reference. A separate integer-time oracle decrements per-job work and
phase counters, without calling the event engine or packing helper. Random
non-exponential traces audit all service/overhead intervals, capacity, the gate
and both waiting-time identities. All-wide M/E2/1 and all-unit M/M/4 compare
with analytical solvers using shared project tolerances: neither case preempts,
so configured costs are not spuriously charged.

The [prespecified 832-run study](epics/EPIC-053-msj-checkpoint-cost.md) keeps
useful arrival load fixed across overhead levels, with paired FirstFit, MSF,
backfilling and zero-cost SF references. See [results and limitations](research/msj-checkpoint-results-2026-10.md)
and [reproduction commands](../works/msj_checkpoint/README.md).

## Limits and primary sources

No checkpoint I/O contention, retained memory, migration, failures, lost work,
periodic checkpointing, random overhead sizes or cost-aware tuning is modelled.
A finite trace is fully drained, even if an infinite arrival stream would be
unstable. Do not infer stability, starvation freedom or a universal break-even
cost from these cohort statistics; the original zero-idle packing property
does not carry over to overhead-holding phases.

- Grosof and Harchol-Balter, *ServerFilling: A better approach to packing
  multiserver jobs*, ApPLIED 2023,
  [DOI](https://doi.org/10.1145/3584684.3597264): original prefix rule, not the
  overhead gate introduced here.
- Chen et al., *Improving nonpreemptive multiserver job scheduling with
  quickswap*, Performance Evaluation 171 (2026), 102525,
  [DOI](https://doi.org/10.1016/j.peva.2025.102525),
  [author manuscript, Appendix D](https://arxiv.org/html/2509.01893v2): its SF
  comparison assumes no preemption/setup overhead. Our cost grid is synthetic,
  not calibrated from that paper.
- [Official Slurm preemption documentation](https://slurm.schedmd.com/preempt.html),
  consulted 2026-10-02: suspend/resume, requeue and cancellation are distinct
  operational mechanisms; suspension can retain memory. This single-resource
  model does not emulate Slurm or assign measured costs to those mechanisms.
