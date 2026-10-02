# Prediction-free MSJ packing

`MsjGeneralSim` replays identical immutable jobs under prediction-based
backfilling or prediction-free packing. Packing needs only the arrival order,
resource requirement and observed completions. General positive service times
are accepted; exponential service is not assumed by the event engine.

## Choose the operational model first

| `discipline` | Decision rule | Domain | Interrupts jobs? |
|---|---|---|---|
| `fcfs` | Stop at the first waiting job that cannot fit | Any integer K≤k | No |
| `first_fit` | Scan FIFO, skip jobs that cannot fit | Any integer K≤k | No |
| `msf` | Descending K, FIFO ties, skip jobs that cannot fit | Any integer K≤k | No |
| `msfq` | Wide phase, small phase, threshold-triggered drain | K∈{1,k} | No |
| `adaptive_quickswap` | MSF plus a queue/service-class drain trigger | Any integer K≤k | No |
| `server_filling` | Largest-first packing of a minimal unfinished FCFS prefix | k and K powers of two | Yes, resume; zero overhead |
| `easy`, `conservative` | Reservations using explicit runtime forecasts | Any integer K≤k | No |

Unsupported domains raise `ValueError`; needs are never rounded or dropped.
Packing does not require `MsjTraceJob.estimate` and ignores valid supplied
estimates. `remaining_predictor` is accepted only for EASY/conservative.
Only the event engine sees actual S. ServerFilling is **not** ServerFilling-SRPT:
the selection helper receives `(arrival-order ID, need)`, not remaining service.

## Runnable example

```python
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob

# Four simultaneous jobs. IDs are their positions in this tuple.
trace = (
    MsjTraceJob(0, 1, 1),   # need 2, service 1
    MsjTraceJob(0, 0, 10),  # need 1, service 10
    MsjTraceJob(0, 1, 3),   # need 2, service 3
    MsjTraceJob(0, 2, 2),   # need 4, service 2
)
sim = MsjGeneralSim(4, "server_filling")
sim.set_servers([1, 2, 4])
result = sim.run_trace(trace)
assert result.start_times == [0, 3, 0, 1]
assert result.completion_times == [1, 13, 5, 3]
assert result.wait_samples == [0, 3, 2, 1]  # includes the pause of job 2
assert result.preemptions == 1
print(result.service_segments)

# A separate, nonpreemptive one-or-all policy:
quickswap = MsjGeneralSim(4, "msfq", msfq_threshold=1)
quickswap.set_servers([1, 4])
quickswap.set_sources([0.3, 0.1])
quickswap.set_servers([1, 4], [(1.0, "M"), (1.0, "M")])
sample = quickswap.run(1000)
assert len(sample.wait_samples) == 1000
```

Zero-length arrival observation windows, as in the simultaneous example,
return `None` for time averages. All jobs still drain and have latency samples.
The generator interface and existing FCFS/backfilling defaults are unchanged.

## Quickswap semantics

MSFQ implements the four phases in §4.2 of Chen et al.:

1. Serve wide (K=k) jobs exclusively until that class empties.
2. Admit small (K=1) jobs while at least k small jobs remain.
3. Continue small jobs until at most ell remain; `0 <= ell < k`.
4. Admit no new small jobs; drain the active small jobs, then return to phase 1.

Phases 2–3 share admission rules in code. On entry from an empty active set,
admit the initial small batch before testing the threshold; otherwise an empty
phase with queued small jobs could cycle without advancing time. The threshold
is **inclusive**, following the formal rule rather than the introductory
“below” wording. Arrivals during drain wait even if there is no wide job waiting.
This literal phase variant does not add the absent-wide exemption used by the
current author simulator's `QuickSwap.cpp`; finite light-load results need not
be bit-for-bit identical to that implementation. `ell=0` is MSF; at k=1 both
reduce to FCFS. Positive ell with only small jobs need not be M/G/k.

Adaptive Quickswap first fills in MSF order. After that admission batch, it
triggers drain if some resource class is waiting but absent from service and
every class in service has an empty waiting queue. In drain, only the **current**
largest waiting job may start (FIFO ties). Once it starts, MSF admission resumes.
This post-batch trigger convention matches the author's `AdaptiveMSF.cpp`.
Classes here mean distinct K; duplicate external class labels with equal K are
grouped. A latched drain is not cancelled by a new small arrival. The drain target
is recomputed, not frozen against later larger arrivals.

## ServerFilling and measurement

At each arrival/completion, take the shortest arrival-ordered prefix of all
unfinished jobs whose total K reaches k, or all jobs if total K<k. Pack that
prefix by descending K, FIFO ties. Under the power-of-two restriction this fills
all k servers whenever the prefix is full. Stop active jobs outside the chosen
set and later resume their remaining service. No service is redrawn, restarted
or lost; paused completion events are cancelled. Service speed is unchanged.

Completions win event ties; all simultaneous arrivals enter before scheduling.
Replays reset all phases and preemption state. No checkpoint delay, migration
cost, minimum timeslice or restart-on-preempt semantics is modelled.

`start_times` contains **first** starts, `completion_times` final completions.
For ServerFilling, W=T−S includes every pause; it is accumulated as off-service
intervals to avoid subtracting almost equal large numbers. Existing waiting
moments, quantiles and deadline-hit statistics use this total W, not just initial
wait. `service_segments` contains `(trace_id, start, end)` for every continuous
service episode; `preemptions_per_job` includes zeros, and `preemptions` is their
sum. These arrays/counters cover warm-up and drain too. For nonpreemptive policies
the two new arrays are empty, and a single episode can be recovered from the
existing start/completion arrays.

Latency moments, class means and p95/p99 exclude the arrival-index warm-up.
Utilization/occupancy/idle-with-queue use the interval from first measured to
last input arrival, excluding drain. Packing makes **no reservations**: zero
promise violations is not a stronger guarantee than backfilling. The experiment
therefore reports a null violation rate when there are no promises.

Weighted mean T uses theoretical class work shares p_c*K_c*E[S_c]/E[K*S], as in
the Quickswap study. This makes wide classes more visible; it is not a fairness
axiom or a starvation bound. Always also examine unweighted T, class T and tails.

## Evidence and limits

- Exact phase/prefix examples, event ties, estimate/hidden-S independence,
  replay reset, unsupported domains, exhaustive short power-of-two packings.
- Seeded non-exponential traces: resource capacity and every job's delivered
  service conserved, including repeated preemptions.
- All-wide M/E2/1 and all-unit M/M/4 comparisons for every new discipline,
  MSFQ ell=0 in the latter; shared project tolerances, first two raw W/T moments
  and occupancy probabilities. These special cases do not validate a general
  analytical formula for MSFQ or ServerFilling.
- [Registered experiment and results](research/msj-packing-results-2026-10.md),
  [reproduction commands](../works/msj_packing/README.md).

The MSFQ throughput theorem uses the paper's one-or-all Poisson/exponential
model; we do not transfer it to arbitrary G or Adaptive Quickswap. A finite
drained trace is not evidence of stability, absence of starvation, or a
stationary tail guarantee. ServerFilling's packing property is verified under
its resource assumptions; the simulation does not prove new performance bounds.
Its preemption capability differs from EASY/MSFQ. Checkpoint cost is studied in
the separate [EPIC-053 extension](msj_checkpoint.md), without changing this
zero-cost model. Real traces, DivisorFilling, Static Quickswap and
ServerFilling-SRPT remain out of scope.

## Primary sources

- Chen et al., *Improving nonpreemptive multiserver job scheduling with
  quickswap*, Performance Evaluation **171 (2026)**, 102525,
  [DOI](https://doi.org/10.1016/j.peva.2025.102525),
  [author manuscript](https://jcpwfloi.com/assets/publications/performance2025-final49.pdf).
  The DOI contains 2025; the issue is dated March 2026. Rules: §4.1–4.4.
  [Author implementation, pinned revision](https://github.com/UniVe-NeDS-Lab/mjqm-simulator/tree/a78f3980c44a142c2e00b229a0f665358ca3e10c),
  consulted 2026-10-02; implementation conventions above are explicit.
- Grosof and Harchol-Balter, *ServerFilling: A better approach to packing
  multiserver jobs*, ApPLIED 2023,
  [DOI](https://doi.org/10.1145/3584684.3597264),
  [author manuscript, §6](https://www.cs.cmu.edu/~harchol/Papers/Applied23.pdf).
  Candidate prefix, power-of-two condition, preemption and W definition.
- Grosof, Harchol-Balter, Scheller-Wolf, *WCFS: a new framework for analyzing
  multiserver systems*, Queueing Systems 102, 143–174 (2022),
  [DOI](https://doi.org/10.1007/s11134-022-09848-6).
  Background for the framework cited by the ServerFilling paper; no new bound
  from this work is implemented here.
