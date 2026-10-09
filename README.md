<div align="center">

# Most-Queue

**Queueing theory in Python: exact analytical solvers paired with discrete-event simulation — for 50+ models from M/M/1 to multiserver-jobs, RDR priorities, SRPT scheduling, age of information, vacations and queueing networks.**

[🇷🇺 Русская версия](README.ru.md)

[![Tests](https://github.com/xabarov/most-queue/actions/workflows/tests.yml/badge.svg)](https://github.com/xabarov/most-queue/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/most-queue)](https://pypi.org/project/most-queue/)
[![Python versions](https://img.shields.io/pypi/pyversions/most-queue)](https://pypi.org/project/most-queue/)
[![License](https://img.shields.io/pypi/l/most-queue)](https://github.com/xabarov/most-queue/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/most-queue)](https://pepy.tech/project/most-queue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21268402.svg)](https://doi.org/10.5281/zenodo.21268402)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/xabarov/most-queue)](https://github.com/xabarov/most-queue/commits/main)

<img src="https://raw.githubusercontent.com/xabarov/most-queue/main/assets/most-queue-nano1.jpeg" alt="Most-Queue banner" width="720"/>

</div>

## Why Most-Queue?

- **Analytics and simulation together.** Nearly every analytical calculator ships with a paired
  discrete-event simulator, and the test suite cross-validates them against each other. You get
  fast exact numbers *and* a way to check them.
- **Models you won't find elsewhere in open source**: size-based scheduling analytics
  (SRPT, SJF, PSJF, SPJF with ML-style size predictions, FB/LAS), M/G/1 vacation models,
  negative customers (RCS / disasters), unreliable servers, multi-server phase-type systems
  solved by the Takahashi–Takami method (including CV < 1 via complex-fit H₂).
- **Moments, not just means**: waiting/sojourn time raw moments, state probabilities,
  utilization — with a uniform `set_sources() / set_servers() / run()` API across all models.
- **Pure Python + NumPy/SciPy**, pip-installable, MIT license.

## Installation

```bash
pip install most-queue
```

Requires Python ≥ 3.10. For network visualization you may also need the system `graphviz` package.

## Quick start: theory vs simulation in 20 lines

```python
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.sim.base import QsSim

# Analytical M/M/3 with a finite queue
calc = MMnrCalc(n=3, r=100)
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
theory = calc.run()

# The same system, simulated
sim = QsSim(3)
sim.set_sources(2.0, "M")
sim.set_servers(1.0, "M")
experiment = sim.run(100_000)

print(f"Mean waiting time: theory {theory.w[0]:.3f} vs simulation {experiment.w[0]:.3f}")
# Mean waiting time: theory 0.444 vs simulation 0.448
```

## Showcase: who pays for the scheduling discipline?

Computed by the library's own calculators — conditional slowdown `E[T(x)]/x` by job size
for FCFS, PS, FB (blind) and SRPT (size-aware):

<img src="https://raw.githubusercontent.com/xabarov/most-queue/main/docs/figures/slowdown.png" alt="Slowdown by job size for FCFS/PS/FB/SRPT" width="720"/>

See the executable comparison of **9 disciplines** in
[`tutorials/disciplines_comparison.ipynb`](tutorials/disciplines_comparison.ipynb).

## What's inside

| Family | Models | Method |
|---|---|---|
| Classic FIFO | M/M/c, M/M/c/r, Erlang B/C, M/G/1, GI/M/c, M/D/c, Eₖ/D/c, M/G/∞ | exact |
| Multi-server phase-type | M/H₂/c, H₂/M/c, H₂/H₂/c (CV < 1 via complex fit) | Takahashi–Takami |
| Size-based scheduling | M/G/1 SRPT, SJF, PSJF, SPJF (with size predictors + graceful-degradation curves), FB/LAS, PS, LCFS-PR | exact (Schrage–Miller, Mitzenmacher) |
| Priorities | M/G/1 PR/NP multi-class, M/G/c PR/NP, M/Ph/c PR; **RDR** M/M/k & M/PH/k multi-class (exact + RDR-A), per-class response variance; M/M/2 with **heterogeneous servers** (exact non-birth-death CTMC); **accumulating priority** (Kleinrock/APQ), priority Erlang-A (impatience), MMAP/PH/1 priorities (NP/PR/RS), retrial with a priority class, preemptive repeat | exact / RDR / CTMC / invariant approximation |
| [Multiserver-job (MSJ)](docs/models/msj.md) | FCFS with phase-type service; general-service FCFS/EASY/conservative, FirstFit, MSF and Quickswap; power-of-two ServerFilling, checkpoint/resume costs and optional useful-service protection | FCFS CTMC / saturated analysis; trace-driven simulation for scheduling policies |
| Load balancing (mean-field) | dispatching over a large pool — power-of-d / JSQ / JIQ / random | mean-field fixed point |
| Polling systems | one server touring Q queues with switchover — exhaustive / gated | pseudo-conservation law (Boxma–Groenevelt) |
| Non-stationary Mt/M/c | time-varying arrival rate λ(t) — blocking & waiting probability | PSA & MOL approximations |
| Age of Information | M/M/1, M/G/1, preemptive-LCFS — average & peak AoI | closed-form + simulation |
| SLA / deadline-violation probability | `P(W > D)` and SLO quantile from raw moments — works on top of any model in this table; LLM-serving TTFT SLO example | H2/Gamma tail fit, M/M/1 exact anchor |
| Vacations & warm-up | M/G/1 multiple vacations, N-policy, warm-up/cooling/delay (M/Ph/c) | Fuhrmann–Cooper, Takahashi–Takami |
| Negative customers | M/G/1 and M/G/c with RCS or disasters | exact / Takahashi–Takami |
| Reliability | M/G/1 and M/M/c with breakdowns & repairs, machine repair problem (spares, R repairmen, 2 heterogeneous repairmen), working breakdowns, disasters with a repair phase, retrial with an unreliable server | Avi-Itzhak–Naor / exact CTMC / birth-death |
| Matrix-analytic (MAP/PH) | MAP/PH/1, M/PH/1, PH/PH/1, MAP/M/c, MAP/PH/c — correlated (bursty) arrivals, single- & multi-server; MMPP fitting | QBD, logarithmic reduction |
| Batch Markovian arrivals | BMAP/M/1, BMAP/PH/1 — correlated batch traffic | level truncation |
| Retrial & abandonment | M/M/1 and M/G/1 retrial (orbit), Erlang-A (M/M/n+M) with staffing | exact / Falin–Templeton |
| GI/G approximations | GI/G/1, GI/G/m mean waiting time | Kingman, Krämer–Langenbach-Belz, Allen–Cunneen |
| Batch arrivals & bulk service | Mˣ/M/1 batch arrivals; M/M^[a,b]/1 bulk (batch) service — LLM inference batching, exact N/W moments and tail for any a≤b; general (Erlang- or H2-fitted, CV≤1 or CV≥1) batch-service time with auto-dispatch by CV and batch-size-dependent parameters; Markovian abandonment (MM1/Erlang/H2); `c` independent servers sharing one queue, incl. batch-size-dependent service | exact |
| Impatience & closed | M/M/1+M, Engset | exact |
| Queueing-inventory | M/M/1, M/M/c, or c heterogeneous servers (identical, exponential, or per-server Erlang-/H2-fitted service) with stock-consuming service, general (s,S) replenishment (exponential or Erlang-fitted lead time), backordering or lost sales | exact QBD |
| EDF scheduling | Earliest-Deadline-First as the actual service discipline (not post-hoc SLA) | DES-exact, no closed form (open problem) |
| Deadline-aware admission control | Accept/reject at arrival based on own-deadline feasibility (not reordering), Exp(θ) deadline — LLM-serving SLO | exact convergent series (level-crossing functional equation) |
| Parallel service | Fork-Join, Split-Join; exact heavy-tailed (Pareto) max-of-n; heterogeneous branches, series-parallel task DAGs, (n,k)-join over heterogeneous/DAG branches | Markovian / order statistics / exact (Beta function) |
| Networks | open (decomposition, exact Jackson, QNA two-moment flows, MAP input), closed (exact MVA / Buzen / Schweitzer), multi-class BCMP, G-networks (Gelenbe, multi-class), tandems with blocking (finite buffers), fork-join stations, time-varying λ(t), priorities, negative customers, routing optimization | decomposition / product form / MVA |

Every model comes with a plain-language explanation and a diagram in the
[illustrated model catalog](docs/models.md).

## Documentation & tutorials

- 📖 [Documentation](docs/README.md) — concepts, calculation and simulation guides (English; Russian versions available via in-page switchers)
- 🎓 [Jupyter tutorials](tutorials/README.md) — counter-intuitive queueing insights for engineers (the utilization trap, why variability dominates delay, multiserver jobs, Age of Information, …)
- 🗺 [Development roadmaps](docs/epics/README.md) & [trends surveys](docs/research/) — literature-driven gap analysis behind each epic, and what's next
- 🧪 [Tests](tests/) — every model validated against simulation; run with `pytest -m "not slow"`

## Applications

Capacity planning for cloud services and data centers · call-center staffing ·
manufacturing lines · telecom traffic · healthcare resource planning ·
scheduling research (SRPT/LAS with ML size predictions).

## Recent highlights

- **2026-10, current source tree** — **Exact SLA tail for batch-service queues**:
  `get_tail`/`get_cdf` give the exact (not moment-fitted, not an upper bound)
  `P(W>D)` for GPU/LLM-style bounded-window batch service with phase-type
  (Erlang/H2, any CV) processing time, via sparse matrix-exponential-action.
  Quantified against Inoue (2021)'s closed-form mean-latency bound (a bounded
  window can push true E[W] up to ~10x above it) and against the library's own
  moment-fit SLA approximation (which overestimates deep-tail violation
  probability by orders of magnitude). 19 new tests, DES-validated.
  [Method](docs/research/batch-service-sla-exact-tail-2026.md), [results](docs/research/batch-service-sla-exact-tail-results-2026.md).
- **2026-10, current source tree** — **Occupancy-dependent continuous batching**: exact
  waiting-time distribution (moments and tail, not just the mean) for the Markovian core
  of LLM-serving "continuous batching" — up to `k` requests served concurrently with a
  per-request rate that depends on current occupancy, capped by accelerator memory
  (KV-cache). Occupancy is pinned at the cap whenever anyone waits, so `W` is an exact,
  closed-form mixture of Erlang distributions — no sparse solve needed. Regresses exactly
  to the classical `M/M/k/N` queue at a constant rate; DES-validated. A simple
  approximation that ignores the occupancy dependence is shown to overstate
  deadline-violation risk by 1-2 orders of magnitude at moderate occupancy and several
  orders of magnitude at a large concurrency cap. [Model](docs/models/continuous-batching.md).
- **2026-10, current source tree** — **Idle-refill, impatience, and multiserver bulk-service**:
  three further generalizations of the bounded-window batch-service model — exact W
  moments/tail for any idle-refill race `a<=b` (not just `a=1`); Markovian abandonment
  (`gamma`-rate reneging) for MM1/Erlang and H2-branching service, each via a dedicated
  absorbing chain tracking surviving-ahead-count and new-arrivals-behind as explicit
  state; and `c` independent servers sharing one FCFS queue, including batch-size-dependent
  `mu` at `gamma==0` and abandonment at constant `mu`. All DES-validated independently of
  the production code; dependent-`mu` combined with abandonment remains an explicit
  reserve. See the [epic registry](docs/epics/README.md) for the full breakdown
  (EPIC-067 through EPIC-072).
- **2026-10, current source tree** — **Availability-aware arrival history**:
  removes a completed-prefix staleness artifact (prefix lag up to 8.2 days)
  by building arrival-mark donor history from submit<cutoff over the already-
  parsed completed+cancelled(+failed/timeout/node_failed) population, not
  stalling at the first job unresolved at cutoff. Lag drops to minutes/hours
  and the donor pool roughly doubles, but this is not a uniform accuracy win:
  queue-error effects go in opposite directions on SDSC versus Kalos. 16 unit
  tests, 792 schedules, byte-identical repeat, independent audit.
  [Method](docs/availability_aware_arrivals.md), [results](docs/research/availability-aware-arrivals-results-2026-10.md).
- **2026-10, current source tree** — **MSJ capacity calendar**: opt-in
  time-varying capacity (`CapacityCalendar`/`run_capacity_calendar`) over MSJ
  replay, with automatic grandfathering, explicit infeasible/unresolved
  outcomes bounded by the calendar's own horizon, and detected (not silently
  absorbed) reservation/calendar mismatches. 23 unit tests, plus three
  prescribed Helios scenarios (fixed/daily/isolated-VC pool) over one
  prescribed window, byte-identical repeat and an independent check.
  [Method](docs/msj_capacity_calendar.md), [results](docs/research/msj-capacity-calendar-results-2026-10.md).
- **2026-10, current source tree** — **Resource observability audit**:
  six source schemas compared; all 3,362,981 Helios records audited with pinned
  inputs, a byte-identical repeat and independent accounting checks. Daily VC
  GPU counts support bounded capacity scenarios, not reconstructed hard quotas:
  17,459 GPU records start with a zero same-day VC count. No scheduler replay claimed.
  [Method](docs/resource_observability.md), [results](docs/research/resource-observability-results-2026-10.md).
- **2026-10, current source tree** — **Joint marked arrivals**:
  empirical gap/K/context/request tuples, circular blocks and matched shuffles,
  with a common coarse S|K mechanism and fixed-arrival baseline. 1176 empty-start
  replays and a byte-identical repeat show no consistent joint-generator gain;
  resolved-prefix staleness and late workload-mix drift remain explicit limitations.
  [Protocol/API](docs/joint_marked_arrivals.md), [results](docs/research/joint-marked-arrivals-results-2026-10.md).
- **2026-10, current source tree** — **Queue-aware service selection**:
  1500 replays with two early selection and two late evaluation blocks per source.
  Queue-selected SDSC request bins improve point uncapped queue-loss versus CRPS selection,
  but not versus fixed coarse; Kalos type candidates coincide on late service tapes.
  Class coverage, timeout rates and policy regret remain separate checks.
  [Protocol/API](docs/queue_aware_selection.md), [results](docs/research/queue-aware-selection-results-2026-10.md).
- **2026-10, current source tree** — **GPU resource envelopes**:
  prescribed GPU-pool versus exclusive-node capacity sensitivity, with fixed
  target IDs and explicit infeasible cells. Requested GPU work and reserved
  GPU-equivalent work remain separate; no quota or placement estimation is claimed.
  1080 replays and a byte-identical repeat; observed p99 ties all six policies
  in every feasible cell, so zero regret does not validate scheduler choice.
  [Protocol/API](docs/gpu_resource_envelope.md), [results](docs/research/gpu-resource-envelope-results-2026-10.md).
- **2026-10, current source tree** — **Feature-conditional service**:
  450 paired replays with early-validation model selection. Requested-time ratios
  reduce SDSC test CRPS by 46.69% and excess timeouts, but do not preserve p99
  policy choice; retrospective Kalos type features degrade on the late holdout.
  [Protocol/API](docs/feature_service.md), [results](docs/research/feature-service-results-2026-10.md).
- **2026-10, current source tree** — **Modern GPU-trace validation**:
  hash-pinned Acme/Kalos ingestion recomputes execution from end-start; the
  released `duration` includes waiting. 1224 replays with separate failed/cancelled
  occupation and an identical full repeat. At nominal GPU-pool capacity all
  observed target waits are zero, so tied policy choices do not validate a
  scheduler. Service-model errors remain large. [Protocol/API](docs/modern_gpu_trace.md),
  [results](docs/research/modern-gpu-trace-results-2026-10.md).
- **2026-10, current source tree** — **Initial state and terminal outcomes**:
  opt-in MSJ replay with running/waiting carry-in, labelled cancelled occupation
  and start-relative hard runtime budgets. Successful completion, cancellation
  and timeout are accounted separately; shortened terminal latency is not treated
  as successful service. The real-trace study keeps a fixed completed target cohort
  and explicitly excludes cancellations with unobservable resource use.
  In 1632 replays, adding observed cancelled occupation changed the best mean-delay
  policy in one period; both service models missed the p99 winner in all four.
  [Protocol/API](docs/real_trace_lifecycle.md),
  [results](docs/research/real-trace-lifecycle-results-2026-10.md).
- **2026-10, current source tree** — **Temporal history and dependent service**:
  four prespecified SDSC SP2 origins, seven workload variants and 1368 replays.
  Recent coarse-group empirical history reduced aggregate mean-delay error
  from 94.57% to 44.12%, but helped neither every period nor every finer model.
  Rank blocks reproduced more serial correlation without a consistent delay
  or p99-decision advantage. These are selected-cohort diagnostics, not causal
  production guarantees. [Protocol](docs/real_trace_temporal.md),
  [results](docs/research/real-trace-temporal-results-2026-10.md).
- **2026-10, current source tree** — **Real-trace service calibration**:
  audited, hash-pinned SDSC SP2 ingestion; completed-history temporal holdout;
  empirical, Exp, PH and lognormal service compared under six existing MSJ
  disciplines. In 594 replays, all four models preserved the best mean-delay
  choice but substantially underestimated delays in the first two blocks;
  p99-based choices differed from observed-duration replay. This is a selected
  historical completed-job cohort, not a production-cluster reconstruction.
  [Protocol](docs/real_trace_calibration.md),
  [results](docs/research/real-trace-calibration-results-2026-10.md).
- **2026-10, current source tree** — **MSJ beyond exponential service**:
  small-system PH-FCFS analytics and common-trace experiments for backfilling
  and prediction-free packing. Runtime forecasts support historical features,
  resource-group calibration and age-aware Kaplan–Meier estimates from censored
  history. MSFQ is one-or-all; ServerFilling is a separate power-of-two,
  zero-cost preemptive-resume model. `MsjCheckpointSim` separately models
  resource-holding checkpoint/resume phases and reports productive utilization.
  Optional `min_service_time` protects each useful episode; protection can be
  selected on independent historical replays, without test-set tuning.
  In 832 paired sensitivity runs, overhead could erase the packing advantage;
  the crossover depended on workload, not a universal cost threshold.
  A further 1920-run protection study found gains over unprotected SF at high
  overhead; zero protection was the historical choice at both lower cost levels.
  [Methods and API](docs/models/msj.md), [packing results](docs/research/msj-packing-results-2026-10.md),
  [checkpoint-cost results](docs/research/msj-checkpoint-results-2026-10.md),
  [protected-service results](docs/research/msj-protected-service-results-2026-10.md),
  [roadmap](docs/roadmaps/msj-ph-backfilling.md). Simulation comparisons are not
  new analytical stability or SLO guarantees; PyPI releases may lag this tree.
- **2026-09** — **Realism wave, part 2: Erlang everywhere, batch-size-dependent params, exact
  moments**: **queueing-inventory** replenishment lead time generalized from `Exp(θ)` to
  **Erlang-fitted** (the phase dimension only applies where an order can be in transit — a first
  for this phase-type family); the per-server heterogeneous model gained an **Erlang-fitted
  (CV≤1) service** sibling to its H2 case, with a real outflow-splitting bug (departure vs.
  mid-service phase-advance rates) caught via a deliberately non-degenerate regression check;
  **bulk-service** batch-service parameters (Erlang and H2) can now depend on batch size (LLM/GPU
  dynamic-batching realism), and the Erlang-fitted case gained **exact raw moments of W** (not
  just the mean), extending EPIC-032's PASTA technique to the phase-augmented CTMC. See the
  [epic registry](docs/epics/README.md) for the full breakdown (EPIC-040 through EPIC-043).
- **2026-09** — **Realism wave: heterogeneous branches, general service, admission control**:
  **fork-join** generalized to heterogeneous branches, series-parallel task DAGs and (n,k)-join
  (exact order-statistics moments, closed-form Pareto); **bulk-service** batch-service time
  generalized from exponential to **Erlang** (CV≤1) and **H2** (CV≥1) phase-type fits, with exact
  N/W moments and an auto-dispatcher picking the family by CV; exact **deadline-aware admission
  control** (accept/reject at arrival based on own-deadline feasibility, Exp(θ) deadline — solved
  via a level-crossing functional equation and truncated power-series moment extraction, after an
  initial "obvious" shortcut was proven wrong); and **queueing-inventory** generalized to `c`
  heterogeneous servers (state-splitting + stacked-boundary QBD) with, going one step further,
  **per-server H2-fitted (non-exponential) service time** — each server can have its own realistic
  service-time distribution instead of a single shared rate. See the
  [epic registry](docs/epics/README.md) for the full breakdown (EPIC-030 through EPIC-039).
- **2026-09** — **Priorities, inventory & scheduling wave**: **M/M/2 priorities with
  heterogeneous servers** (exact non-birth-death CTMC, canonicalized state construction);
  **queueing-inventory** generalized to lost-sales, the full **(s,S)** reorder-point policy
  (not just (0,S)), and **M/M/c** multi-server stock-sharing (exact QBD via a stacked
  boundary superblock, reduces exactly to M/M/1 at c=1); and **EDF (Earliest Deadline First)**
  as an actual service discipline rather than a post-hoc SLA metric — deliberately DES-only,
  since exact finite-state EDF analysis turned out to be a genuinely open problem (four
  candidate exact reductions were tried and numerically/analytically disproven; see
  [`docs/research/edf-scheduling-2026.md`](docs/research/edf-scheduling-2026.md)).
- **2026-09** — **SLA & exact-extensions wave**: a horizontal **SLA/deadline-violation
  probability** layer (`P(W > D)` and SLO quantiles from raw moments, on top of *any* calculator
  in the table above — LLM-serving TTFT SLO example); exact **heavy-tailed (Pareto) fork-join**
  (closed-form max-of-n via the Beta function, no approximation); **machine repair with two
  heterogeneous repairmen** (Krishnamoorthi's non-birth-death CTMC technique); and the first
  **queueing-inventory system** (M/M/1 with stock-consuming service, (0,S) replenishment,
  backordering — an exact QBD, reusing the MAP/PH stack's solver). See the
  [research](docs/research/) folder for the literature review behind each.
- **2026** — **Scale & dynamics wave**: **load balancing** in the mean-field limit
  (power-of-d / JSQ / JIQ — the "power of two choices" behind modern dispatchers);
  **polling systems** (one server touring Q queues with switchover, exhaustive/gated, the
  Boxma–Groenevelt pseudo-conservation law); and **non-stationary Mt/M/c** with time-varying
  load (PSA & MOL approximations for surging demand — call-center staffing, autoscaling).
  Each with a paired simulator and tests. See the
  [trends survey](docs/research/queueing-trends-2026.md).
- **2026 (v2.9)** — **Datacenter & multi-priority wave**: **RDR** for multi-server multi-class
  preemptive priorities (M/M/k and M/PH/k, exact + RDR-A, per-class response-time variance);
  the **multiserver-job** model (jobs holding several servers at once — FCFS response time and
  saturated-system stability, the first open-source implementation); **Age of Information**
  (average & peak AoI); **bulk-service** queues for LLM inference batching; and
  **graceful-degradation curves** for prediction-based scheduling. See the
  [trends survey](docs/research/queueing-trends-2026.md).
- **2026** — **Matrix-analytic MAP/PH stack**: PH distributions and MAPs
  (`most_queue.random.map_ph`), a QBD solver with logarithmic reduction, and exact calculators
  for MAP/PH/1, M/PH/1, PH/PH/1, **MAP/M/c**, **MAP/PH/c**, plus **BMAP/M/1** and **BMAP/PH/1**
  for batch arrivals and **MMPP fitting** from data; MAP and PH sources in the simulator. Plus
  retrial queues (orbit) and Erlang-A abandonment with a staffing helper.
  See [`tutorials/map_ph_correlation.ipynb`](tutorials/map_ph_correlation.ipynb).
- **2026** — Wave of exact classics: Erlang B/C, M/G/∞, GI/G approximations, M/G/1 vacation
  models (multiple vacations, N-policy), PS, LCFS-PR, FB/LAS, unreliable server — each with a
  paired simulator and tests. Illustrated model catalog with generated diagrams.
- **2026** — Size-based scheduling analytics: SRPT / SJF / PSJF / SPJF with prediction models
  (reproduces the Mitzenmacher–Shahout 2025 table in tests) + `SizeBasedQsSim`.
- **2026 (preprint)** — Multi-server queues with negative customers via Takahashi–Takami:
  [preprint & reproduction code](works/negative_queues/).
- **2025 (paper)** — Multi-channel system with warm-up, cooling and cooling delay:
  Lokhvitsky, Khabarov, Yakovlev, DOI [10.25791/aviakosmos.1.2025.1456](https://doi.org/10.25791/aviakosmos.1.2025.1456).

## Contributing

Issues and pull requests are welcome! Open an [issue](https://github.com/xabarov/most-queue/issues)
for bugs or model requests. Development conventions: [docs/PROJECT.md](docs/PROJECT.md),
definition of done: [docs/DOD.md](docs/DOD.md).

## Citation

If you use Most-Queue in research, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{most_queue,
  author  = {Khabarov, Roman},
  title   = {Most-Queue: queueing theory calculations and simulation in Python},
  url     = {https://github.com/xabarov/most-queue},
  doi     = {10.5281/zenodo.21268402},
  license = {MIT}
}
```

## License

[MIT](LICENSE) © Roman Khabarov
