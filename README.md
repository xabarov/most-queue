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

Requires Python ≥ 3.9. For network visualization you may also need the system `graphviz` package.

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
| Priorities | M/G/1 PR/NP multi-class, M/G/c PR/NP, M/Ph/c PR; **RDR** M/M/k & M/PH/k multi-class (exact + RDR-A), per-class response variance | exact / RDR / invariant approximation |
| Multiserver-job (MSJ) | jobs holding several servers at once — FCFS response time, saturated-system stability/throughput | exact CTMC / saturated product-form |
| Age of Information | M/M/1, M/G/1, preemptive-LCFS — average & peak AoI | closed-form + simulation |
| Vacations & warm-up | M/G/1 multiple vacations, N-policy, warm-up/cooling/delay (M/Ph/c) | Fuhrmann–Cooper, Takahashi–Takami |
| Negative customers | M/G/1 and M/G/c with RCS or disasters | exact / Takahashi–Takami |
| Reliability | M/G/1 with breakdowns & repairs | Avi-Itzhak–Naor |
| Matrix-analytic (MAP/PH) | MAP/PH/1, M/PH/1, PH/PH/1, MAP/M/c, MAP/PH/c — correlated (bursty) arrivals, single- & multi-server; MMPP fitting | QBD, logarithmic reduction |
| Batch Markovian arrivals | BMAP/M/1, BMAP/PH/1 — correlated batch traffic | level truncation |
| Retrial & abandonment | M/M/1 and M/G/1 retrial (orbit), Erlang-A (M/M/n+M) with staffing | exact / Falin–Templeton |
| GI/G approximations | GI/G/1, GI/G/m mean waiting time | Kingman, Krämer–Langenbach-Belz, Allen–Cunneen |
| Batch arrivals & bulk service | Mˣ/M/1 batch arrivals; M/M^[a,b]/1 bulk (batch) service — LLM inference batching | exact |
| Impatience & closed | M/M/1+M, Engset | exact |
| Parallel service | Fork-Join, Split-Join | Markovian / order statistics |
| Networks | open networks, priority networks, networks with negative customers, routing optimization | decomposition |

Every model comes with a plain-language explanation and a diagram in the
[illustrated model catalog](docs/models.md).

## Documentation & tutorials

- 📖 [Documentation](docs/README.md) — concepts, calculation and simulation guides (English; Russian versions available via in-page switchers)
- 🎓 [Jupyter tutorials](tutorials/README.md) — counter-intuitive queueing insights for engineers (the utilization trap, why variability dominates delay, multiserver jobs, Age of Information, …)
- 🗺 [Development roadmaps](docs/epics/README.md) — what's next (closed networks / MVA, multi-server retrial, polling)
- 🧪 [Tests](tests/) — every model validated against simulation; run with `pytest -m "not slow"`

## Applications

Capacity planning for cloud services and data centers · call-center staffing ·
manufacturing lines · telecom traffic · healthcare resource planning ·
scheduling research (SRPT/LAS with ML size predictions).

## Recent highlights

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
