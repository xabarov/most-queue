# Tutorials — counter-intuitive queueing insights for engineers

[🇷🇺 Русская версия](README.ru.md)

Short, runnable notebooks. Each one uses the library's own calculators (paired with simulation to
validate) to demonstrate a result that surprises most engineers — the kind of thing that changes how
you size systems, schedule work, or reason about latency.

### Capacity & latency

| Notebook | The insight |
|---|---|
| [utilization_trap.ipynb](utilization_trap.ipynb) | Why the **last 10% of utilization is the most expensive** (latency blows up like `1/(1-ρ)`), and why **pooling servers** beats many small queues at the same load. |
| [variability_matters.ipynb](variability_matters.ipynb) | Same *average* service time, **10×+ different delay** — waiting scales with the service-time variability `C²`, not just the mean (Pollaczek–Khinchine). |
| [littles_law.ipynb](littles_law.ipynb) | `L = λW` holds for **any** system, discipline and distribution — turn the metric you can't measure into the two you can (latency from concurrency). |
| [heavy_traffic.ipynb](heavy_traffic.ipynb) | Near saturation the waiting time becomes **exponential for any service distribution** — the queue "forgets" the shape; only the first two moments matter. |

### Scheduling

| Notebook | The insight |
|---|---|
| [disciplines_comparison.ipynb](disciplines_comparison.ipynb) | The **scheduling discipline** alone (FCFS, PS, LCFS, FB, SJF, PSJF, SRPT, predicted-size) moves mean response time a lot — and reshapes *who* pays. |
| [srpt_basics.ipynb](srpt_basics.ipynb) | **Size-aware scheduling** (SRPT/SJF) slashes response time; predictions can stand in for exact sizes, with graceful degradation. |
| [conservation_law.ipynb](conservation_law.ipynb) | **Priority is a zero-sum game**: blind class-priority conserves the load-weighted total wait `Σ ρ_i W_i` — it redistributes delay, it can't remove it. |

### Sampling & paradoxes

| Notebook | The insight |
|---|---|
| [pasta.ipynb](pasta.ipynb) | **Only Poisson arrivals see time averages** — with regular/bursty traffic, what an arriving request experiences differs from the average, biasing measurements and formulas. |
| [inspection_paradox.ipynb](inspection_paradox.ipynb) | A random observer **waits longer than half the average gap** — you over-sample the long intervals (`E[X²]/2E[X]`). The bus-waiting paradox. |

### Load balancing & modern systems

| Notebook | The insight |
|---|---|
| [power_of_two_choices.ipynb](power_of_two_choices.ipynb) | Sampling **two** random servers and taking the shorter is dramatically better than one and nearly as good as polling all `N` — the "power of two choices" behind modern load balancers. |
| [multiserver_jobs.ipynb](multiserver_jobs.ipynb) | Jobs that grab **several servers at once** (GPU/core reservations): you **can't use `k·μ`** — packing and blocking cost 20–30% of capacity. |
| [time_varying_load.ipynb](time_varying_load.ipynb) | **Peak congestion isn't at peak demand** — a busy system lags the load, so staffing to the instantaneous rate arrives late. When to use PSA vs MOL for time-varying `Mt/M/c`. |
| [age_of_information.ipynb](age_of_information.ipynb) | **Sending updates faster can make your data staler** — freshness (Age of Information) is U-shaped in the update rate, with an optimum in the middle. |
| [map_ph_correlation.ipynb](map_ph_correlation.ipynb) | **Correlated (bursty) arrivals** blow up delay even at the *same* average rate — the MAP/PH matrix-analytic stack captures it. |

### Queueing networks

| Notebook | The insight |
|---|---|
| [closed_network_capacity.ipynb](closed_network_capacity.ipynb) | **Users self-throttle**: a closed network doesn't die at ρ=1 — the capacity knee is at `N* = 1 + Z/b`, computable without any solver, and past it response grows linearly. The open-model approximation diverges exactly where you need the answer. |
| [network_bottleneck.ipynb](network_bottleneck.ipynb) | Rank services by **demand `e·b`** (visits × time per visit), not per-call latency: the bottleneck is often the *fast* node everyone calls three times, and optimizing anything else is capped at a few percent (Amdahl for networks). |
| [poisson_illusion.ipynb](poisson_illusion.ipynb) | **Internal network traffic is not Poisson** — departures inherit service variability, so plain node-by-node decomposition hides ~20% of latency on a bursty tandem; QNA's two-moment flows recover it. Burke's theorem (exponential service) is the only legal case. |

Run any of them with Jupyter after `pip install most-queue` (plus `jupyter`, `matplotlib`, `pandas`).
Every model shown here is documented in the [illustrated model catalog](../docs/models.md).
