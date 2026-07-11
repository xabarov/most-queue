# Tutorials — counter-intuitive queueing insights for engineers

Short, runnable notebooks. Each one uses the library's own calculators (paired with simulation to
validate) to demonstrate a result that surprises most engineers — the kind of thing that changes how
you size systems, schedule work, or reason about latency.

### Capacity & latency

| Notebook | The insight |
|---|---|
| [utilization_trap.ipynb](utilization_trap.ipynb) | Why the **last 10% of utilization is the most expensive** (latency blows up like `1/(1-ρ)`), and why **pooling servers** beats many small queues at the same load. |
| [variability_matters.ipynb](variability_matters.ipynb) | Same *average* service time, **10×+ different delay** — waiting scales with the service-time variability `C²`, not just the mean (Pollaczek–Khinchine). |
| [littles_law.ipynb](littles_law.ipynb) | `L = λW` holds for **any** system, discipline and distribution — turn the metric you can't measure into the two you can (latency from concurrency). |

### Scheduling

| Notebook | The insight |
|---|---|
| [disciplines_comparison.ipynb](disciplines_comparison.ipynb) | The **scheduling discipline** alone (FCFS, PS, LCFS, FB, SJF, PSJF, SRPT, predicted-size) moves mean response time a lot — and reshapes *who* pays. |
| [srpt_basics.ipynb](srpt_basics.ipynb) | **Size-aware scheduling** (SRPT/SJF) slashes response time; predictions can stand in for exact sizes, with graceful degradation. |
| [conservation_law.ipynb](conservation_law.ipynb) | **Priority is a zero-sum game**: blind class-priority conserves the load-weighted total wait `Σ ρ_i W_i` — it redistributes delay, it can't remove it. |

### Modelling pitfalls & modern systems

| Notebook | The insight |
|---|---|
| [pasta.ipynb](pasta.ipynb) | **Only Poisson arrivals see time averages** — with regular/bursty traffic, what an arriving request experiences differs from the average, biasing measurements and formulas. |
| [age_of_information.ipynb](age_of_information.ipynb) | **Sending updates faster can make your data staler** — freshness (Age of Information) is U-shaped in the update rate, with an optimum in the middle. |
| [multiserver_jobs.ipynb](multiserver_jobs.ipynb) | Jobs that grab **several servers at once** (GPU/core reservations): you **can't use `k·μ`** — packing and blocking cost 20–30% of capacity. |
| [map_ph_correlation.ipynb](map_ph_correlation.ipynb) | **Correlated (bursty) arrivals** blow up delay even at the *same* average rate — the MAP/PH matrix-analytic stack captures it. |

Run any of them with Jupyter after `pip install most-queue` (plus `jupyter`, `matplotlib`, `pandas`).
Every model shown here is documented in the [illustrated model catalog](../docs/models.md).
