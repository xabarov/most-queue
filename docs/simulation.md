# Queueing System Simulation Guide

[🇷🇺 Русская версия](simulation.ru.md)

This guide describes how to use the simulation module of the Most-Queue library for modeling queueing systems.

## Introduction

Simulation (discrete-event modeling) makes it possible to model the behavior of queueing systems that have no analytical solutions or that have a complex structure. The Most-Queue library provides the `QsSim` class for simulating various types of queueing systems.

## The QsSim Base Class

### Creating a Simulator

```python
from most_queue.sim.base import QsSim

# Create a simulator with the given number of channels
qs = QsSim(num_of_channels=3)

# With additional parameters
qs = QsSim(
    num_of_channels=3,      # number of service channels
    buffer=50,              # maximum queue length (None = unbounded)
    verbose=True,           # print progress information
    buffer_type="list"      # buffer type: "list" or "deque"
)
```

### Constructor Parameters

- **`num_of_channels`** (int) — number of service channels (required)
- **`buffer`** (int, optional) — maximum queue length. If `None`, the queue is unbounded
- **`verbose`** (bool) — print detailed information during the simulation (defaults to `True`)
- **`buffer_type`** (str) — queue implementation type: `"list"` or `"deque"`

## Configuring the Arrival Process

### The set_sources() Method

The `set_sources()` method configures the arrival process of jobs entering the system.

```python
qs.set_sources(params, kendall_notation="M")
```

**Parameters:**
- **`params`** — parameters of the inter-arrival time distribution
- **`kendall_notation`** (str) — Kendall notation of the distribution

### Arrival Configuration Examples

#### Exponential Distribution (M)

```python
# Poisson arrival process with rate λ = 0.5
qs.set_sources(0.5, "M")
```

#### Hyperexponential Distribution (H)

```python
from most_queue.random.distributions import H2Distribution, H2Params

# Create H2 distribution parameters
h2_params = H2Params(p1=0.3, mu1=1.0, mu2=2.0)
qs.set_sources(h2_params, "H")
```

#### Gamma Distribution

```python
from most_queue.random.distributions import GammaDistribution, GammaParams

# Create parameters from the mean and coefficient of variation
gamma_params = GammaDistribution.get_params_by_mean_and_cv(
    mean=2.0,      # mean inter-arrival time
    cv=0.5         # coefficient of variation
)
qs.set_sources(gamma_params, "Gamma")
```

#### Deterministic Distribution (D)

```python
# Constant interval between jobs
qs.set_sources(2.0, "D")  # interval = 2.0 time units
```

## Configuring Service

### The set_servers() Method

The `set_servers()` method configures the service time distribution for all channels.

```python
qs.set_servers(params, kendall_notation="M")
```

**Parameters:**
- **`params`** — parameters of the service time distribution
- **`kendall_notation`** (str) — distribution notation

### Service Configuration Examples

#### Exponential Service (M)

```python
# Service rate μ = 1.0
qs.set_servers(1.0, "M")
```

#### Gamma-Distributed Service Time

```python
from most_queue.random.distributions import GammaDistribution

# Create parameters from the mean service time and CV
service_mean = 2.5
service_cv = 0.8
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
qs.set_servers(gamma_params, "Gamma")
```

## Running the Simulation

### The run() Method

Once the arrival process and service are configured, start the simulation:

```python
results = qs.run(num_of_jobs)
```

**Parameters:**
- **`num_of_jobs`** (int) — number of jobs to process

**Returns:**
- A `QueueResults` object with the simulation results

### Complete Example

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import GammaDistribution

# Create an M/G/3 simulator
qs = QsSim(num_of_channels=3)

# Configure a Poisson arrival process
qs.set_sources(0.8, "M")  # λ = 0.8

# Configure a gamma-distributed service time
service_mean = 3.0
service_cv = 0.6
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
qs.set_servers(gamma_params, "Gamma")

# Run the simulation
results = qs.run(50000)

# Retrieve the results
print(f"Mean waiting time: {results.w[0]:.4f}")
print(f"Mean sojourn time: {results.v[0]:.4f}")
print(f"Utilization: {results.utilization:.4f}")
```

## Simulation Results

### The QueueResults Structure

The results object contains the following attributes:

- **`w`** (list[float]) — raw moments of the waiting time in the queue
  - `w[0]` — mean waiting time (first moment)
  - `w[1]` — second moment
  - `w[2]` — third moment
  - and so on

- **`v`** (list[float]) — raw moments of the sojourn time in the system
  - `v[0]` — mean sojourn time
  - `v[1]` — second moment
  - and so on

- **`p`** (list[float]) — system state probabilities
  - `p[0]` — probability that there are 0 jobs in the system
  - `p[1]` — probability that there is 1 job in the system
  - and so on

- **`utilization`** (float) — system utilization (0 ≤ ρ ≤ 1)

- **`duration`** (float) — simulation run time in seconds

### Accessing Results Directly

After `run()` completes, the results are also available through the attributes of the `qs` object:

```python
qs.run(10000)

# Direct access to the results
w_sim = qs.w          # waiting time moments
v_sim = qs.v          # sojourn time moments
p_sim = qs.get_p()    # state probabilities
ro = qs.load          # utilization
```

## Usage Examples

### Example 1: M/M/1 System

```python
from most_queue.sim.base import QsSim

qs = QsSim(num_of_channels=1)
qs.set_sources(0.5, "M")   # λ = 0.5
qs.set_servers(1.0, "M")   # μ = 1.0

results = qs.run(10000)

print(f"M/M/1 results:")
print(f"  Mean waiting time: {results.w[0]:.4f}")
print(f"  Mean sojourn time: {results.v[0]:.4f}")
print(f"  Utilization: {results.utilization:.4f}")
```

### Example 2: GI/M/1 System

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import GammaDistribution

qs = QsSim(num_of_channels=1)

# Gamma-distributed inter-arrival times
arrival_mean = 2.0
arrival_cv = 0.7
gamma_arrival = GammaDistribution.get_params_by_mean_and_cv(arrival_mean, arrival_cv)
qs.set_sources(gamma_arrival, "Gamma")

# Exponential service
qs.set_servers(0.6, "M")  # μ = 0.6

results = qs.run(20000)
print(f"GI/M/1 results:")
print(f"  Mean waiting time: {results.w[0]:.4f}")
```

### Example 3: M/M/c/r System with a Bounded Queue

```python
from most_queue.sim.base import QsSim

# System with 3 channels and a maximum queue of 20
qs = QsSim(num_of_channels=3, buffer=20)

qs.set_sources(2.0, "M")   # λ = 2.0
qs.set_servers(1.0, "M")   # μ = 1.0

results = qs.run(50000)

print(f"M/M/3/20 results:")
print(f"  Mean waiting time: {results.w[0]:.4f}")
print(f"  Number of lost jobs: {qs.dropped}")
print(f"  Loss probability: {qs.dropped / qs.arrived:.4f}")
```

## Comparing Against Analytical Results

To verify the correctness of a simulation, you can compare the results against analytical calculations:

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

# System parameters
arrival_rate = 0.8
service_rate = 1.0
num_channels = 2
num_jobs = 30000

# Simulation
qs = QsSim(num_channels)
qs.set_sources(arrival_rate, "M")
qs.set_servers(service_rate, "M")
sim_results = qs.run(num_jobs)

# Analytical calculation
calc = MMnrCalc(n=num_channels)
calc.set_sources(l=arrival_rate)
calc.set_servers(mu=service_rate)
calc_results = calc.run()

# Comparison
print("Comparison of waiting time moments:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nComparison of sojourn time moments:")
print_sojourn_moments(sim_results.v, calc_results.v)
```

## Simulation Parameters

### Number of Jobs

To obtain stable results, use a sufficiently large number of jobs:
- Minimum: 10,000 jobs
- Recommended: 50,000+ jobs
- For systems under high load: 100,000+ jobs

### Utilization

Before running a simulation, you can check the utilization:

```python
qs = QsSim(num_of_channels=2)
qs.set_sources(1.5, "M")
qs.set_servers(1.0, "M")

load = qs.calc_load()
print(f"Utilization: {load:.4f}")

if load >= 1.0:
    print("Warning: the system is overloaded!")
else:
    print("The system is stable")
    results = qs.run(50000)
```

## Interpreting the Results

### Waiting Time vs Sojourn Time

- **Waiting time** (`w`) — time from a job's arrival until service begins
- **Sojourn time** (`v`) — total time from arrival until service completion
- Relationship: `v = w + service_time`

### State Probabilities

The probabilities `p[i]` show the fraction of time the system spends in the state with `i` jobs:

```python
results = qs.run(50000)
p = results.p

print(f"Idle probability: {p[0]:.4f}")
print(f"Probability of 1 job in the system: {p[1]:.4f}")
print(f"Probability of 2+ jobs in the system: {sum(p[2:]):.4f}")
```

### Utilization

Utilization shows the fraction of time the channels are busy:
- ρ < 1 — the system is stable
- ρ = 1 — the system is on the boundary of stability
- ρ > 1 — the system is overloaded (the queue grows)

## Specialized Simulation Classes

The library also provides specialized classes for various types of queueing systems:

- **`PriorityQueueSimulator`** — systems with priorities (see [Priority Systems](priorities.md))
- **`ForkJoinSim`** — Fork-Join systems
- **`QueueingSystemBatchSim`** — systems with batch arrivals
- **`NetworkSimulator`** — queueing networks (see [Queueing Networks](networks.md))
- **`SizeBasedQsSim`** — single-channel M/G/1 with size-based or prediction-based disciplines (see below)

## SizeBasedQsSim (Size-Based Disciplines)

The theoretical formulas for SRPT/SJF/PSJF/SPJF, the numerical integration scheme used in the calculators, and the typical procedure for comparison against simulation are described on the **[SRPT / SPJF: Methods and Verification](srpt_spjf_methods.md)** page.

The `most_queue.sim.size_based.SizeBasedQsSim` class inherits from `QsSim`, but:

1. **Size on arrival** — for all disciplines except `FCFS`, `_sample_size` is called when a job is created: `Task.original_size` is sampled from the same distribution as in `set_servers`, the full amount of work is written to `Task.service_remaining`, and optionally `Task.predicted_size` is set via a predictor.
2. **Queue** — a min-heap keyed by rank (`PrioritySizeQueue`): for SRPT/SPRPT, before preemption `service_remaining` is updated with the actual remaining work on the channel.
3. **Predictor** — `set_predictor(obj)` for `SPJF`, `PSPJF`, `SPRPT`; an object with a method `predict(true_size, rng) -> float`. Built-in **`PerfectSimPredictor`**: \(Y=X\). For noise in the simulation: `most_queue.sim.utils.predictor.ExpNoiseSimPredictor`, `LognormalNoiseSimPredictor`.
4. **Slowdown** — with `track_slowdown=True`, after each service completion the ratio \(T/X\) is appended to a list; read a copy with `get_slowdown()`.

Supported `discipline` values: `"FCFS"`, `"SJF"`, `"PSJF"`, `"SRPT"`, `"SPJF"`, `"PSPJF"`, `"SPRPT"`. Currently only **`num_of_channels=1`** is supported.

**SRPT example (H₂ service, Poisson arrivals):**

```python
import numpy as np
from most_queue.random.distributions import H2Distribution
from most_queue.sim.size_based import SizeBasedQsSim

sim = SizeBasedQsSim(1, discipline="SRPT", verbose=False)
sim.generator = np.random.default_rng(42)
h2 = H2Distribution.get_params_by_mean_and_cv(0.7, 1.2)
sim.set_servers(h2, "H")
sim.set_sources(1.0, "M")
res = sim.run(100_000)
print(res.v[0], res.w[0])
```

**SPJF example with exponential prediction noise:**

```python
import numpy as np
from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.sim.utils.predictor import ExpNoiseSimPredictor

sim = SizeBasedQsSim(1, discipline="SPJF", verbose=False)
sim.set_predictor(ExpNoiseSimPredictor())
sim.generator = np.random.default_rng(7)
sim.set_servers(1.0, "M")
sim.set_sources(0.5, "M")
res = sim.run(50_000)
```

For a comparison against `QsSim` in `FCFS` mode, see the test `tests/test_size_based_fcfs_regression.py`.

## Usage Tips

1. **Start with simple models** — verify things work on M/M/1 before moving to complex systems
2. **Use enough jobs** — accurate results require a lot of data
3. **Check the utilization** — make sure the system is stable
4. **Compare against analytics** — whenever possible, verify results with a calculation
5. **Analyze the state probabilities** — they give a complete picture of system behavior

---

**See also:**
- [Numerical Methods](calculation.md) — analytical calculations
- [Distributions](distributions.md) — distribution reference
- [Usage Examples](examples.md) — extended examples
