# Quick Start

[🇷🇺 Русская версия](getting_started.ru.md)

This guide will help you get up and running with the Most-Queue library.

## Installation

Install the library via pip:

```bash
pip install most-queue
```

Or install from source:

```bash
git clone https://github.com/xabarov/most-queue.git
cd most-queue
pip install -e .
```

## Requirements

- Python >= 3.9
- NumPy
- SciPy >= 1.13.0, < 2.0
- Other dependencies are listed in `requirements.txt`

## First Example: M/M/1 System

Let us consider the simplest queueing system, M/M/1:
- **M** — Poisson job arrival process
- **M** — exponential service time distribution
- **1** — a single server (channel)

### Simulation

```python
from most_queue.sim.base import QsSim

# Create a simulator with one channel
qs = QsSim(num_of_channels=1)

# Configure the arrival process
# Parameter: arrival rate (mean number of jobs per unit of time)
qs.set_sources(0.5, "M")  # λ = 0.5

# Configure the service
# Parameter: service rate (mean number of service completions per unit of time)
qs.set_servers(1.0, "M")  # μ = 1.0

# Run the simulation
# Parameter: number of jobs to process
results = qs.run(10000)

# Print the results
print(f"Mean waiting time in queue: {results.w[0]:.4f}")
print(f"Mean sojourn time in system: {results.v[0]:.4f}")
print(f"Utilization factor: {results.utilization:.4f}")
```

### Numerical Calculation

For the same system, exact analytical results can be obtained:

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

# Create an M/M/1 calculator (a special case of M/M/c)
calc = MMnrCalc(n=1)  # n - number of channels

# Configure the arrival process
calc.set_sources(l=0.5)  # l - arrival rate

# Configure the service
calc.set_servers(mu=1.0)  # mu - service rate

# Perform the calculation
results = calc.run()

# Print the results
print(f"Mean waiting time: {results.w[0]:.4f}")
print(f"Mean sojourn time: {results.v[0]:.4f}")
print(f"Utilization factor: {results.utilization:.4f}")
```

## Comparing Simulation and Calculation

One of the strengths of the library is the ability to compare simulation results against analytical calculations:

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

# System parameters
arrival_rate = 0.5
service_rate = 1.0
num_channels = 1
num_jobs = 10000

# Simulation
qs = QsSim(num_channels)
qs.set_sources(arrival_rate, "M")
qs.set_servers(service_rate, "M")
sim_results = qs.run(num_jobs)

# Calculation
calc = MMnrCalc(n=num_channels)
calc.set_sources(l=arrival_rate)
calc.set_servers(mu=service_rate)
calc_results = calc.run()

# Compare the results
print("Comparison of waiting time moments:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nComparison of sojourn time moments:")
print_sojourn_moments(sim_results.v, calc_results.v)
```

## Project Structure

The library consists of the following main modules:

### `most_queue.sim` — Simulation

Module for discrete-event simulation of queueing systems:
- `base.py` — base simulation class `QsSim`
- `priority.py` — systems with priorities
- `fork_join.py` — Fork-Join systems
- `networks/` — simulation of queueing networks
- And other specialized classes

### `most_queue.theory` — Numerical Methods

Module for analytical calculation of queueing systems:
- `base_queue.py` — base class for calculations
- `fifo/` — systems with the FIFO discipline
- `priority/` — systems with priorities
- `networks/` — calculation of queueing networks
- And other modules

### `most_queue.random` — Distributions

Module for working with random distributions:
- `distributions.py` — distribution classes
- `utils/` — utilities for working with distributions

### `most_queue.io` — Input/Output

Module for visualization and output of results:
- `tables.py` — formatted table output
- `plots.py` — plotting

## When to Use Simulation vs. Calculation?

### Use simulation (`sim`) when:
- No analytical solution exists for your model
- You need to model complex system behavior
- Transient analysis is required
- The model contains non-standard elements

### Use calculation (`theory`) when:
- An analytical solution exists
- High accuracy of results is needed
- Fast results are required
- You need to validate the correctness of a simulation

## Next Steps

1. Learn the [core concepts](concepts.md) of queueing theory
2. Read the [simulation guide](simulation.md)
3. Study the [numerical methods](calculation.md)
4. Look through the [usage examples](examples.md)

## Useful Tips

- Start with simple models (M/M/1, M/M/c)
- Use simulation to cross-check calculation results
- To get stable simulation results, use a sufficiently large number of jobs (typically 10000+)
- Keep an eye on the utilization factor: the system is stable when ρ < 1
