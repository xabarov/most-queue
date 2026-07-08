# Numerical Methods Guide for Queueing System Analysis

[🇷🇺 Русская версия](calculation.ru.md)

This guide describes how to use the numerical methods module of the Most-Queue library for the analytical calculation of queueing system characteristics.

## Introduction

Numerical methods provide exact analytical results for queueing systems that have known mathematical solutions. Unlike simulation, calculation yields results instantly, without the need to model a large number of jobs.

## The BaseQueue Base Class

All calculation classes inherit from the `BaseQueue` base class, which provides a unified interface:

```python
from most_queue.theory.base_queue import BaseQueue
```

### Common API

All calculation classes follow a single pattern:

1. **Object creation** — initialization with the system parameters
2. **`set_sources()`** — configure the arrival process
3. **`set_servers()`** — configure service
4. **`run()`** — perform the calculation
5. **Retrieving results** — access the system characteristics

## Examples of Calculation Classes

### M/G/1 System

The `MG1Calc` class calculates the M/G/1 system (Poisson arrivals, arbitrary service time distribution).

```python
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution

# Create the calculator
mg1 = MG1Calc()

# Configure the arrival process (Poisson)
mg1.set_sources(l=0.5)  # λ = 0.5

# Configure service via distribution moments
# First create the H2 distribution parameters
h2_params = H2Distribution.get_params_by_mean_and_cv(2.0, 0.8)  # mean, cv

# Compute the distribution moments
b = H2Distribution.calc_theory_moments(h2_params, 5)
# b[0] - mean, b[1] - second moment, and so on

# Set the service moments
mg1.set_servers(b)

# Perform the calculation
results = mg1.run()

# Retrieve the results
print(f"Mean waiting time: {results.w[0]:.4f}")
print(f"Mean sojourn time: {results.v[0]:.4f}")
print(f"Utilization: {results.utilization:.4f}")
```

### GI/M/1 System

The `GIM1Calc` class calculates the GI/M/1 system (general arrival process, exponential service).

```python
from most_queue.theory.fifo.gi_m_1 import GIM1Calc
from most_queue.random.distributions import GammaDistribution

# Create the calculator
gim1 = GIM1Calc()

# Configure the arrival process via moments
# Create the gamma distribution parameters
gamma_params = GammaDistribution.get_params_by_mean_and_cv(2.0, 0.6)  # mean, cv

# Compute the inter-arrival time moments
a = GammaDistribution.calc_theory_moments(gamma_params)
gim1.set_sources(a)

# Configure service (exponential)
mu = 0.6  # service rate
gim1.set_servers(mu)

# Perform the calculation
results = gim1.run()

print(f"GI/M/1 results:")
print(f"  Mean waiting time: {results.w[0]:.4f}")
print(f"  Mean sojourn time: {results.v[0]:.4f}")
```

### M/M/c System

The `MMnrCalc` class calculates the M/M/c system (Poisson arrivals, exponential service, c channels).

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

# Create a calculator for M/M/3
mm3 = MMnrCalc(n=3)  # 3 channels

# Configure the arrival process
mm3.set_sources(l=2.0)  # λ = 2.0

# Configure service
mm3.set_servers(mu=1.0)  # μ = 1.0

# Perform the calculation
results = mm3.run(num_of_moments=4)

print(f"M/M/3 results:")
print(f"  Mean waiting time: {results.w[0]:.4f}")
print(f"  Utilization: {results.utilization:.4f}")
```

### H₂/M/c System

The `H2MnCalc` class handles a system with hyperexponential arrivals and exponential service (algorithm of §7.6.1):

```python
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.random.distributions import H2Distribution

calc = H2MnCalc(n=3)

h2_params = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2, is_clx=True)  # mean, cv
#
# For CV<1 use the complex fit: is_clx=True.
# Important: the `QsSim` simulator cannot generate H2 with complex parameters,
# so comparison against simulation is only possible for real-valued parameters.
calc.set_sources(h2_params)

calc.set_servers(b=2.0)  # mean service time
results = calc.run()

print(f"H2/M/3: p0={results.p[0]:.4f}, waiting time={results.w[0]:.4f}")
```

**CV<1 and validation:** for \(CV<1\), H₂ uses a complex fit (complex-valued approximation parameters).
`QsSim` does not support generating H₂ with complex parameters, so in such cases a "theory vs simulation"
comparison is best done via a `Gamma` simulation with the same mean and CV (see `tests/test_tt_vs_sim_gamma_cvl1.py` for an example).

### M/M/c/r System with a Bounded Queue

The same `MMnrCalc` class with the `r` parameter:

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

# M/M/3/20 system (3 channels, queue of up to 20 jobs)
mm3r = MMnrCalc(n=3, r=20)

mm3r.set_sources(l=2.0)
mm3r.set_servers(mu=1.0)

results = mm3r.run()
print(f"Loss probability: {1 - sum(results.p):.4f}")
```

## Working with Distribution Moments

### What Are Moments?

Distribution moments are numerical characteristics of a random variable:
- **First moment** (b[0]) — the expected value (mean)
- **Second moment** (b[1]) — the expected value of the square
- **Third moment** (b[2]) — the expected value of the cube
- And so on

### Computing Moments from Distributions

The library provides methods for computing the moments of various distributions:

```python
from most_queue.random.distributions import (
    H2Distribution, 
    GammaDistribution,
    ErlangDistribution
)

# H2 distribution (for CV<1 the complex fit is required)
h2_params = H2Distribution.get_params_by_mean_and_cv(2.0, 0.8, is_clx=True)  # mean, cv
b_h2 = H2Distribution.calc_theory_moments(h2_params, num=5)

# Gamma distribution
gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
b_gamma = GammaDistribution.calc_theory_moments(gamma_params, num=5)

# Erlang distribution
erlang_params = ErlangDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.5)
b_erlang = ErlangDistribution.calc_theory_moments(erlang_params, num=5)
```

### Example: Calculating M/G/1 with Different Distributions

```python
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution, GammaDistribution

arrival_rate = 0.4
service_mean = 2.5
service_cv = 0.7

# Option 1: H2 distribution
h2_params = H2Distribution.get_params_by_mean_and_cv(service_mean, service_cv)
b_h2 = H2Distribution.calc_theory_moments(h2_params, 5)

mg1_h2 = MG1Calc()
mg1_h2.set_sources(l=arrival_rate)
mg1_h2.set_servers(b_h2)
results_h2 = mg1_h2.run()

# Option 2: Gamma distribution
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
b_gamma = GammaDistribution.calc_theory_moments(gamma_params, 5)

mg1_gamma = MG1Calc()
mg1_gamma.set_sources(l=arrival_rate)
mg1_gamma.set_servers(b_gamma)
results_gamma = mg1_gamma.run()

# Compare the results
print(f"H2: mean waiting time = {results_h2.w[0]:.4f}")
print(f"Gamma: mean waiting time = {results_gamma.w[0]:.4f}")
```

## Result Structure

### QueueResults

All calculation classes return a `QueueResults` object:

```python
@dataclass
class QueueResults:
    v: list[float] | None = None      # sojourn time moments
    w: list[float] | None = None      # waiting time moments
    p: list[float] | None = None      # state probabilities
    pi: list[float] | None = None     # probabilities at arrival instants
    utilization: float | None = None   # utilization
    duration: float = 0.0              # calculation time in seconds
```

### Accessing Results

```python
results = calc.run()

# Waiting time moments
w_mean = results.w[0]      # mean waiting time
w_second = results.w[1]    # second moment

# Sojourn time moments
v_mean = results.v[0]       # mean sojourn time

# State probabilities
p0 = results.p[0]          # idle probability
p1 = results.p[1]          # probability of 1 job in the system

# Utilization
ro = results.utilization

# Calculation time
calc_time = results.duration
```

## Comparing Calculation and Simulation

To verify correctness, you can compare the calculation results against a simulation:

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution, H2Params
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

arrival_rate = 0.5
service_mean = 2.0
service_cv = 0.8

# H2 distribution parameters
h2_params = H2Distribution.get_params_by_mean_and_cv(service_mean, service_cv)
b = H2Distribution.calc_theory_moments(h2_params, 5)

# Calculation
mg1_calc = MG1Calc()
mg1_calc.set_sources(l=arrival_rate)
mg1_calc.set_servers(b)
calc_results = mg1_calc.run()

# Simulation
qs = QsSim(num_of_channels=1)
qs.set_sources(arrival_rate, "M")
qs.set_servers(h2_params, "H")
sim_results = qs.run(50000)

# Comparison
print("Comparison of waiting time moments:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nComparison of sojourn time moments:")
print_sojourn_moments(sim_results.v, calc_results.v)
```

## Available Calculation Classes

### FIFO Systems

- **`MG1Calc`** — M/G/1 system
- **`GIM1Calc`** — GI/M/1 system
- **`GiMn`** — GI/M/c system
- **`MMnrCalc`** — M/M/c/r system
- **`MDnCalc`** — M/D/c system
- **`EkDnCalc`** — E_k/D/c system
- **`MGnCalc`** — M/G/c system (Takahashi-Takami method)
- **`MG1SrptCalc`**, **`MG1SjfCalc`**, **`MG1PsjfCalc`**, **`MG1SpjfCalc`** — M/G/1 with size-based disciplines (see below)

## Size-Based M/G/1 Calculators

For a detailed description of the numerical implementation (grid, `cumulative_trapezoid`, interpolation, outer integral via Simpson vs `quad` for SPJF) and the comparison against simulation, see the **[SRPT / SPJF: Methods and Verification](srpt_spjf_methods.md)** page.

Single-channel **M/G/1** with a Poisson arrival process (rate \(\lambda\)) and an arbitrary job size distribution \(X\) with density \(f(x)\), CDF \(F(x)\), and service moments \(E[S]=b_0\), \(E[S^2]=b_1\). Partial load contributed by jobs of size at most \(x\): \(\rho_x = \lambda \int_0^x t f(t)\,dt\).

### SRPT (`MG1SrptCalc`)

Conditional mean sojourn time for a job of size \(x\) (Schrage–Miller, 1966):

$$
\mathbb{E}[T^{\mathrm{SRPT}}(x)]
= \frac{\lambda \int_0^x t^2 f(t)\,dt + \lambda x^2 (1-F(x))}{2(1-\rho_x)^2}
+ \int_0^x \frac{dt}{1-\rho_t}.
$$

Unconditional: \(\mathbb{E}[T^{\mathrm{SRPT}}] = \int_0^\infty f(x)\,\mathbb{E}[T^{\mathrm{SRPT}}(x)]\,dx\), \(\mathbb{E}[W] = \mathbb{E}[T] - E[S]\). Implementation: numerical grid + `simpson` over \(x\) (stable under high load).

```python
from most_queue.theory.srpt import MG1SrptCalc

srpt = MG1SrptCalc()
srpt.set_sources(0.5)
srpt.set_servers(1.0, "M")  # Exp(rate): mean service 1
r = srpt.run()
print(r.v[0], r.w[0])
```

### SJF (`MG1SjfCalc`)

Non-preemptive priority by size (Conway–Maxwell–Miller):

$$
\mathbb{E}[W^{\mathrm{SJF}}(x)] = \frac{\lambda\, \mathbb{E}[S^2]}{2(1-\rho_x)^2}, \qquad
\mathbb{E}[T^{\mathrm{SJF}}] = \mathbb{E}[W^{\mathrm{SJF}}] + E[S].
$$

### PSJF (`MG1PsjfCalc`)

$$
\mathbb{E}[T^{\mathrm{PSJF}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\,dt}{2(1-\rho_x)^2} + \frac{x}{1-\rho_x}.
$$

### SPJF with Predictions (`MG1SpjfCalc`)

The joint density of \((X,Y)\) is defined by a **predictor** (protocol in `most_queue.theory.srpt.utils.predictor`). Effective load contributed by jobs with prediction \(\le y\): \(\rho'_y\). Then

$$
\mathbb{E}[W^{\mathrm{SPJF}}(y)] = \frac{\lambda\, \mathbb{E}[S^2]}{2(1-\rho'_y)^2}, \qquad
\mathbb{E}[W^{\mathrm{SPJF}}] = \int g_Y(y)\,\mathbb{E}[W^{\mathrm{SPJF}}(y)]\,dy.
$$

Predictor implementations: **`PerfectPredictor`** (\(Y=X\), coincides with SJF), **`ExpNoisePredictor`** (\(Y\mid X=x \sim \mathrm{Exp}(1/x)\)), **`LognormalNoisePredictor`**.

```python
from most_queue.theory.srpt import MG1SpjfCalc
from most_queue.theory.srpt.utils.predictor import ExpNoisePredictor

calc = MG1SpjfCalc()
calc.set_sources(0.5)
calc.set_servers(1.0, "M")
calc.set_predictor(ExpNoisePredictor())
r = calc.run()
```

Top-level package import: `from most_queue.theory.srpt import MG1SrptCalc, ExpNoisePredictor, ...`.

## Extending the Takahashi-Takami Method

The `MGnCalc` class implements the Takahashi-Takami numerical method for calculating multi-channel M/G/c systems. This class is designed for easy extension, allowing you to create custom calculation methods for various types of queueing systems.

### Extensibility Architecture

`MGnCalc` uses the Template Method pattern, splitting the algorithm into overridable hooks:

- **Matrix construction methods** — define the structure of the transition matrices
- **Iteration hooks** — allow customizing the algorithm's logic
- **Result calculation methods** — configure how the system characteristics are computed

### Basic Usage of MGnCalc

```python
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.random.distributions import H2Distribution

calc = MGnCalc(n=5)  # 5 channels
calc.set_sources(l=2.0)

h2_params = H2Distribution.get_params_by_mean_and_cv(2.0, 1.2, is_clx=True)
calc.set_servers(h2_params)

results = calc.run()
```

### Creating a Custom Extension

To create your own calculator based on the Takahashi-Takami method:

1. Inherit from `MGnCalc`
2. Override the matrix construction methods (if needed)
3. Override the iteration hooks (if needed)
4. Override the result calculation methods (if needed)

**Example:** Adding negative customers

```python
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
import numpy as np

class CustomNegativeQueueCalc(MGnCalc):
    def __init__(self, n, buffer=None, calc_params=None):
        super().__init__(n, buffer, calc_params)
        self.l_neg = None  # rate of negative customers
    
    def set_sources(self, l_pos, l_neg):
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.l = l_pos  # base rate
        self.is_sources_set = True
    
    def _build_big_d_matrix(self, num):
        # Override the D matrix to account for negative customers
        base_matrix = super()._build_big_d_matrix(num)
        # Add negative customers to the diagonal elements
        for i in range(base_matrix.shape[0]):
            base_matrix[i, i] += self.l_neg
        return base_matrix
```

### Extension Points

**Required to override (if the matrix structure changes):**
- `fill_cols()` — define the column structure for each level
- `_build_big_a_matrix(num)` — upward transition matrix (arrivals)
- `_build_big_b_matrix(num)` — downward transition matrix (service)
- `_build_big_d_matrix(num)` — diagonal elements of the matrix

**Optional to override:**
- `_pre_run_setup()` — preparation before the main loop
- `_update_level_j(j)` — update variables for level j
- `_update_level_0()` — update level 0
- `_calculate_p()` — calculate the state probabilities
- `get_results()` — assemble the results

### Examples of Existing Extensions

- **`MGnNegativeRCSCalc`** — system with negative customers and the RCS discipline
- **`MGnNegativeDisasterCalc`** — system with negative customers (disasters)
- **`MH2nH2Warm`** — system with warm-up periods
- **`MPhNPrty`** — system with priorities

For detailed documentation and examples, see `most_queue/theory/fifo/takahasi_base.py`.

### Priority Systems

- **`MG1Preemptive`** — M/G/1 with preemptive priority
- **`MG1NonPreemptive`** — M/G/1 with non-preemptive priority
- **`MGnInvarApproximation`** — M/G/c with priorities (invariant relations method)

### Specialized Systems

- **`BatchMM1`** — M^x/M/1 with batch arrivals
- **`EngsetCalc`** — closed Engset system
- **`ForkJoinMarkovianCalc`** — Fork-Join M/M/c system
- And others (see [Queueing Models](models.md))

## Calculation Parameters

### CalcParams

Some classes accept calculation parameters:

```python
from most_queue.theory.calc_params import CalcParams

calc_params = CalcParams(
    p_num=100,              # number of state probabilities to calculate
    tolerance=1e-6,         # computation tolerance
    approx_distr="gamma"    # distribution type for approximation
)

mg1 = MG1Calc(calc_params=calc_params)
```

## Usage Tips

1. **Check stability** — make sure ρ < 1 before calculating
2. **Use enough moments** — accuracy requires several distribution moments
3. **Compare against simulation** — verify results on simple cases
4. **Handle errors** — some systems may not have a solution
5. **Use appropriate distributions** — pick distributions that match your real data

## Performance

Numerical methods are usually much faster than simulation:

```python
import time

# Calculation
start = time.time()
results = calc.run()
calc_time = time.time() - start

# Simulation
start = time.time()
results = qs.run(50000)
sim_time = time.time() - start

print(f"Calculation: {calc_time:.4f} s")
print(f"Simulation: {sim_time:.4f} s")
print(f"Speedup: {sim_time/calc_time:.1f}x")
```

---

**See also:**
- [Queueing System Simulation](simulation.md) — discrete-event modeling
- [Distributions](distributions.md) — distribution reference
- [Queueing Models](models.md) — catalog of supported models
