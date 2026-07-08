# Extended Usage Examples

[🇷🇺 Русская версия](examples.ru.md)

This section contains practical examples of using the Most-Queue library to solve real-world problems.

## Example 1: Modeling a Call Center

### Problem

Model a call center with several operators and two types of calls: regular and priority (VIP).

### Solution

```python
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_sojourn_multiclass

# Call center parameters
num_operators = 10
num_classes = 2  # regular and VIP calls

# Arrival rates
# Regular calls: 5 calls per minute
# VIP calls: 1 call per minute
arrival_rates = [5.0 / 60, 1.0 / 60]  # convert to seconds

# Mean service times
# Regular: 3 minutes
# VIP: 5 minutes (more complex requests)
service_means = [3.0 * 60, 5.0 * 60]  # in seconds
service_cv = 0.7

# Create a simulator with non-preemptive priority
# (VIP calls have priority, but a call in progress is not interrupted)
qs = PriorityQueueSimulator(num_operators, num_classes, "NP")

# Configure the arrival processes
sources = []
servers_params = []

for j in range(num_classes):
    sources.append({"type": "M", "params": arrival_rates[j]})
    
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=service_means[j],
        cv=service_cv
    )
    servers_params.append({"type": "Gamma", "params": gamma_params})

qs.set_sources(sources)
qs.set_servers(servers_params)

# Simulate 10000 calls
qs.run(10000)

# Analyze the results
print("Call center simulation results:")
print(f"Regular calls:")
print(f"  Mean waiting time: {qs.v[0][0] / 60:.2f} minutes")
print(f"  Mean sojourn time: {qs.v[0][0] / 60:.2f} minutes")

print(f"\nVIP calls:")
print(f"  Mean waiting time: {qs.v[1][0] / 60:.2f} minutes")
print(f"  Mean sojourn time: {qs.v[1][0] / 60:.2f} minutes")

# Check the load
total_load = sum(arrival_rates[i] * service_means[i] for i in range(num_classes))
utilization = total_load / num_operators
print(f"\nUtilization: {utilization:.2%}")
```

## Example 2: Analyzing Cloud Infrastructure

### Problem

Analyze the performance of a cloud server with several virtual machines processing requests with different characteristics.

### Solution

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import H2Distribution
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

# Server parameters
num_vms = 8  # number of virtual machines

# Request stream: 100 requests per second
arrival_rate = 100.0

# Request processing time: mean 50 ms, CV = 1.2
service_mean = 0.05  # seconds
service_cv = 1.2

# Create H2 distribution parameters to model
# a high coefficient of variation
h2_params = H2Distribution.get_params_by_mean_and_cv(
    mean=service_mean,
    cv=service_cv
)

# Simulation
qs = QsSim(num_of_channels=num_vms)
qs.set_sources(arrival_rate, "M")
qs.set_servers(h2_params, "H")

results = qs.run(100000)

# Analyze the results
print("Cloud server analysis:")
print(f"Number of virtual machines: {num_vms}")
print(f"Request rate: {arrival_rate} req/s")
print(f"\nResults:")
print(f"  Mean waiting time: {results.w[0] * 1000:.2f} ms")
print(f"  Mean processing time: {results.v[0] * 1000:.2f} ms")
print(f"  Utilization: {results.utilization:.2%}")

# Analyze the state probabilities
p = results.p
print(f"\nState probabilities:")
print(f"  Server idle: {p[0]:.2%}")
print(f"  1-4 requests in processing: {sum(p[1:5]):.2%}")
print(f"  5-8 requests in processing: {sum(p[5:9]):.2%}")
print(f"  Queue (9+ requests): {sum(p[9:]):.2%}")

# Recommendations
if results.utilization > 0.8:
    print("\n⚠️  Warning: high load! Consider increasing the number of VMs.")
elif results.w[0] > 0.1:
    print("\n⚠️  Warning: long waiting time! Optimization is recommended.")
```

## Example 3: Optimizing a Transportation System

### Problem

Optimize the operation of a vehicle service station with several service bays and different types of work.

### Solution

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_waiting_moments

# Station parameters
arrival_rate = 10.0 / 60  # 10 cars per hour = cars per minute

# Service time: mean 20 minutes, CV = 0.6
service_mean = 20.0
service_cv = 0.6

# Test different numbers of service bays
num_posts_options = [2, 3, 4, 5]

print("Analysis of different station configurations:")
print(f"Arrival rate: {arrival_rate * 60:.1f} cars/hour")
print(f"Mean service time: {service_mean} minutes\n")

best_config = None
best_waiting_time = float('inf')

for num_posts in num_posts_options:
    # Compute the service rate for the target utilization
    target_utilization = 0.75
    service_rate = arrival_rate / (num_posts * target_utilization)
    
    # Simulation
    qs = QsSim(num_of_channels=num_posts)
    qs.set_sources(arrival_rate, "M")
    
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=1.0 / service_rate,
        cv=service_cv
    )
    qs.set_servers(gamma_params, "Gamma")
    
    results = qs.run(50000)
    
    # Analysis
    waiting_time_min = results.w[0]
    utilization = results.utilization
    
    print(f"{num_posts} bay(s):")
    print(f"  Mean waiting time: {waiting_time_min:.2f} minutes")
    print(f"  Utilization: {utilization:.2%}")
    
    if waiting_time_min < best_waiting_time:
        best_waiting_time = waiting_time_min
        best_config = num_posts
    
    print()

print(f"Recommended configuration: {best_config} bay(s)")
print(f"Expected waiting time: {best_waiting_time:.2f} minutes")
```

## Example 4: Comparing Simulation and Calculation

### Problem

Verify the correctness of a simulation by comparing its results against analytical calculations.

### Solution

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution
from most_queue.io.tables import (
    print_waiting_moments,
    print_sojourn_moments,
    probs_print
)

# M/G/1 system parameters
arrival_rate = 0.4
service_mean = 2.0
service_cv = 0.8

# Create H2 distribution parameters
h2_params = H2Distribution.get_params_by_mean_and_cv(
    mean=service_mean,
    cv=service_cv
)

# Compute the moments for the calculation
b = H2Distribution.calc_theory_moments(h2_params, 5)

print("Comparison of simulation and calculation for M/G/1:")
print(f"Arrival rate: {arrival_rate}")
print(f"Mean service time: {service_mean}")
print(f"Coefficient of variation: {service_cv}\n")

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
print("Waiting time moments:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nSojourn time moments:")
print_sojourn_moments(sim_results.v, calc_results.v)

print("\nState probabilities:")
probs_print(sim_results.p, calc_results.p, size=10)

# Check the accuracy
w_error = abs(sim_results.w[0] - calc_results.w[0]) / calc_results.w[0] * 100
v_error = abs(sim_results.v[0] - calc_results.v[0]) / calc_results.v[0] * 100

print(f"\nRelative error:")
print(f"  Waiting time: {w_error:.2f}%")
print(f"  Sojourn time: {v_error:.2f}%")
```

## Example 5: Performance Analysis with Different Distributions

### Problem

Study the influence of the service time coefficient of variation on the system characteristics.

### Solution

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import (
    H2Distribution,
    GammaDistribution,
    ErlangDistribution
)

arrival_rate = 0.5
service_mean = 2.0
num_channels = 2

# Different coefficients of variation
cvs = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]

print("Influence of the coefficient of variation on system characteristics:")
print(f"Arrival rate: {arrival_rate}")
print(f"Mean service time: {service_mean}")
print(f"Number of channels: {num_channels}\n")

results_table = []

for cv in cvs:
    # Choose the distribution depending on CV
    if cv < 1.0:
        # Use Erlang for CV < 1
        params = ErlangDistribution.get_params_by_mean_and_cv(
            mean=service_mean,
            cv=cv
        )
        dist_type = "E"
    elif cv == 1.0:
        # Exponential for CV = 1
        params = 1.0 / service_mean
        dist_type = "M"
    else:
        # Use H2 for CV > 1
        params = H2Distribution.get_params_by_mean_and_cv(
            mean=service_mean,
            cv=cv
        )
        dist_type = "H"
    
    # Simulation
    qs = QsSim(num_of_channels=num_channels)
    qs.set_sources(arrival_rate, "M")
    qs.set_servers(params, dist_type)
    
    results = qs.run(50000)
    
    results_table.append({
        'CV': cv,
        'waiting': results.w[0],
        'sojourn': results.v[0],
        'utilization': results.utilization
    })

# Print the results
print(f"{'CV':<8} {'Waiting':<12} {'Sojourn':<12} {'Load':<10}")
print("-" * 45)
for r in results_table:
    print(f"{r['CV']:<8.2f} {r['waiting']:<12.4f} {r['sojourn']:<12.4f} {r['utilization']:<10.2%}")

# Conclusions
print("\nConclusions:")
print("- As CV increases, the waiting time increases")
print("- The system becomes less predictable")
print("- Minimizing service time variability is recommended")
```

## Example 6: Visualizing Results

### Problem

Visualize the system state probabilities for different configurations.

### Solution

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
import matplotlib.pyplot as plt

# Parameters
arrival_rate = 2.0
service_rate = 1.0
num_channels_options = [1, 2, 3, 4]

# Compute the probabilities for different configurations
probabilities = {}

for n in num_channels_options:
    calc = MMnrCalc(n=n)
    calc.set_sources(l=arrival_rate)
    calc.set_servers(mu=service_rate)
    results = calc.run()
    probabilities[n] = results.p[:15]  # first 15 states

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

for n, probs in probabilities.items():
    states = list(range(len(probs)))
    ax.plot(states, probs, marker='o', label=f'M/M/{n}')

ax.set_xlabel('Number of jobs in the system')
ax.set_ylabel('Probability')
ax.set_title('State probability distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('state_probabilities.png', dpi=150)
print("Plot saved to state_probabilities.png")
```

## Tips for Using the Examples

1. **Adapt the parameters** — change the values to fit your problem
2. **Check stability** — make sure ρ < 1
3. **Use enough jobs** — accuracy requires 50000+ jobs
4. **Compare results** — use the calculation to verify the simulation
5. **Analyze the probabilities** — they give a complete picture of system behavior

---

**See also:**
- [Getting Started](getting_started.md) — usage basics
- [Queueing System Simulation](simulation.md) — simulation details
- [Numerical Methods](calculation.md) — analytical calculations
