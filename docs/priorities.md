# Priority Systems

[🇷🇺 Русская версия](priorities.ru.md)

Priority systems make it possible to split jobs into classes with different service priorities. The Most-Queue library supports both preemptive (PR) and non-preemptive (NP) priority.

## Priority Types

### Preemptive Priority (PR - Preemptive Resume)

With preemptive priority, the service of a low-priority job can be interrupted when a higher-priority job arrives. After the interruption, the service of the low-priority job resumes from the point where it was interrupted (resume).

**Characteristics:**
- High-priority jobs are served immediately
- Low-priority jobs can be preempted
- After preemption, service is resumed

### Non-Preemptive Priority (NP - Non-Preemptive)

With non-preemptive priority, service that has already started is never interrupted. Priorities are taken into account only when selecting the next job from the queue after the current service is completed.

**Characteristics:**
- Service that has started is completed in full
- Priorities affect only the order of selection from the queue
- A fairer discipline for low-priority jobs

## Simulation of Priority Systems

### The PriorityQueueSimulator Class

The `PriorityQueueSimulator` class is used to simulate multi-channel systems with priorities.

### Creating a Simulator

```python
from most_queue.sim.priority import PriorityQueueSimulator

# Create a simulator
# num_of_channels - number of channels
# num_of_classes - number of priority classes
# prty_type - priority type: "PR" or "NP"
qs = PriorityQueueSimulator(
    num_of_channels=5,
    num_of_classes=3,
    prty_type="PR"  # or "NP"
)
```

### Configuring the Arrival Flows

A separate arrival flow is configured for each priority class:

```python
# List of dictionaries with flow parameters for each class
sources = []

for j in range(num_of_classes):
    sources.append({
        "type": "M",                    # distribution type
        "params": arrival_rates[j]      # parameters (for M - the arrival rate)
    })

qs.set_sources(sources)
```

### Configuring the Service

Service time parameters are configured for each class:

```python
from most_queue.random.distributions import GammaDistribution

servers_params = []

for j in range(num_of_classes):
    # Service time distribution parameters for class j
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=service_means[j],
        cv=service_cv
    )
    servers_params.append({
        "type": "Gamma",
        "params": gamma_params
    })

qs.set_servers(servers_params)
```

### Complete Simulation Example

```python
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_sojourn_multiclass

# System parameters
num_channels = 5
num_classes = 3
arrival_rates = [0.1, 0.2, 0.3]
service_means = [2.25, 4.5, 6.75]
service_cv = 0.8

# Create a simulator with preemptive priority
qs = PriorityQueueSimulator(num_channels, num_classes, "PR")

# Configure the flows
sources = []
servers_params = []

for j in range(num_classes):
    # Arrival flow
    sources.append({"type": "M", "params": arrival_rates[j]})
    
    # Service parameters
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=service_means[j],
        cv=service_cv
    )
    servers_params.append({"type": "Gamma", "params": gamma_params})

qs.set_sources(sources)
qs.set_servers(servers_params)

# Run the simulation
qs.run(50000)

# Get the results
v_sim = qs.v  # sojourn time moments for each class
# v_sim[i][j] - j-th moment for class i
```

## Calculation of Priority Systems

### M/G/1 with Preemptive Priority

The `MG1Preemptive` class for calculating a single-channel system:

```python
from most_queue.theory.priority.preemptive.mg1 import MG1Preemptive

calc = MG1Preemptive(num_of_classes=3)

# Arrival rates for each class
calc.set_sources([0.1, 0.2, 0.3])

# Service time moments for each class
# b[i][j] - j-th moment for class i
b = [
    [2.25, 5.06, 15.19],  # class 1 (highest priority)
    [4.5, 24.3, 145.8],   # class 2
    [6.75, 54.68, 410.1]  # class 3 (lowest priority)
]
calc.set_servers(b)

results = calc.run()

# Results for each class
print(f"Class 1: mean sojourn time = {results.v[0][0]:.4f}")
print(f"Class 2: mean sojourn time = {results.v[1][0]:.4f}")
print(f"Class 3: mean sojourn time = {results.v[2][0]:.4f}")
```

### M/G/1 with Non-Preemptive Priority

The `MG1NonPreemptive` class:

```python
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptive

calc = MG1NonPreemptive(num_of_classes=3)
calc.set_sources([0.1, 0.2, 0.3])
calc.set_servers(b)
results = calc.run()
```

### M/G/c with Priorities

The `MGnInvarApproximation` class for multi-channel systems:

```python
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation

# Preemptive priority
calc_pr = MGnInvarApproximation(n=5, priority="PR")
calc_pr.set_sources([0.1, 0.2, 0.3])
calc_pr.set_servers(b)
results_pr = calc_pr.get_v()

# Non-preemptive priority
calc_np = MGnInvarApproximation(n=5, priority="NP")
calc_np.set_sources([0.1, 0.2, 0.3])
calc_np.set_servers(b)
results_np = calc_np.get_v()
```

## Comparing PR and NP Priorities

### Comparison Example

```python
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_sojourn_multiclass

num_channels = 5
num_classes = 3
arrival_rates = [0.1, 0.2, 0.3]
service_means = [2.25, 4.5, 6.75]
service_cv = 0.8

# Prepare the parameters
gamma_params = []
for j in range(num_classes):
    gamma_params.append(
        GammaDistribution.get_params_by_mean_and_cv(
            mean=service_means[j],
            cv=service_cv
        )
    )

sources = [{"type": "M", "params": arrival_rates[j]} for j in range(num_classes)]
servers_params = [{"type": "Gamma", "params": gamma_params[j]} for j in range(num_classes)]

# Preemptive priority
qs_pr = PriorityQueueSimulator(num_channels, num_classes, "PR")
qs_pr.set_sources(sources)
qs_pr.set_servers(servers_params)
qs_pr.run(50000)

# Non-preemptive priority
qs_np = PriorityQueueSimulator(num_channels, num_classes, "NP")
qs_np.set_sources(sources)
qs_np.set_servers(servers_params)
qs_np.run(50000)

# Compare the results
print("Preemptive priority (PR):")
for i in range(num_classes):
    print(f"  Class {i+1}: {qs_pr.v[i][0]:.4f}")

print("\nNon-preemptive priority (NP):")
for i in range(num_classes):
    print(f"  Class {i+1}: {qs_np.v[i][0]:.4f}")
```

### Key Properties

**Preemptive priority (PR):**
- High-priority jobs are served faster
- Low-priority jobs may wait a very long time
- Suitable for mission-critical jobs

**Non-preemptive priority (NP):**
- A fairer distribution of waiting time
- Low-priority jobs do not "starve"
- Suitable for systems where fairness matters

## Result Structure

### Multiclass Results

For priority systems, the results are structured by class:

```python
# Sojourn time moments
v = results.v  # v[i][j] - j-th moment for class i

# Waiting time moments
w = results.w  # w[i][j] - j-th moment for class i

# State probabilities (usually for low-priority jobs)
p = results.p
```

### Example of Result Analysis

```python
results = calc.run()

print("Analysis of results by class:")
for i in range(num_classes):
    print(f"\nClass {i+1} (priority {i+1}):")
    print(f"  Mean waiting time: {results.w[i][0]:.4f}")
    print(f"  Mean sojourn time: {results.v[i][0]:.4f}")
    
    # Second moment for computing the variance
    if len(results.w[i]) > 1:
        variance = results.w[i][1] - results.w[i][0]**2
        print(f"  Waiting time variance: {variance:.4f}")
```

## Practical Recommendations

### Choosing the Priority Type

1. **Use PR** when:
   - Fast service of high-priority jobs is critical
   - Low-priority jobs can wait
   - Examples: real-time systems, emergency services

2. **Use NP** when:
   - Fairness of service matters
   - Low-priority jobs must not "starve"
   - Examples: fair resource sharing

### Configuring the Classes

1. **Determine the number of classes** — usually 2-5 classes are enough
2. **Set the arrival rates** — account for the actual distribution of jobs
3. **Choose the service distributions** — use data on real service times
4. **Check the load** — make sure the system is stable for all classes

### Analyzing the Results

1. **Compare waiting times** — verify that the priorities work as expected
2. **Check fairness** — with NP priority, low-priority jobs should not wait too long
3. **Optimize the parameters** — tune the arrival rates and distributions to meet your goals

## Usage Examples

Detailed examples can be found in the tests:
- `test_qs_sim_prty.py` — simulation of priority systems
- `test_mmn_prty_busy_approx.py` — calculation of M/M/c with priorities
- `test_m_ph_n_prty.py` — systems with phase-type distribution and priorities

---

**See also:**
- [Queueing System Simulation](simulation.md) — simulation basics
- [Numerical Methods](calculation.md) — analytical calculations
- [Queueing Networks](networks.md) — networks with priorities in the nodes
