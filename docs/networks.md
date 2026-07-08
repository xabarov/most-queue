# Queueing Networks

[🇷🇺 Русская версия](networks.ru.md)

Queueing networks are systems consisting of several nodes (individual queueing systems) between which jobs can move. The Most-Queue library supports simulation and calculation of open queueing networks.

## Introduction to Queueing Networks

### Basic Concepts

- **Network node** — an individual queueing system within the network
- **Transition matrix** — routing rules for jobs between nodes
- **External flow** — jobs arriving at the network from outside
- **Output flow** — jobs leaving the network

### Network Types

- **Open network** — jobs can arrive from the external environment and leave the network
- **Closed network** — a fixed number of jobs circulates within the network

## Simulation of Queueing Networks

### The NetworkSimulator Class

The `NetworkSimulator` class is used for discrete-event simulation of queueing networks.

### Creating a Network

```python
from most_queue.sim.networks.network import NetworkSimulator

# Create a network simulator
network = NetworkSimulator()
```

### Configuring the Transition Matrix

The transition matrix defines how jobs move between the network nodes.

```python
import numpy as np

# Transition matrix R
# R[i, j] - probability of transition from node i to node j
# R[i, 0] - probability of leaving the network from node i
# First row (index 0) - entry into the network
# Last column (index n+1) - exit from the network

R = np.matrix([
    [1, 0, 0, 0, 0, 0],      # entry: all jobs go to node 1
    [0, 0.4, 0.6, 0, 0, 0],  # node 1: 40% to node 2, 60% to node 3
    [0, 0, 0.2, 0.4, 0.4, 0], # node 2: 20% stay, 40% to node 4, 40% to node 5
    [0, 0, 0, 0, 1, 0],      # node 3: all to node 5
    [0, 0, 0, 0, 1, 0],      # node 4: all to node 5
    [0, 0, 0, 0, 0, 1],      # node 5: all leave the network
])

network.set_sources(arrival_rate=1.0, R=R)
```

### Configuring the Network Nodes

Each node is configured separately, specifying the number of channels and the service parameters.

```python
from most_queue.random.distributions import H2Distribution

# Service parameters for each node
serv_params = []
num_channels = [3, 2, 3, 4, 3]  # number of channels in each node

for i in range(5):  # 5 nodes
    # Create H2-distribution parameters for node i
    h2_params = H2Distribution.get_params_by_mean_and_cv(
        mean=2.0,
        cv=0.8
    )
    serv_params.append({
        "type": "H",
        "params": h2_params
    })

# Configure the nodes
network.set_nodes(serv_params=serv_params, n=num_channels)
```

### Running the Simulation

```python
# Run the simulation for 50000 jobs
results = network.run(50000)

# Results
print(f"Mean sojourn time in the network: {results.v[0]:.4f}")
print(f"Node arrival rates: {results.intensities}")
print(f"Node utilizations: {results.loads}")
```

### Complete Simulation Example

```python
import numpy as np
from most_queue.sim.networks.network import NetworkSimulator
from most_queue.random.distributions import H2Distribution

# Create the network
network = NetworkSimulator()

# Transition matrix for a network with 3 nodes
R = np.matrix([
    [1, 0, 0, 0],      # entry -> node 1
    [0, 0.5, 0.5, 0], # node 1 -> node 2 (50%) or node 3 (50%)
    [0, 0, 0, 1],     # node 2 -> exit
    [0, 0, 0, 1],     # node 3 -> exit
])

network.set_sources(arrival_rate=2.0, R=R)

# Configure the nodes
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

network.set_nodes(serv_params=serv_params, n=num_channels)

# Simulation
results = network.run(50000)

print(f"Sojourn time: {results.v[0]:.4f}")
print(f"Arrival rates: {results.intensities}")
print(f"Utilizations: {results.loads}")
```

## Calculation of Queueing Networks

### The OpenNetworkCalc Class

The `OpenNetworkCalc` class is used for analytical calculation of open queueing networks by the decomposition method.

### Calculation Example

```python
import numpy as np
from most_queue.theory.networks.open_network import OpenNetworkCalc
from most_queue.random.distributions import H2Distribution

# Create the calculator
net_calc = OpenNetworkCalc()

# Transition matrix
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Configure the external flow
net_calc.set_sources(R=R, arrival_rate=2.0)

# Compute service moments for each node
b = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    b.append(H2Distribution.calc_theory_moments(h2_params, 4))

# Configure the nodes
net_calc.set_nodes(b=b, n=num_channels)

# Calculation
results = net_calc.run()

print(f"Node arrival rates: {results.intensities}")
print(f"Node utilizations: {results.loads}")
print(f"Mean sojourn time: {results.v[0]:.4f}")
```

## Networks with Priorities

### The PriorityNetworkSimulator Class

The `PriorityNetworkSimulator` class is used to simulate networks with priority disciplines in the nodes.

### Example of a Network with Priorities

```python
import numpy as np
from most_queue.sim.networks.priority_network import PriorityNetworkSimulator
from most_queue.random.distributions import GammaDistribution

# Create a network with priorities
network = PriorityNetworkSimulator()

# Transition matrix
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.6, 0.4, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Configure the flows for each priority class
arrival_rates = [1.0, 0.5]  # arrival rates for two classes
network.set_sources(arrival_rates=arrival_rates, R=R)

# Configure the nodes with priorities
serv_params = []
num_channels = [2, 3]
num_classes = 2

for i in range(2):
    # Service parameters for each class in the node
    node_params = []
    for j in range(num_classes):
        gamma_params = GammaDistribution.get_params_by_mean_and_cv(
            mean=1.5 + j * 0.5,  # different means for different classes
            cv=0.7
        )
        node_params.append({
            "type": "Gamma",
            "params": gamma_params
        })
    serv_params.append(node_params)

network.set_nodes(
    serv_params=serv_params,
    n=num_channels,
    priority="PR"  # or "NP"
)

# Simulation
results = network.run(50000)

# Results for each class
print(f"Sojourn time, class 1: {results.v[0][0]:.4f}")
print(f"Sojourn time, class 2: {results.v[1][0]:.4f}")
```

### Calculation of Networks with Priorities

```python
from most_queue.theory.networks.open_network_prty import OpenNetworkPrtyCalc

calc = OpenNetworkPrtyCalc()
calc.set_sources(arrival_rates=[1.0, 0.5], R=R)

# Service moments for each class in each node
b = []  # b[i][j] - moments for node i, class j
calc.set_nodes(b=b, n=num_channels, priority="PR")

results = calc.run()
```

## Networks with Negative Customers

### Calculation of Networks with Negative Customers

For analytical calculation of networks with negative customers, use the `NegativeNetworkCalc` class, which implements the flow decomposition method. A detailed description of the mathematical calculation method with formulas is given in the document [Calculation of Networks with Negative Customers](negative_networks_calculation.md).

### The NegativeNetworkCalc Class

The `NegativeNetworkCalc` class is used for analytical calculation of open queueing networks with negative customers by the decomposition method.

### Calculation Example

```python
import numpy as np
from most_queue.theory.networks.negative_network import NegativeNetworkCalc
from most_queue.sim.negative import NegativeServiceType
from most_queue.random.distributions import H2Distribution

# Create a calculator with global negative customers
net_calc = NegativeNetworkCalc(negative_arrival_type="global")

# Transition matrix
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Configure the sources
net_calc.set_sources(
    arrival_rate=2.0,
    R=R,
    negative_arrival_rate=0.1  # global arrival rate of negative customers
)

# Compute service moments for each node
b = []
num_channels = [2, 3, 2]
negative_types = [
    NegativeServiceType.DISASTER,  # node 1: DISASTER type
    NegativeServiceType.RCS,       # node 2: RCS type
    NegativeServiceType.DISASTER,  # node 3: DISASTER type
]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    b.append(H2Distribution.calc_theory_moments(h2_params, 4))

# Configure the nodes
net_calc.set_nodes(b=b, n=num_channels, negative_types=negative_types)

# Calculation
results = net_calc.run()

print(f"Node arrival rates: {results.intensities}")
print(f"Node utilizations: {results.loads}")
print(f"Mean sojourn time: {results.v[0]:.4f}")
```

### Example with Per-Node Negative Customers

```python
# Create a calculator with per-node negative customers
net_calc = NegativeNetworkCalc(negative_arrival_type="per_node")

# Configure the sources with individual arrival rates
net_calc.set_sources(
    arrival_rate=2.0,
    R=R,
    negative_arrival_rates=[0.1, 0.05, 0.15]  # individual rates for each node
)

# The remaining setup is the same as in the previous example
```

### The NegativeNetwork Class

The `NegativeNetwork` class is used to simulate networks with negative customers (negative jobs) in each node. Negative customers can interrupt the service of ordinary (positive) jobs, depending on the negative service type.

### Types of Negative Customers

Negative customers can affect the system in the following ways:

- **DISASTER** — removes all jobs from the node (both in service and in the queue)
- **RCS** (Remove Customer in Service) — removes one job from service
- **RCH** (Remove Customer at Head) — removes the job at the head of the queue
- **RCE** (Remove Customer at End) — removes the job at the end of the queue

### Negative Customer Arrival Modes

`NegativeNetwork` supports two arrival modes for negative customers:

1. **"global"** — negative customers arrive globally and affect all nodes simultaneously
2. **"per_node"** — each node has its own flow of negative customers

### Creating a Network with Negative Customers

```python
from most_queue.sim.networks.negative_network import NegativeNetwork
from most_queue.sim.negative import NegativeServiceType
import numpy as np

# Create a network with global negative customers
network = NegativeNetwork(negative_arrival_type="global")

# Or with individual negative customers for each node
network = NegativeNetwork(negative_arrival_type="per_node")
```

### Configuring the Sources

#### Global Negative Customers

```python
# Transition matrix
R = np.matrix([
    [1, 0, 0, 0],      # entry -> node 1
    [0, 0.5, 0.5, 0], # node 1 -> node 2 (50%) or node 3 (50%)
    [0, 0, 0, 1],     # node 2 -> exit
    [0, 0, 0, 1],     # node 3 -> exit
])

# Configure the sources with global negative customers
network.set_sources(
    positive_arrival_rate=2.0,      # arrival rate of positive jobs
    R=R,
    negative_arrival_rate=0.1       # arrival rate of global negative customers
)
```

#### Individual Negative Customers for Each Node

```python
# IMPORTANT: set_nodes() must be called BEFORE set_sources() for the per_node type
network.set_nodes(...)  # see below

# Configure the sources with individual negative customers
network.set_sources(
    positive_arrival_rate=2.0,
    R=R,
    negative_arrival_rates=[0.1, 0.05, 0.15]  # rates for each node
)
```

### Configuring the Nodes

```python
from most_queue.random.distributions import H2Distribution

# Service parameters for each node
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

# Negative customer types for each node
negative_types = [
    NegativeServiceType.DISASTER,  # node 1: removes all jobs
    NegativeServiceType.RCS,       # node 2: removes the job in service
    NegativeServiceType.RCH,        # node 3: removes the job at the head of the queue
]

# Configure the nodes
network.set_nodes(
    serv_params=serv_params,
    n=num_channels,
    negative_types=negative_types,  # negative customer types
    buffers=[None, 50, None]        # buffer sizes (optional)
)
```

### Complete Example

```python
import numpy as np
from most_queue.sim.networks.negative_network import NegativeNetwork
from most_queue.sim.negative import NegativeServiceType
from most_queue.random.distributions import H2Distribution

# Create a network with global negative customers
network = NegativeNetwork(negative_arrival_type="global")

# Transition matrix
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Configure the sources
network.set_sources(
    positive_arrival_rate=2.0,
    R=R,
    negative_arrival_rate=0.1  # global arrival rate of negative customers
)

# Configure the nodes
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

# All nodes use DISASTER by default
network.set_nodes(serv_params=serv_params, n=num_channels)

# Simulation
results = network.run(50000)

print(f"Sojourn time: {results.v[0]:.4f}")
print(f"Jobs served: {results.served}")
print(f"Jobs arrived: {results.arrived}")
```

### Example with Individual Negative Customers

```python
# Create a network with individual negative customers
network = NegativeNetwork(negative_arrival_type="per_node")

# Configure the nodes first (required before set_sources for per_node)
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

negative_types = [
    NegativeServiceType.DISASTER,
    NegativeServiceType.RCS,
    NegativeServiceType.RCH,
]

network.set_nodes(
    serv_params=serv_params,
    n=num_channels,
    negative_types=negative_types
)

# Then configure the sources
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

network.set_sources(
    positive_arrival_rate=2.0,
    R=R,
    negative_arrival_rates=[0.1, 0.05, 0.15]  # individual rates
)

# Simulation
results = network.run(50000)
```

### Important Notes

1. **Call order for the per_node type**: When using `negative_arrival_type="per_node"`, you must call `set_nodes()` first and then `set_sources()`, because configuring the sources requires knowing the number of nodes.

2. **Disabling negative customers**: To disable negative customers, pass `negative_arrival_rate=None` (for global) or `negative_arrival_rates=None` (for per_node).

3. **Default negative customer types**: If `negative_types` is not specified in `set_nodes()`, all nodes will use `NegativeServiceType.DISASTER` by default.

4. **Results**: `NegativeNetwork` returns a standard `NetworkResults` object with information about the sojourn time and the numbers of served and arrived jobs.

## Network Optimization

The library also provides methods for optimizing the network transition matrix to minimize the sojourn time of jobs.

### Optimization Example

```python
from most_queue.theory.networks.opt.transition import TransitionOptimization

# Create the optimizer
optimizer = TransitionOptimization()

# Initial transition matrix
R0 = np.matrix([...])

# Optimization
R_opt = optimizer.optimize(
    R0=R0,
    arrival_rate=2.0,
    b=b,
    n=num_channels
)

print(f"Optimized matrix:\n{R_opt}")
```

## Result Structures

### NetworkResults

```python
@dataclass
class NetworkResults:
    v: list[float] | None              # moments of the sojourn time in the network
    intensities: list[float] | None    # effective arrival rates at the nodes
    loads: list[float] | None          # node utilizations
    duration: float = 0.0              # calculation/simulation time
    arrived: int = 0                    # number of arrived jobs (simulation)
    served: int = 0                     # number of served jobs (simulation)
```

### NetworkResultsPriority

For networks with priorities:

```python
@dataclass
class NetworkResultsPriority:
    v: list[list[float]] | None        # moments for each class
    intensities: list[list[float]] | None  # arrival rates for each class
    loads: list[float] | None          # node utilizations
    duration: float = 0.0
    arrived: int = 0
    served: int = 0
```

## Building the Transition Matrix

### Construction Rules

1. **First row (index 0)** — entry into the network
   - `R[0, i]` — probability that a job enters node `i-1`
   - The sum must equal 1

2. **Rows 1..n** — network nodes
   - `R[i, j]` — probability of transition from node `i-1` to node `j-1`
   - `R[i, 0]` — probability of leaving the network from node `i-1`
   - `R[i, n+1]` — exit from the network (usually 1 in the last column)

3. **Last column** — exit from the network
   - Usually all elements equal 1

### Construction Example

```python
import numpy as np

# Network with 3 nodes
num_nodes = 3

R = np.matrix([
    # Entry -> nodes
    [1, 0, 0, 0],           # all jobs go to node 1
    
    # Node 1
    [0, 0.3, 0.5, 0.2],     # 30% stay, 50% to node 2, 20% leave
    
    # Node 2
    [0, 0, 0.4, 0.6],       # 40% stay, 60% leave
    
    # Node 3
    [0, 0, 0, 1],           # all leave
])
```

## Tips for Working with Networks

1. **Check the transition matrix** — make sure the probabilities in each row sum to 1
2. **Check stability** — every node must be stable (ρ < 1)
3. **Use simulation for verification** — compare calculation and simulation results
4. **Analyze node utilizations** — find the bottlenecks in the network
5. **Optimize routing** — use the optimization methods to improve performance characteristics

## Usage Examples

Detailed examples can be found in the tests:
- `test_network_no_prty.py` — network without priorities
- `test_network_im_prty.py` — network with priorities
- `test_network_opt.py` — network optimization
- `test_negative_network.py` — network with negative customers

---

**See also:**
- [Queueing System Simulation](simulation.md) — simulation basics
- [Priority Systems](priorities.md) — working with priorities
- [Usage Examples](examples.md) — practical examples
