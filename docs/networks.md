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

## Exact Jackson Networks (Product Form)

For a Markovian open network (Poisson external arrivals, exponential M/M/n
nodes) the `JacksonNetworkCalc` class gives the **exact** product-form
solution (Jackson, 1957/1963). Mean values are exact, so it serves as a
baseline for the approximate decomposition of `OpenNetworkCalc`.

```python
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc

calc = JacksonNetworkCalc()
calc.set_sources(arrival_rate=1.0, R=R)      # same routing format as OpenNetworkCalc
calc.set_nodes(mu=[1.0, 2.0, 1.5], n=[2, 3, 2])
res = calc.run()
print(res.v[0], res.mean_jobs, res.loads)    # exact means
```

## QNA — Non-Poisson Internal Flows (Whitt)

The plain decomposition treats every internal flow as Poisson. The
`OpenNetworkCalcQNA` class implements Whitt's **Queueing Network Analyzer**
(1983): the squared coefficient of variation of interarrival times is
propagated through departure, splitting and superposition operations, and
each node is approximated as a GI/G/n queue with the Kraemer &
Langenbach-Belz correction. On networks with highly variable service the
error drops substantially (e.g. from ~20% to ~2% on an H2 tandem with
c² = 4 at utilization 0.8).

```python
from most_queue.theory.networks.qna import OpenNetworkCalcQNA

qna = OpenNetworkCalcQNA()
qna.set_sources(arrival_rate=1.0, R=R, arrival_cv2=1.0)
qna.set_nodes(b=b, n=num_channels)           # raw service moments per node
res = qna.run()
print(res.v[0], qna.arrival_cv2_nodes)       # mean sojourn + per-node arrival cv²
```

## Closed Networks (MVA and Buzen Convolution)

A closed network has no external arrivals: a fixed population of N jobs
circulates over the nodes (Gordon–Newell model). The `ClosedNetworkCalc`
class provides three solvers:

- `method="mva"` — exact Mean Value Analysis (Reiser–Lavenberg, 1980),
  including multi-server stations (via marginal probabilities) and
  infinite-server (delay) nodes — pass `n=[..., None, ...]` for a delay node;
- `method="convolution"` — Buzen's convolution algorithm (1973) for the
  normalization constant G(N); matches MVA to machine precision;
- `method="schweitzer"` — Schweitzer–Bard approximate MVA for large
  populations (multi-server stations via the Seidmann approximation).

```python
import numpy as np
from most_queue.theory.networks.closed_network import ClosedNetworkCalc

# Central-server model: CPU + 2 disks, 8 jobs
routing = np.array([
    [0.1, 0.5, 0.4],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
])

calc = ClosedNetworkCalc(method="mva")
calc.set_sources(R=routing, N=8)             # m x m matrix, rows sum to 1
calc.set_nodes(b=[0.02, 0.06, 0.08], n=[2, 1, 1])
res = calc.run()
print(res.throughput, res.mean_jobs, res.v[0])   # X, L_i, mean cycle time N/X
```

The paired simulator is `ClosedNetworkSim` (`most_queue.sim.networks.closed_network`),
with the same `set_sources` / `set_nodes` interface (Kendall-notation service
distributions) and a `seed` parameter.

## G-Networks (Negative Customers, Gelenbe Product Form)

For M/M/1 nodes the `GNetworkCalc` class gives the **exact** product-form
solution of a G-network (Gelenbe, 1991): a job completing service moves to
the next node as a positive customer or as a **negative signal** that removes
one customer from a non-empty node. External positive and negative Poisson
flows are set per node. The nonlinear traffic equations are solved by
fixed-point iteration. This complements the approximate
`NegativeNetworkCalc` decomposition (DISASTER/RCS, M/G/n nodes) with an
exact baseline for the Markovian single-channel case.

```python
import numpy as np
from most_queue.theory.networks.g_network import GNetworkCalc

calc = GNetworkCalc()
calc.set_sources(
    positive_rates=[0.5, 0.2],
    P_plus=np.array([[0.0, 0.4], [0.2, 0.0]]),    # movement as positive customers
    P_minus=np.array([[0.0, 0.2], [0.1, 0.0]]),   # movement as negative signals
    negative_rates=[0.1, 0.0],                     # external negative flows
)
calc.set_nodes(mu=[1.0, 1.5])
res = calc.run()
print(res.loads, res.mean_jobs, res.negative_intensities)
```

## BCMP Networks (Multi-Class Product Form)

The BCMP theorem (Baskett–Chandy–Muntz–Palacios, 1975) extends product form
to **multi-class** networks with four station types: FCFS (exponential,
class-independent rate), PS, LCFS-PR and IS (delay); PS/LCFS-PR/IS stations
are insensitive — only mean service times matter.

- `BCMPOpenNetworkCalc` — open network, per-class Poisson arrivals and
  per-class routing matrices; exact per-class means.
- `BCMPClosedNetworkCalc` — closed multi-chain network solved by **exact
  multi-chain MVA** (recursion over all population vectors).

```python
from most_queue.theory.networks.bcmp_network import BCMPClosedNetworkCalc

calc = BCMPClosedNetworkCalc()
calc.set_sources(R=[routing_class1, routing_class2], N=[3, 2])
calc.set_nodes(
    s=[[0.5, 0.8], [0.3, 0.4]],                  # s[node][class] mean service times
    station_types=["ps", "fcfs"],
)
res = calc.run()
print(res.throughput, res.mean_jobs)             # per class
```

## Tandems with Finite Buffers (Blocking After Service)

`TandemBlockingCalc` handles a production-line tandem where node i holds at
most K_i jobs: a job finishing service stays on the server (blocking it)
while the next node is full; external arrivals that find node 1 full are
lost. Two-pass decomposition (Brandwajn & Jow 1988; Dallery & Frein 1993):
throughput within ~1% of the exact CTMC on small lines. Paired simulator —
`TandemBlockingSim` (`most_queue.sim.networks.tandem_blocking`).

```python
from most_queue.theory.networks.blocking import TandemBlockingCalc

calc = TandemBlockingCalc()
calc.set_sources(arrival_rate=0.8)
calc.set_nodes(mu=[1.0, 1.2], capacity=[4, 3])   # None = unlimited node
res = calc.run()
print(calc.throughput, calc.loss_prob, calc.blocking_probs)
```

## Fork-Join Stations Inside a Network

`OpenNetworkCalcForkJoin` embeds fork-join stations into a routed open
network: a job forks into k parallel single-server branches and continues
when the last sub-task finishes (response approximations of Nelson–Tantawi /
Varma). Paired simulator — `ForkJoinNetworkSim`.

```python
from most_queue.theory.networks.fork_join_network import OpenNetworkCalcForkJoin

net = OpenNetworkCalcForkJoin()
net.set_sources(arrival_rate=0.5, R=R)
net.set_nodes([
    {"kind": "queue", "mu": 0.4, "n": 2},
    {"kind": "fork_join", "mu": 1.0, "k": 3},
])
res = net.run()
```

## MAP External Flow

`NetworkSimulator.set_sources(..., source_kendall="MAP", source_params=map_params)`
drives the network with a bursty MAP flow; on the analytic side feed QNA with
the MAP interarrival variability via `map_arrival_cv2(map_params)`. QNA is a
two-moment method: it captures the interarrival cv² but not the
autocorrelation, so for strongly correlated MAPs it lower-bounds congestion.

## Time-Varying Networks (PSA)

`TimeVaryingNetworkCalc` solves an open Markovian network with arrival rate
λ(t) by the pointwise stationary approximation — a stationary Jackson
snapshot at every grid instant (accurate for slow modulation). Paired
simulator — `TimeVaryingNetworkSim` (NHPP by thinning, phase-bucketed
statistics).

```python
from most_queue.theory.networks.time_varying_network import TimeVaryingNetworkCalc

calc = TimeVaryingNetworkCalc()
calc.set_sources(lam_fn=lambda t: 0.5 + 0.2 * math.sin(2 * math.pi * t / 2000), R=R)
calc.set_nodes(mu=[1.0, 1.4], n=[1, 1])
res = calc.run(t_grid=range(0, 2000, 100))   # res.v, res.mean_jobs_total per instant
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
