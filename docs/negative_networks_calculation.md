# Calculation of Queueing Networks with Negative Customers

[🇷🇺 Русская версия](negative_networks_calculation.ru.md)

## Introduction

This document describes a method for the analytical calculation of open queueing networks whose nodes are subject to negative customers. The method is based on flow decomposition (the flow decomposition method) and makes it possible to compute the raw moments of the sojourn time of jobs in the network.

## Problem Statement

We consider an open queueing network consisting of $m$ nodes. Each node $i$ ($i = 1, 2, \ldots, m$) is a multi-channel M/H2/$n_i$ queueing system with negative customers, where $n_i$ is the number of service channels in node $i$.

### Network Parameters

- **External flow of positive jobs**: a Poisson flow with rate $\lambda_0$
- **Routing matrix**: $R = \{r_{ij}\}$, where $r_{ij}$ is the probability that a job moves from node $i$ to node $j$ ($i, j = 0, 1, \ldots, m$; node $0$ is the external source)
- **Negative customers**: arrive independently of the positive jobs and do not follow the routing matrix

### Types of Negative Customers

Each node may use one of the following negative customer types:

1. **DISASTER** — when a negative customer arrives, all jobs (in service and in the queue) are removed from the node
2. **RCS (Remove Customer in Service)** — when a negative customer arrives, one job currently in service is removed

> **Note**: The current implementation supports only the DISASTER and RCS types. The RCE (Remove Customer at End) and RCH (Remove Customer at Head) types are not supported in the calculation.

### Negative Customer Arrival Modes

1. **Global mode** (`"global"`): all nodes of the network receive negative customers with the same rate $\lambda_{neg}$
2. **Per-node mode** (`"per_node"`): each node $i$ has its own negative customer rate $\lambda_{neg}^{(i)}$

## Calculation Method

The calculation method is based on flow decomposition and consists of the following stages:

1. Computing the effective arrival rates of positive jobs at each node via the balance equations
2. Computing the characteristics of each node in isolation, taking negative customers into account
3. Recombining the results for the whole network by the flow decomposition method

### Stage 1: Computing the Effective Arrival Rates of Positive Jobs

The flow balance equations hold for positive jobs. Let $\lambda_i$ denote the effective arrival rate of positive jobs at node $i$.

The balance equations have the form:

$$\lambda_i = \lambda_0 r_{0i} + \sum_{j=1}^{m} \lambda_j r_{ji}, \quad i = 1, 2, \ldots, m$$

In matrix form:

$$\boldsymbol{\lambda} = \boldsymbol{\lambda}_0 + \boldsymbol{\lambda} Q^T$$

where:
- $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \ldots, \lambda_m)^T$ — the vector of effective arrival rates
- $\boldsymbol{\lambda}_0 = (\lambda_0 r_{01}, \lambda_0 r_{02}, \ldots, \lambda_0 r_{0m})^T$ — the vector of external arrivals
- $Q = \{r_{ij}\}_{i,j=1}^{m}$ — the routing submatrix between the nodes

Hence:

$$(I - Q^T) \boldsymbol{\lambda} = \boldsymbol{\lambda}_0$$

$$\boldsymbol{\lambda} = (I - Q^T)^{-1} \boldsymbol{\lambda}_0$$

where $I$ is the identity matrix of size $m \times m$.

### Stage 2: Computing the Characteristics of Individual Nodes

For each node $i$, the moments of the sojourn time of a job in the node are computed with negative customers taken into account.

#### Node Parameters

- Arrival rate of positive jobs: $\lambda_i$ (computed at Stage 1)
- Arrival rate of negative customers:
  - in global mode: $\lambda_{neg}^{(i)} = \lambda_{neg}$
  - in per-node mode: $\lambda_{neg}^{(i)}$ is set individually
- Number of channels: $n_i$
- Service time moments: $b_i^{(k)}$, $k = 1, 2, 3, 4$ (raw moments of the service time distribution)

#### Calculation for a Node of the DISASTER Type

For a node with negative customers of the DISASTER type, the `MGnNegativeDisasterCalc` class is used, which implements the calculation method for the M/H2/$n$ system with "disaster"-type negative customers.

**Mathematical model:**

The system is described by a Markov process whose states account for:
- The number of jobs in the system
- The phases of the hyperexponential service time distribution (H2 distribution)
- Special "disaster" states

When a negative customer arrives, all jobs are removed from the system through artificial states with a high transition rate $\gamma \gg \mu$, where $\mu$ is the service rate.

**Sojourn time moments:**

The sojourn time moments $v_i^{(k)}$ ($k = 1, 2, 3, 4$) are computed via the Laplace-Stieltjes transform (LST) of the waiting time $W_i(s)$ and the service time with interruptions taken into account.

The sojourn time in a node is composed of the waiting time and the service time:

$$V_i = W_i + S_i$$

where $S_i$ is the service time accounting for possible interruptions by negative customers.

For a system with negative customers of the DISASTER type:

$$S_i = \min(H2_i, \exp(\lambda_{neg}^{(i)}))$$

where $H2_i$ is the random service time with the H2 distribution, and $\exp(\lambda_{neg}^{(i)})$ is an exponential distribution with parameter $\lambda_{neg}^{(i)}$.

The moments of $S_i$ are computed as:

$$S_i^{(k)} = \mathbb{E}[\min(H2_i, \exp(\lambda_{neg}^{(i)}))^k]$$

The sojourn time moments are obtained by convolving the waiting time and service time moments:

$$v_i^{(k)} = \sum_{j=0}^{k} \binom{k}{j} w_i^{(j)} S_i^{(k-j)}$$

where $w_i^{(j)}$ are the waiting time moments, computed numerically via the LST.

#### Calculation for a Node of the RCS Type

For a node with negative customers of the RCS type, the `MGnNegativeRCSCalc` class is used.

**Mathematical model:**

When a negative customer arrives, one job currently in service is removed. If all channels are busy, the rate at which a job is removed from service depends on the number of busy channels.

For a system with $j$ busy channels ($j = 1, 2, \ldots, n_i$), the effective removal rate of a job from service equals $\lambda_{neg}^{(i)} / j$.

**Sojourn time moments:**

The service time accounting for interruptions:

$$S_i = \min(H2_i, \exp(\lambda_{neg}^{(i)} / j))$$

conditioned on $j$ channels being busy at the moment service starts.

The conditional service time moments given $j$ busy channels:

$$S_i^{(k)}(j) = \mathbb{E}[\min(H2_i, \exp(\lambda_{neg}^{(i)} / j))^k]$$

Taking into account the probabilities of the states with $j$ busy channels:

$$S_i^{(k)} = \sum_{j=1}^{n_i} \pi_j^{(i)} S_i^{(k)}(j)$$

where $\pi_j^{(i)}$ is the stationary probability that $j$ channels are busy in node $i$.

The sojourn time moments are computed analogously to the DISASTER case, via the convolution of the waiting time and service time moments.

### Stage 3: Recombining the Results for the Whole Network

After the sojourn time moments in each node $v_i^{(k)}$ ($i = 1, 2, \ldots, m$, $k = 1, 2, 3, 4$) have been computed, the network sojourn time moments $v_{net}^{(k)}$ are derived.

#### Approximation of the Sojourn Time Distributions

The sojourn time distribution in each node is approximated by a gamma distribution fitted to the first two moments.

The gamma distribution parameters for node $i$:

$$\alpha_i = \frac{(v_i^{(1)})^2}{v_i^{(2)} - (v_i^{(1)})^2}$$

$$\mu_i = \frac{v_i^{(1)}}{v_i^{(2)} - (v_i^{(1)})^2}$$

where $\alpha_i$ is the shape parameter and $\mu_i$ is the scale parameter.

#### Laplace-Stieltjes Transform

The LST of the gamma distribution:

$$\mathcal{L}_i(s) = \left(\frac{\mu_i}{\mu_i + s}\right)^{\alpha_i}$$

#### Flow Decomposition

The flow decomposition method is used to compute the LST of the sojourn time in the network.

Denote:
- $P = (r_{01}, r_{02}, \ldots, r_{0m})$ — the vector of probabilities that a job from the external source enters each node
- $T = (r_{1,m+1}, r_{2,m+1}, \ldots, r_{m,m+1})^T$ — the vector of exit probabilities from the nodes
- $Q = \{r_{ij}\}_{i,j=1}^{m}$ — the transition matrix between the nodes
- $N(s) = \text{diag}(\mathcal{L}_1(s), \mathcal{L}_2(s), \ldots, \mathcal{L}_m(s))$ — the diagonal matrix of the node sojourn time LSTs

The LST of the sojourn time in the network:

$$G_{net}(s) = P (I - N(s) Q)^{-1} N(s) T$$

where $I$ is the identity matrix of size $m \times m$.

#### Computing the Moments

The network sojourn time moments are computed by numerical differentiation of the LST.

A small step $h$ is chosen (e.g., $h = 10^{-4}$) and the LST values are evaluated at the points:

$$s_k = h \cdot k, \quad k = 1, 2, 3, 4$$

$$g_k = G_{net}(s_k)$$

The moments are computed by the five-point numerical differentiation formula:

$$v_{net}^{(k)} = (-1)^k \frac{d^k G_{net}(s)}{ds^k}\Big|_{s=0}$$

For $k = 1, 2, 3$, the corresponding numerical differentiation formulas are used.

## Calculation Algorithm

A step-by-step algorithm for calculating a network with negative customers is given below.

### Input Data

1. External flow rate: $\lambda_0$
2. Routing matrix: $R$ of size $(m+1) \times (m+1)$
3. Negative customer mode: `"global"` or `"per_node"`
4. Negative customer rate(s): $\lambda_{neg}$ or $\{\lambda_{neg}^{(i)}\}_{i=1}^{m}$
5. For each node $i$:
   - Number of channels: $n_i$
   - Service time moments: $\{b_i^{(k)}\}_{k=1}^{4}$
   - Negative customer type: DISASTER or RCS

### Algorithm

**Step 1.** Solve the balance equations for the positive jobs:

$$\boldsymbol{\lambda} = (I - Q^T)^{-1} \boldsymbol{\lambda}_0$$

**Step 2.** For each node $i = 1, 2, \ldots, m$:

2.1. Determine the negative customer rate:
- If the mode is `"global"`: $\lambda_{neg}^{(i)} = \lambda_{neg}$
- If the mode is `"per_node"`: $\lambda_{neg}^{(i)}$ is given individually

2.2. Choose the calculation method according to the negative customer type:
- If DISASTER: create an `MGnNegativeDisasterCalc(n_i)` object
- If RCS: create an `MGnNegativeRCSCalc(n_i)` object

2.3. Set the parameters:
- `calc.set_sources(l_pos=λ_i, l_neg=λ_{neg}^{(i)})`
- `calc.set_servers(b={b_i^{(k)}})`

2.4. Run the calculation: `calc.run()`

2.5. Obtain the sojourn time moments: $v_i^{(k)}$ are extracted from the calculation result (the `calc.get_results().v` method)

**Step 3.** Approximate the sojourn time distributions by gamma distributions:

For each node $i$:
- Compute the gamma distribution parameters $(\alpha_i, \mu_i)$ from the moments $v_i^{(1)}$ and $v_i^{(2)}$

**Step 4.** Compute the LST of the sojourn time in the network:

For $k = 1, 2, 3, 4$:
- $s_k = h \cdot k$
- $N(s_k) = \text{diag}(\mathcal{L}_1(s_k), \ldots, \mathcal{L}_m(s_k))$
- $G(s_k) = (I - N(s_k) Q)^{-1}$
- $g_k = P G(s_k) N(s_k) T$

**Step 5.** Compute the network sojourn time moments:

$$v_{net}^{(k)} = (-1)^k \frac{d^k G_{net}(s)}{ds^k}\Big|_{s=0}$$

using numerical differentiation over the values $\{g_k\}_{k=1}^{4}$.

### Output Data

- Network sojourn time moments: $\{v_{net}^{(k)}\}_{k=1}^{4}$
- Effective arrival rates at the nodes: $\{\lambda_i\}_{i=1}^{m}$
- Node utilization coefficients: $\{\rho_i = \lambda_i b_i^{(1)} / n_i\}_{i=1}^{m}$

## Implementation Details

### Handling Zero Columns in the Routing Matrix

When solving the balance equations, zero columns of the routing matrix (nodes that receive no jobs) are handled specially. Such columns are excluded from the system of equations, and the corresponding rates are set to zero.

### Numerical Stability

To ensure numerical stability when computing the inverse matrix $(I - N(s) Q)^{-1}$, a singularity check is performed and special cases are handled.

### Gamma Distribution Approximation

Approximating the node sojourn time distribution by a gamma distribution fitted to the first two moments provides sufficient accuracy for most practical problems, provided that the coefficient of variation of the sojourn time is not too large.

## Limitations of the Method

1. The method applies only to open networks (there is an external flow and an exit from the network)
2. Only the DISASTER and RCS negative customer types are supported
3. The service time distribution is approximated by an H2 distribution
4. The method relies on the assumption of node independence (the decomposition approximation)

## Comparison with Simulation

To validate the calculation results, they are compared against the results of discrete-event simulation (the `NegativeNetwork` class). As a rule, the discrepancy between the calculated and simulated values of the first sojourn time moments does not exceed 5-10% at moderate node utilization coefficients ($\rho_i < 0.8$).

## References

1. Gelenbe, E. (1991). Product-form queueing networks with negative and positive customers. *Journal of Applied Probability*, 28(3), 656-663.

2. Gelenbe, E., & Schassberger, R. (1992). Stability of product form queueing networks with negative customers. *Journal of Applied Probability*, 29(4), 890-901.

3. Takahashi, Y., & Takami, Y. (1976). A numerical method for the steady-state probabilities of a multi-server queueing system. *Management Science*, 22(6), 656-663.

4. Kleinrock, L. (1979). *Queueing Theory* [Теория массового обслуживания]. Moscow: Mashinostroenie.

5. Bocharov, P. P., & Pechinkin, A. V. (2004). *Queueing Theory* [Теория массового обслуживания]. Moscow: RUDN.
