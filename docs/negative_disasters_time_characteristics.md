# Time Characteristics of Queueing Systems with Disasters (negative customers that clear the system)

[🇷🇺 Русская версия](negative_disasters_time_characteristics.ru.md)

This document describes the approach used in Most-Queue to compute the time characteristics (waiting time and sojourn time) of systems with **negative customers of the DISASTER type**, which upon arrival **instantly clear the system** (remove all customers from the queue and from service).

## 1. Model and Terminology

We consider a single-class queueing system with:

- positive arrivals (usually a Poisson flow) with rate \(\lambda\);
- service with time \(B\) (in numerical queueing-theory calculations it is often approximated by an \(H_2\) or \(C_2\) distribution; for simulation, a Gamma distribution is typically used);
- negative arrivals (disasters) — a Poisson flow with rate \(\delta\) (`l_neg`) which, upon a disaster event, **removes all customers from the system**.

In the simulator `QsSimNegatives(..., NegativeServiceType.DISASTER)` the semantics are as follows:

- if a disaster occurs at time \(t\), then for every customer currently in the system, its sojourn time is recorded as \(t - t_{\text{arrival}}\);
- for customers in the queue, their accumulated waiting is likewise recorded as \(t - t_{\text{start\_wait}}\);
- the system becomes empty instantly.

### Which Random Variables We Want to Obtain

For a "typical" positive customer:

- \(W\) — the **waiting time in the queue** (until service starts **or** until a disaster, whichever comes first).
- \(V\) — the **sojourn time in the system** (until service completes **or** until a disaster, whichever comes first).

Conditional characteristics are also useful:

- \(V_{\text{served}}\) — \(V\) conditioned on the customer being **successfully served** (finished service before a disaster).
- \(V_{\text{broken}}\) — \(V\) conditioned on the customer being **removed by a disaster** (the disaster occurred before service completion).

## 2. The Key Idea: "min(·, Exp(δ))" and the LST

Let \(Y\sim \mathrm{Exp}(\delta)\) be the time until the **first disaster after the customer's arrival**.

By the independence of Poisson flows, \(Y\) is independent of all of the system's "internal" behavior, viewed as a function of the state at the arrival instant.

It is then convenient to define two quantities that refer to a **hypothetical** system in which disasters *after the customer's arrival are switched off*:

- \(W_0\) — the time until the customer's service starts, **assuming no disasters occur after its arrival** (but starting from the stationary state of the system with disasters at the arrival instant).
- \(Z_0 = W_0 + B\) — the time until the customer's service completes in the same "no-disaster-after-arrival" picture.

Then the quantities matching the simulation are expressed via the minimum:

- \(W = \min(W_0,\, Y)\)
- \(V = \min(Z_0,\, Y)\)

### 2.1. LST Formulas

Denote the Laplace–Stieltjes transform (LST):
\[
T^*(s) = \mathbb{E}\left[e^{-sT}\right],\quad s\ge 0.
\]

For \(Y\sim \mathrm{Exp}(\delta)\) and an independent \(T\), it holds that:
\[
\mathbb{E}\left[e^{-s\min(T,Y)}\right]
= \frac{\delta}{s+\delta} + \frac{s}{s+\delta}\, T^*(s+\delta).
\]

From this we immediately obtain:
\[
W^*(s) = \frac{\delta}{s+\delta} + \frac{s}{s+\delta}\, W_0^*(s+\delta),
\]
\[
V^*(s) = \frac{\delta}{s+\delta} + \frac{s}{s+\delta}\, Z_0^*(s+\delta).
\]

### 2.2. How to Obtain Moments from the LST

If \(T\) has moments up to order \(m\), then:
\[
\mathbb{E}[T^m] = (-1)^m \left.\frac{d^m}{ds^m}T^*(s)\right|_{s=0}.
\]

In the Most-Queue code, numerical differentiation via `scipy.misc.derivative(...)` is used for this (note that it is marked as deprecated in SciPy; if desired, it can be replaced with a more modern numerical differentiation routine).

## 3. The Single-Channel Case: `MG1Disasters`

For M/G/1 with disasters, a PK-like formula in terms of the LST is used (see Jain & Sigman, 1996).

Important: a disaster affects **both waiting and service** (via the minimum with \(Y\)), so a naive "moment convolution" of the form
\[
V \stackrel{\Large\text{x}}{=} W + \min(B,Y)
\]
is in general **incorrect** — because \(Y\) may occur during the waiting period, i.e. it "truncates" both \(W\) and \(B\).

The correct approach in the implementation `most_queue/theory/negative/mg1_disasters.py`:

- first, the waiting-time LST \(W^*(s)\) is built for the system with clearings;
- then the sojourn-time LST is expressed as:
\[
V^*(s)=\frac{\delta}{s+\delta}+\frac{s}{s+\delta}\,W^*(s+\delta)\,\beta(s+\delta),
\]
where \(\beta(s)=\mathbb{E}[e^{-sB}]\) is the LST of the service time.

## 4. The Multi-Channel Case: `MGnNegativeDisasterCalc` (Takahashi–Takami)

In the multi-channel case (M/H₂/n with a discrete level and micro-states) there is no direct "closed-form" PK-type formula, so the following construction is used:

1) **The stationary state distribution** for the system with disasters is computed by an extended Takahashi–Takami method (see `docs/negative_queues_takahasi_takami.md`). This yields weights `Y[k][0, i]`, interpreted as the probabilities that a typical arrival sees level \(k\) and micro-state \(i\) (PASTA).

2) To compute \(W_0^*(s)\) and \(Z_0^*(s)\), the "hypothetical" dynamics *without disasters after arrival* are used, but **conditioned on the state the customer saw in the system with disasters**:

- if the customer arrived when the number of customers was \(k<n\), the wait until service start is zero, \(W_0=0\);
- if \(k\ge n\), the wait until service start equals the sum of the service completion times needed for a server to become free. In the H₂ model this yields an LST as a product of exponential LSTs with rates depending on the micro-state.

Technically, this is implemented in the code as a summation over levels \(k\ge n\) with matrix powers of the "upward" transitions of the base system **without disasters** (`base_mgn.calc_up_probs(...)`), rather than through manual convolution of moments.

3) After obtaining \(W_0^*(s)\) and \(Z_0^*(s)\), the general formulas for the minimum with an exponential are applied:

- \(W^*(s)\) via \(W_0^*(s+\delta)\)
- \(V^*(s)\) via \(Z_0^*(s+\delta)\)

4) The conditional distributions for `served` and `broken` are likewise expressed via \(Z_0^*\):

- probability of successful service:
\[
p_{\text{served}} = \mathbb{P}(Z_0<Y)=\mathbb{E}[e^{-\delta Z_0}] = Z_0^*(\delta)
\]
- LST for served customers:
\[
\mathbb{E}[e^{-sV}\mid served] = \frac{Z_0^*(s+\delta)}{Z_0^*(\delta)}
\]
- LST for customers removed by a disaster:
\[
\mathbb{E}[e^{-sV}\mid broken] = \frac{\delta}{s+\delta}\,\frac{1-Z_0^*(s+\delta)}{1-Z_0^*(\delta)}
\]

## 5. What This Gives in Practice

- **Consistency with simulation**: the formulas \(W=\min(W_0,Y)\), \(V=\min(Z_0,Y)\) match exactly how the simulator terminates customers upon a disaster.
- **Correctness for arbitrary parameters**: there is no need to assume that a disaster affects only the service or only the queue.
- **Conditional served/broken metrics** are obtained from the same underlying quantity \(Z_0^*\) without additional heuristics.

## 6. Related Implementations in the Code

- `most_queue/theory/negative/mg1_disasters.py` — M/G/1 with disasters, LST approach + moments via derivatives.
- `most_queue/theory/negative/mgn_disaster.py` — M/H₂/n with disasters (Takahashi–Takami extension) + computation of \(W, V, V_{served}, V_{broken}\) via the LST.
- `most_queue/sim/negative.py` — the simulation semantics of disasters (`NegativeServiceType.DISASTER`).

## 7. References

1. Jain, G., Sigman, K. *A Pollaczek–Khintchine formula for M/G/1 queues with disasters.* Journal of Applied Probability 33(4), 1996.
2. Gelenbe, E. *Product-form queueing networks with negative and positive customers.* J. Appl. Prob., 1991.
