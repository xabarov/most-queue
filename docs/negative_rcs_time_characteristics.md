# Time Characteristics of Queueing Systems with RCS Negative Customers (Remove Customer in Service)

[🇷🇺 Русская версия](negative_rcs_time_characteristics.ru.md)

This document describes the approach to computing the time characteristics of systems with negative customers of the **RCS** type — such a negative event removes **one job currently in service** from the system (the simulator picks a random busy server).

The document is aligned with the semantics of the `QsSimNegatives(..., NegativeServiceType.RCS)` simulation.

## 1. Model and the Key Difference from DISASTER

Let:

- \(\lambda\) — the rate of positive arrivals;
- \(B\) — the service time (in theory often approximated by \(H_2\) or \(C_2\); in simulation, Gamma is frequently used);
- \(\delta\) — the rate of negative events (RCS), `l_neg`.

**Difference from DISASTER:**

- with DISASTER the entire system is cleared, i.e. a negative event can terminate a customer both in the queue and in service;
- with RCS a negative event removes **only** the customer in service; the queue is not affected.

This leads to an important conclusion:

- the waiting time in the queue \(W\) is **not truncated** by a minimum with \(\mathrm{Exp}(\delta)\), yet it **depends** on \(\delta\), because RCS can free servers earlier (the customer in service is removed → the server becomes free).

## 2. Simulation Semantics (RCS)

In `most_queue/sim/negative.py`, for `NegativeServiceType.RCS`:

- if the system is empty, a negative event does nothing;
- otherwise, a random busy server is selected and the job in service is removed;
- for the removed job, the sojourn time \(V\) is recorded as \(t - t_{\text{arrival}}\);
- the waiting time \(W\) of the removed job is whatever it had accumulated before its service started (if it managed to start service).

## 3. The Effective "Removal Hazard" in Service

When \(m\) servers are busy, each negative event picks one of the \(m\) busy servers uniformly at random. Therefore, for a **specific** job in service the removal rate is:
\[
r(m) = \frac{\delta}{m}.
\]

If we assume that \(m\) is approximately constant during the service, the time in service for a typical job becomes:
\[
S = \min(B,\; Y),\quad Y\sim \mathrm{Exp}(r(m)).
\]

### 3.1. LST for \(\min(B, \mathrm{Exp}(r))\)

For independent \(B\) and \(Y\sim \mathrm{Exp}(r)\):
\[
S^*(s) = \mathbb{E}[e^{-s\min(B,Y)}]
= \frac{r}{s+r} + \frac{s}{s+r}\,\beta(s+r),
\]
where \(\beta(s)=\mathbb{E}[e^{-sB}]\) is the LST of the service time.

## 4. The Waiting Time \(W\): LST via the Level Structure (Takahashi–Takami)

In the multi-channel Takahashi–Takami method, the waiting time until service start under FIFO can be obtained via the LST by summing over levels \(k\ge n\) and micro-states (PASTA).

For RCS, what changes is the "time until the next server release" at level \(k\ge n\):

- in micro-state \(j\), the total service completion rate is \(\text{service\_rate}(j)\);
- additionally, a server can be released due to RCS with rate \(\delta\) (one of the jobs in service is removed).

Therefore, the waiting-time LST uses an exponential factor of the form:
\[
\mathrm{Exp}(\text{service\_rate}(j) + \delta).
\]

The resulting structure:

- with probability \(\mathbb{P}(k<n)\) the wait is zero;
- the defective part \( \mathbb{E}[e^{-sW};\,k\ge n] \) is computed by matrix summation over the levels.

In Most-Queue this is implemented in `most_queue/theory/negative/mgn_rcs.py` as a function of the waiting-time LST \(W^*(s)\), after which the moments are obtained by numerical differentiation at zero.

## 5. Sojourn Time \(V\), served/broken

Under RCS a customer can leave in two ways:

- **served**: service completed before removal;
- **broken**: the customer was removed by a negative event during service.

For a fixed \(m\) (servers busy at the moment service starts), the conditional structure is simple:
\[
V = W + \min(B, \mathrm{Exp}(\delta/m)).
\]

The practical difficulty is that \(m\) depends on the state:

- if the customer arrives at level \(k<n\), it starts service immediately, and naturally \(m=k+1\);
- if the customer arrives at level \(k\ge n\), it waits and then, as a rough approximation, starts service with \(m\approx n\).

Most-Queue uses exactly this approximation (a mixture of the cases \(k<n\) and \(k\ge n\)), but the computation is carried out via the LST in order to:

- correctly account for the fact that \(W\) already "incorporates" the influence of \(\delta\);
- obtain \(V\), \(V_{\text{served}}\), \(V_{\text{broken}}\) without a manual convolution of moments with an incorrect expectation.

## 6. Related Implementations in the Code

- `most_queue/theory/negative/mgn_rcs.py` — M/H₂/n with negative RCS customers.
- `most_queue/sim/negative.py` — the simulation semantics of `NegativeServiceType.RCS`.
