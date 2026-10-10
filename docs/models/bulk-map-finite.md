# Bulk service with correlated arrivals and a finite buffer: MAP/PH^(a,b)/1/N

[🇷🇺 Русская версия](bulk-map-finite.ru.md) · [← Model catalog](../models.md)

**In plain words:** a single server works in batches under the general `(a, b)` rule — it waits
until `a` customers have gathered, then takes up to `b` of them at once — arrivals are *bursty*
rather than Poisson, and the waiting room is finite, so an arrival that finds it full is turned
away. The quantity of interest is the whole distribution of how long an admitted customer waits,
not just its mean.

Two things separate this page from the rest of the [batch-service](batch.md) family. Arrivals come
from a **Markovian Arrival Process**, so they can be correlated in time — the realistic case for
request traffic, which clusters. And the buffer is genuinely **finite**, so blocking is part of the
model rather than a numerical truncation.

### The model

Arrivals follow a MAP `(D0, D1)`. The buffer holds at most `N` *waiting* customers; the batch in
service does not occupy it, and an arrival finding `N` waiting is lost. The server takes
`min(queue, b)` customers whenever at least `a` are present, and the batch's service time is
PH-distributed `(β, S)` — the same distribution whatever the batch size. Everyone in a batch
leaves together when it finishes, so

```
sojourn time = queueing time + one whole batch service
```

and the two are independent, because the batch a customer joins has not started while it waits.

### Exact distribution, without inverting anything

**Class:** `BulkServiceMapPhCalc` (`most_queue.theory.batch.map_ph_finite_buffer`)

```python
import numpy as np
from most_queue.random.map_ph import MAPParams, PHParams
from most_queue.theory.batch.map_ph_finite_buffer import BulkServiceMapPhCalc

calc = BulkServiceMapPhCalc(a=3, b=8, capacity=25)
calc.set_sources(MAPParams(D0=np.array([[-9.0, 0.6], [0.4, -1.4]]),
                           D1=np.array([[8.0, 0.4], [0.2, 0.8]])))
calc.set_servers(PHParams(alpha=np.array([0.3, 0.7]),
                          T=np.array([[-2.0, 1.0], [0.0, -5.0]])))

calc.get_w(2)                      # exact raw moments of the queueing time
calc.get_w_cdf([0.5, 1.0, 2.0])    # exact CDF, one point or many
calc.get_v_cdf(1.0)                # sojourn time
calc.get_loss_probability()
calc.run()                         # everything, plus state probabilities
```

For the classical Poisson/exponential case there are shorthands:
`set_poisson_sources(6.7)` and `set_exponential_servers(1.7)`.

**Method.** The model is that of Banik A.D., Chaudhry M.L., Barik S. & Singh G., *On the Heuristic
Computational Procedures of the Virtual Waiting-Time Distribution in a Non-renewal Input
Finite-Buffer Bulk-Service Queues: MAP/R^(a,b)/1/N*, J. Indian Soc. Probab. Stat. 26 (2025)
585–630, [doi:10.1007/s41096-025-00232-0](https://doi.org/10.1007/s41096-025-00232-0), and — for
the Poisson case — of its open-access predecessor Chaudhry M.L., Banik A.D., Barik S. & Goswami V.,
*A Novel Computational Procedure for the Waiting-Time Distribution (In the Queue) for Bulk-Service
Finite-Buffer Queues with Poisson Input*, Mathematics 11(5) (2023) 1142,
[doi:10.3390/math11051142](https://doi.org/10.3390/math11051142). **The model is theirs.**

What differs is the route. Both papers reach the waiting time through transforms: the queue-length
distribution gives a generating function, a functional relation converts it into the
Laplace–Stieltjes transform of the queueing time, and that transform is inverted numerically. The
LST itself is exact and elementary — equation (39) of the 2023 paper is a finite sum of powers of
`(1 − s/λ')`. The *inversion* is where the approximation lives: the authors fit a Padé rational
function to it (with Maple's `pade`) and expand in partial fractions. The 2025 MAP paper calls its
procedures heuristic in its own title for the same reason.

This implementation never forms a transform, so there is nothing to invert. It follows a **tagged
customer** through an absorbing Markov chain and reads the waiting time off directly as a
phase-type distribution: `1 − α exp(T t) 1` for the CDF, `k! α (−T)^−k 1` for the moments. Exact to
machine precision, with no Padé step and no inversion tolerance to tune.

**Why the chain stays small.** Follow a customer that joins at position `p`. At every service
completion the server takes exactly `b` from the front whenever more than `b` are waiting, so `p`
falls by `b` and nothing else matters — if `p > b` then the queue holds at least `p > b ≥ a`, so
another batch certainly starts. The queue length only ever matters at the single completion where
`p ≤ b`, to decide whether the server can start at once or must idle until `a` have gathered. And
*that* check needs the number of customers **behind** the tagged one only up to `a − 1`: any more
and the queue is certainly long enough. So the chain is indexed by position and by a count capped
at `a`, which makes it `O(N·a·m_s·m_a)` rather than `O(N²·m_s·m_a)`.

### Reproducing the published Poisson results

The 2023 paper's Example 6 is `M/M^(3,6)/1/200` with `λ = 6.7`, `μ = 1.7`. Its Table 7 gives the
queueing-time CDF:

| `t` | published | exact (here) |
|---|---|---|
| 2 | 0.859884 | 0.859882 |
| 4 | 0.977085 | 0.977084 |
| 6 | 0.996252 | 0.996252 |
| 8 | 0.999387 | 0.999387 |
| 10 | 0.999900 | 0.999900 |
| 12 | 0.999984 | 0.999984 |
| 14 | 0.999997 | 0.999997 |

and its Table 6 the first moments, which agree to seven digits: `E(Vq) = 0.974087`,
`E(V) = 1.562322`.

### Where transform inversion costs something

Two places, both consequences of the same step, and both checked against an independent arbiter.

**The second moment.** Table 6 of the 2023 paper prints two values for the same quantity:
`E(Vq²) = 2.093599` from its own procedure and `2.105153` from Medhi. The exact computation gives
**2.105153** — Medhi's figure, to the last printed digit. This is not a close call: the
infinite-buffer solver already in this library ([EPIC-032](../epics/EPIC-032-bulk-service-waiting-moments.md),
which shares no code with the tagged chain) independently gives the same number to seven digits,
and the loss probability at `N = 200` is `3·10⁻¹²`, so the two systems are the same system.

The mechanism is stated in the paper itself: the Padé fit is imposed under two side conditions —
that it reproduce the total mass and the first moment. Nothing constrains the rest. So the mean
survives the inversion exactly and the second moment, 0.55% adrift, does not.

**The CDF near the origin.** For Example 4, `M/PH₂^(5,5)/1/N` with `λ = 6`, the agreement from
`t = 4` onwards is six decimals or better. At `t = 2` three published sources already disagree
among themselves — Chaudhry et al. print 0.618004 at `N = 300`, Yu and Tang 0.610358 for the
infinite buffer, and with a loss probability of `10⁻¹¹` those are the same system. The exact value
is **0.612085**, and a simulation of that system puts it at 0.612577 ± 0.001862: a quarter of a
standard error from the exact value and 2.9 away from the published one.

Near the origin is exactly where a rational approximation of a transform has the hardest time,
because that is where the distribution climbs fastest.

**Honest limits of the evidence.** On the second moment the simulation *cannot* adjudicate — the
two candidates are 0.55% apart, far inside what it can resolve, and a test records that explicitly
rather than claiming support it does not have. What settles that one is the agreement of two
independent exact constructions with Medhi's published value, plus the mechanism above. The 2025
MAP paper is paywalled, so its numbers could not be checked directly; what is reproduced here is
its open-access Poisson predecessor, whose method it extends.

### What correlated arrivals change

Arriving customers do **not** see the time-stationary state unless arrivals are Poisson. Under a
MAP the right distribution is the arrival-epoch one, `π D1 / (π D1 1)`, and the difference is not
a nicety. For the bursty two-phase MAP above, the server is busy 34.2% of the time but is busy on
50.8% of arrivals — bursts land on a server that the previous burst already got working. Using the
time-stationary law instead inflates `E(W)` from 0.3956 to 0.5756, **+45%**, which dwarfs any of the
inversion errors this page is about. A test pins this, with the simulation (0.3966) as arbiter.

**Validation:** reproduction of the published Poisson tables above; closed forms in the degenerate
corners (`a = b = 1` with a wide buffer is plain M/M/1 — mean wait, mean sojourn and `P(W = 0) = 1 − ρ`
all exact to twelve digits); Little's law `Lq = λ' E(W)` to machine precision, including at 70%
blocking; the loss probability against the full-buffer state probability under PASTA; the CDF
against the moments by integrating the tail; a one-phase MAP reproducing the Poisson route exactly;
agreement with this library's independent infinite-buffer solver; and a slot-by-slot simulator for
the MAP case, where no published numbers exist. See
[EPIC-083](../epics/EPIC-083-map-ph-bulk-finite-buffer.md).

### Simulation

**Class:** `BulkServiceMapPhSim` (`most_queue.sim.bulk_map_finite`)

```python
import numpy as np
from most_queue.random.map_ph import MAPParams, PHParams
from most_queue.sim.bulk_map_finite import BulkServiceMapPhSim

sim = BulkServiceMapPhSim(a=3, b=8, capacity=25, seed=1)
sim.set_sources(MAPParams(D0=np.array([[-9.0, 0.6], [0.4, -1.4]]),
                          D1=np.array([[8.0, 0.4], [0.2, 0.8]])))
sim.set_servers(PHParams(alpha=np.array([0.3, 0.7]),
                         T=np.array([[-2.0, 1.0], [0.0, -5.0]])))
results = sim.run(total_served=50_000)
results.w[0], results.cdf(1.0, "v"), results.loss_probability
```

**Scope.** One server; the batch service time does not depend on the batch size (the
batch-size-dependent variant is a separate and much harder line of work — see
[batch service](batch.md)); the buffer must hold at least `a`, or no batch could ever form. The
chain grows as `N·a·m_s·m_a`, so very large buffers with many phases get expensive; `N` in the
hundreds with a handful of phases is comfortable.

### Related models in this library

- [Batch / bulk service](batch.md) — the Poisson, infinite-buffer family: Erlang and H2 service,
  batch-size-dependent rates, impatience, multiple servers.
- [Matrix-analytic models (MAP/PH)](map-ph.md) — the same correlated arrival process without batch
  service.
- [SLA / deadline-violation probability](sla.md) — what to do with an exact tail once you have one.
