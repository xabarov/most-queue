# Systems with batch arrivals

[🇷🇺 Русская версия](batch.ru.md) · [← Model catalog](../models.md)

![Batch arrivals diagram](../figures/batch.png)

**In plain words:** jobs arrive not one at a time but in batches of random size — a bus full of
tourists, a bundle of transactions, a batch of tasks from a scheduler. Even at the same average
arrival rate, batching noticeably lengthens the queue: jobs that arrive together are forced to
wait for one another.

### Mˣ/M/1

**Description:** A system where jobs arrive in batches of random size.

**Calculator class:** `BatchMM1`

**Example:**

```python
from most_queue.theory.batch.mm1 import BatchMM1

calc = BatchMM1()
# Batch size probabilities: [p(1), p(2), p(3), ...]
batch_probs = [0.2, 0.3, 0.1, 0.2, 0.2]
calc.set_sources(l=0.5, batch_probs=batch_probs)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/M^[a,b]/1 — bulk (batch) service

**Description:** The server serves customers in **batches**: it starts once at least `a` are queued
and takes up to `b` of them, finishing the whole batch after one exponential batch-service time. This
is the base model for **request batching in LLM inference serving** — the batch-service rate may
depend on the batch size (a bigger batch is slower per batch but amortises fixed cost across
requests, so there is an optimal maximum batch size). Solved as an exact CTMC on (batch-in-service,
number waiting).

**Calculator class:** `BulkServiceMM1Calc` (`most_queue.theory.batch.bulk_service`) ·
**Simulator:** `BulkServiceSim` (`most_queue.sim.bulk_service`)

**Example:**

```python
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc

calc = BulkServiceMM1Calc(a=1, b=8)   # serve up to 8 at a time
calc.set_sources(2.0)
calc.set_servers(lambda size: 1.0 / (0.3 + 0.08 * size))  # LLM-style: batch time grows with size
results = calc.run()  # results.v[0] mean sojourn, results.w[0] mean wait
```

### Exact waiting-time moments (not just the mean)

**Description:** `run()`'s `w` used to be a coarse mean-only estimate (`E[V] - 1/mu(fixed size)`,
inaccurate whenever the batch size actually varies, i.e. whenever `a<b`). `get_w(num=4)` gives the
**exact** raw moments of the waiting time instead, via PASTA: an arriving customer sees the
stationary `(batch-in-service, waiting)` state; if the server is busy, their wait is the remaining
current batch (`Exp(mu(i))`, memoryless) plus however many *full* batches of size `b` are ahead —
a hypoexponential distribution. `run()` now uses this exact result automatically whenever `a=1`
(the common case for GPU/LLM dynamic batching, which has no minimum batch threshold — see
[`docs/research/bulk-service-waiting-moments-2026.md`](../research/bulk-service-waiting-moments-2026.md)
for the GPU-inference literature this targets). `get_n_moments(num=4)` gives exact raw moments of
the number in system for **any** `a`, `b` — trivial, direct summation over the already-solved
stationary distribution.

```python
calc = BulkServiceMM1Calc(a=1, b=8)          # a=1 required for exact W moments
calc.set_sources(2.0)
calc.set_servers(lambda size: 1.0 / (0.3 + 0.08 * size))
w_moments = calc.get_w(num=4)                 # E[W], E[W^2], E[W^3], E[W^4] -- exact
n_moments = calc.get_n_moments(num=4)         # E[N], E[N^2], ... -- exact for any a, b
```

**Accuracy boundary:** `get_w()` raises `ValueError` for `a>1` — the derivation assumes a batch of
`min(b, remaining)` forms immediately once the server frees up, which only holds at `a=1`; for
`a>1` the server can go idle waiting for the threshold to refill after a partial remainder,
breaking the simple decomposition (confirmed numerically: ~18% error at `a=2` before this
restriction was added). `run()` keeps the old approximate mean for `a>1`, now clearly documented
as such. Sojourn-time (`V`) moments beyond the mean are not provided even at `a=1` — a tagged
customer's own eventual batch size depends on arrivals during their own wait, correlated with `W`
itself, so `V ≠ W + S` by simple convolution (see the research doc's "reserve" section).

**See also:** [SLA / deadline-violation probability](sla.md) — turn these moments into a deadline-violation probability or SLO quantile.
