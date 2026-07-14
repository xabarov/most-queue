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
