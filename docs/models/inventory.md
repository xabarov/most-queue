# Queueing-inventory systems

[🇷🇺 Русская версия](inventory.ru.md) · [← Model catalog](../models.md)

![Queueing-inventory diagram](../figures/inventory.png)

**In plain words:** a repair shop, a pharmacy counter, a 3D-printing bureau — anywhere a server
needs a physical (or licensed, or reserved) **unit of stock** to complete each job, not just its
own time. When stock runs out, the server can't work even if it's free and a customer is waiting —
the customer *backorders* (waits) until a replenishment order arrives. This is qualitatively
different from vacations (server idles by its own schedule) or breakdowns (server itself fails):
here an *external resource* gates service, and its own replenishment dynamics couple back into
queue delay.

### M/M/1, (0,S) policy, backordering

**Description:** Poisson arrivals, exponential service that consumes one stock unit on
completion. Stock is capped at `S`; the moment it hits 0, a replenishment order for `S` units is
placed automatically and arrives after `Exp(θ)` ("positive lead time"). While stock is 0, arriving
customers still queue (**backorder** — never lost) but the server is blocked until restock. Exact
solution: the system is precisely a QBD process — level = number of customers `n` (unbounded),
phase = stock level `i ∈ {0,...,S}` — solved via the library's general QBD solver (the same
logarithmic-reduction machinery behind the MAP/PH stack), not a new numerical method. Classic
formulation: Schwarz & Daduna, *M/M/1 Queueing systems with inventory*, Queueing Systems, 2006.

Reduces exactly to plain M/M/1 as `S` and `θ` grow large (stock essentially never runs out) — a
useful sanity check when tuning parameters.

**Calculator class:** `MM1QueueingInventoryCalc` (`most_queue.theory.inventory`)
**Simulation:** `MM1QueueingInventorySim` (`most_queue.sim.inventory`)

**Example:**

```python
from most_queue.theory.inventory import MM1QueueingInventoryCalc

calc = MM1QueueingInventoryCalc(s_max=4)     # stock capped at 4
calc.set_sources(l=0.5)                       # arrival rate
calc.set_servers(mu=1.0, theta=0.5)           # service rate, replenishment rate
res = calc.run()
# res.v[0] mean sojourn, res.w[0] mean wait (V = W + 1/mu exactly, FCFS)
# res.stockout_prob = P(stock == 0), res.fill_rate = 1 - stockout_prob
# res.stock_distribution[i] = P(stock level == i)
```

### Accuracy and scope

Exact (matrix-geometric QBD, not an approximation) for the `(0,S)` backorder model above. Two
natural extensions are **not yet implemented** (see
[`docs/research/queueing-inventory-2026.md`](../research/queueing-inventory-2026.md) for the
gap-analysis and literature): the **lost-sales** variant (arrivals during a stockout are turned
away rather than queueing) and the general **`(s,S)` policy** (`s > 0`, requiring an extra "order
already placed" state bit). Both are QBD-compatible extensions of the same solver, just a
different block structure.
