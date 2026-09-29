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

### Lost-sales variant

**Description:** Set `policy="lost_sales"`: an arrival that finds the stock empty (`i = 0`) is
turned away instead of queueing — more realistic for retail/e-commerce, where a customer seeing
"out of stock" leaves rather than joining a virtual queue. The difference from backordering is
**exactly one transition**: an arrival at `i = 0` is not a state change at all under lost sales (it
never enters the CTMC), so `A0`/`B01` lose their phase-0 row and the `i = 0` diagonal entry of
`A1`/`B00` loses the `λ` term — everything else (service, replenishment) is identical. Implemented
as a parameter on the same calculator, not a separate class. Classic formulation: Saffari, Haji &
Hassanzadeh, *The M/M/1 queue with inventory, lost sale, and general lead times*, Queueing
Systems, 2013.

Mean sojourn/queue length are computed via Little's law using the **effective** (accepted)
arrival rate `λ·(1 − loss_prob)`, not the nominal `λ` — a subtlety worth knowing if you reuse the
internals: `level n` under lost sales only counts customers that were actually admitted.

```python
calc = MM1QueueingInventoryCalc(s_max=4, policy="lost_sales")
calc.set_sources(l=0.5)
calc.set_servers(mu=1.0, theta=0.5)
res = calc.run()
# res.loss_prob = P(an arriving customer is turned away) -- equals stockout_prob by PASTA
```

### General (s,S) policy

**Description:** pass `s` (reorder point, `0 <= s < s_max`) to either calculator: replenishment
triggers as soon as stock drops to `s`, not only when it hits 0 — the usual generalization used to
keep a safety-stock buffer instead of running all the way to a stockout before reordering.
Surprisingly, this needs **no extra state dimension**: because the lead time is exponential
(memoryless), "an order is currently in transit" is fully determined by `i <= s` alone, exactly as
`i == 0` alone determined it in the `(0,S)` special case — so the QBD phase space stays
`i ∈ {0,...,S}` and only the replenishment transition's active range changes (`i <= s` instead of
`i == 0`). `s=0` reproduces the `(0,S)` models above exactly. Compatible with both `policy`
values. See [`docs/research/queueing-inventory-general-sS-2026.md`](../research/queueing-inventory-general-sS-2026.md).

```python
calc = MM1QueueingInventoryCalc(s_max=4, s=2, policy="backorder")  # reorder as soon as stock hits 2
calc.set_sources(l=0.5)
calc.set_servers(mu=1.0, theta=0.5)
res = calc.run()
# larger s (more safety stock) never increases stockout_prob or mean wait
```

### Multi-server case (M/M/c)

**Description:** `MMcQueueingInventoryCalc(c, s_max, s=0, policy="backorder")` generalizes the
`(s,S)` model above to `c` identical servers, each service still consuming one unit from the same
shared stock. The number of busy servers is `min(n, c)` — a deterministic function of the number
of customers `n` in the system, exactly as in the ordinary `M/M/c` queue — so no extra state
dimension is needed. What *does* change is that the service rate depends on `n` while `n < c`
(rate `n·μ`, not all servers saturated yet) and only becomes level-independent (`c·μ`) once
`n ≥ c`, giving `c` distinct boundary levels instead of one; these are stacked into a single
super-block for the QBD solver (still the same `QBDSolver`, no new numerical method). `c=1`
reproduces `MM1QueueingInventoryCalc` exactly. See
[`docs/research/queueing-inventory-multiserver-2026.md`](../research/queueing-inventory-multiserver-2026.md)
(Yue, Zhao & Yue 2016; Krishnamoorthy, Manikandan & Dhanya 2015) and
[`docs/roadmaps/queueing_inventory_multiserver_roadmap.md`](../roadmaps/queueing_inventory_multiserver_roadmap.md)
for the block derivation.

```python
from most_queue.theory.inventory import MMcQueueingInventoryCalc

calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1)  # 2 servers, reorder at stock=1
calc.set_sources(l=1.0)
calc.set_servers(mu=1.0, theta=0.5)
res = calc.run()
# rho = l / (c*mu) must be < 1 -- necessary, not sufficient (stockouts can block all servers)
```

### Heterogeneous servers (c=2)

**Description:** `MM2QueueingInventoryHeterogeneousCalc(s_max, s=0, policy="backorder")` drops the
"identical servers" assumption of the `M/M/c` model above for the `c=2` case: server 1 and server 2
may have different rates `mu1 != mu2`, sharing the same stock pool. Combines two already-validated
techniques rather than new math: the state-splitting technique for two heterogeneous exponential
servers (Krishnamoorthi 1963, already used for [machine repair](reliability.md) and
[priority queues](priority-dynamic.md)) with the stacked-boundary-superblock QBD trick from the
`M/M/c` model above. State tracks *which* server is the sole busy one only when there's exactly one
customer in the system (`n=1`) — for `n=0` nobody is busy, for `n≥2` both are, so no ambiguity
there. `mu1=mu2` reproduces `MMcQueueingInventoryCalc(c=2, ...)` exactly. See
[`docs/research/queueing-inventory-heterogeneous-servers-2026.md`](../research/queueing-inventory-heterogeneous-servers-2026.md)
and
[`docs/roadmaps/queueing_inventory_heterogeneous_servers_roadmap.md`](../roadmaps/queueing_inventory_heterogeneous_servers_roadmap.md)
for the block derivation. Limited to `c=2` — general `c` heterogeneous servers would need tracking
*which subset* of servers is busy at every intermediate level (`n=1,...,c-1`), a combinatorial
state-space blow-up left as a reserve item.

```python
from most_queue.theory.inventory import MM2QueueingInventoryHeterogeneousCalc

calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1)
calc.set_sources(l=1.0)
calc.set_servers(mu1=1.5, mu2=0.7, theta=1.0)  # server 1 faster than server 2
res = calc.run()
```

### Accuracy and scope

Exact (matrix-geometric QBD, not an approximation) for `(0,S)`/general `(s,S)`, `c=1`, identical
`c>1` servers, or `c=2` heterogeneous servers, backorder or lost-sales — all of the models above.
