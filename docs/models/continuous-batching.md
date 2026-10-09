# Systems with occupancy-dependent continuous batching

[🇷🇺 Русская версия](continuous-batching.ru.md) · [← Model catalog](../models.md)

![Occupancy-dependent continuous batching diagram](../figures/continuous_batching.png)

**In plain words:** modern LLM-inference engines (vLLM/SGLang-style "continuous batching")
don't serve a fixed group of requests atomically like classical bulk service (see
[batch arrivals](batch.md)) — a request joins and leaves a shared, capacity-capped pool
independently, one token-generation step at a time. The cap (`k`) on how many requests can be
resident at once comes from available accelerator memory (KV-cache), and the per-request
completion rate commonly *depends on how many others are currently resident* — a bigger shared
batch means more amortized throughput per step, but also more contention for compute/memory
bandwidth.

### Occupancy-dependent queue with a concurrency cap

**Description:** Poisson arrivals; up to `k` requests are served concurrently with a per-request
completion rate `mu(occupancy)` that may depend on how many are *currently* resident (not fixed per
server, as in classical M/M/c); requests beyond the cap wait FCFS. The state-dependent birth-death
chain this produces has an exact (closed-form) stationary distribution, and — because occupancy is
pinned at exactly `k` whenever anyone waits — the waiting time of a newly-arriving request is an
exact mixture of Erlang distributions, giving exact raw moments *and* an exact tail `P(W>t)`, not
just the mean (unlike the one directly comparable recent treatment of this exact system class,
which is explicitly approximate and mean-only).

**Calculator class:** `OccupancyDependentQueueCalc`
(`most_queue.theory.continuous_batching.occupancy_dependent`)

```python
from most_queue.theory.continuous_batching import OccupancyDependentQueueCalc

calc = OccupancyDependentQueueCalc(k=8, queue_truncation=200)   # k = concurrency cap (KV-cache slots)
calc.set_sources(l=4.0)
calc.set_servers(lambda occupancy: 2.0 / (0.5 + 0.3 * occupancy))  # slower per-request as batch grows
res = calc.run()                                 # res.w -- exact raw moments of waiting time
p_wait = calc.get_p_wait()                       # exact P(W > 0) -- cap already full on arrival
p_violation = calc.get_tail(0.5)                 # exact P(W > 0.5) -- SLA-style tail, not approximated
```

`mu` may also be a plain scalar (occupancy-independent) — this exactly reduces to the classical
`M/M/k/N` queue (`most_queue.theory.fifo.mmnr.MMnrCalc`), and `k=1` reduces to classical `M/M/1/N`.

**Accuracy and scope:** exact for any `k` and any occupancy-dependent rate function. Models a fixed
(exogenous) concurrency cap `k` and an occupancy-independent per-request memory footprint — the
*growing* per-request KV-cache footprint seen in real engines, and non-exponential (phase-type)
remaining-service-length, are explicit, documented reserves for future work (see EPIC-073). `W` is
the queueing delay before admission into the active pool only, not the full sojourn time.
