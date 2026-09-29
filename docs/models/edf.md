# EDF (Earliest Deadline First) scheduling

[🇷🇺 Русская версия](edf.ru.md) · [← Model catalog](../models.md)

**In plain words:** every job carries its own deadline (a hard delivery promise, an SLO, a
real-time constraint); whenever the server is free, it picks the *waiting* job whose deadline is
closest, not the one that arrived first (FCFS) or the one from the highest fixed-priority class
(the [static](priority.md)/[dynamic](priority-dynamic.md) priority families). This is different
from the [SLA/deadline layer](sla.md) (EPIC-021): that layer computes a deadline-violation
probability *on top of* an already-fixed discipline's waiting-time distribution — it never changes
who gets served next. Here the deadline **is** the scheduling rule.

### Why there is no exact calculator

Every model on this page's siblings ships an exact theory calculator cross-validated against DES.
EDF is the exception, deliberately: exact finite-state/closed-form analysis of EDF is a genuinely
open problem even for the simplest 2-class M/M/1 case — confirmed both by the literature (only
heavy-traffic/fluid-limit results exist, e.g. Panwar & Towsley 2001) and by three candidate
"surprising reduction" shortcuts that were tried and numerically **disproven** here: exponential
deadlines are *not* equivalent to random-order-of-service; a per-class exponential-clock "race"
does *not* reduce EDF to a simple `(n1,n2)`-level Markov chain; and the classical
work-conservation law does *not* hold exactly once reneging (deadline misses) is non-negligible.
See [`docs/research/edf-scheduling-2026.md`](../research/edf-scheduling-2026.md) for the full
numerical evidence — recorded so nobody re-derives (and re-rejects) the same shortcuts twice.

### EDF discipline (simulation, DES-exact)

**Description:** `K` classes share one server; each job's absolute deadline is fixed at arrival
(`arrival_time + D`, `D` drawn per-class from any supported distribution — deterministic or
random); the waiting job with the smallest absolute deadline is served next; a job whose deadline
elapses before service starts leaves unserved (a miss/reneging event). The discipline itself is
simulated exactly (no approximation in *what happens*) — only the analytical cross-checks below
are approximate/limited in scope.

**Simulation:** `EDFQueueSim` (`most_queue.sim.edf`)

```python
from most_queue.sim.edf import EDFQueueSim

sim = EDFQueueSim(n_classes=2, seed=42)
sim.set_sources([{"type": "M", "params": 0.4}, {"type": "M", "params": 0.4}])
sim.set_servers([{"type": "M", "params": 1.0}, {"type": "M", "params": 1.0}])
sim.set_deadlines([{"type": "M", "params": 0.3}, {"type": "M", "params": 0.6}])  # mean deadline 1/theta per class
res = sim.run(300_000)
# res.w[k], res.v[k] -- waiting/sojourn moments of SERVED class-k jobs
# res.miss_prob[k] -- fraction of class-k arrivals that missed their deadline (never served)
```

Pass `discipline="fcfs"` to get an apples-to-apples FCFS-with-reneging baseline (same arrival,
service and reneging mechanics, only the selection rule changes) — used to confirm EDF reduces the
overall miss rate relative to FCFS on identical parameters.

### Composing with the SLA layer (EPIC-021)

For a *deterministic* per-class deadline, `most_queue.theory.utils.sla.deadline_violation_prob`
applied to the served-jobs' waiting moments gives a genuine (if approximate, fit-based) consistency
check: a served job's realized wait is bounded by its deadline by construction, so the fitted
`P(W > D)` should come out small (not exactly zero, since the smooth H2/Gamma fit doesn't know
about the hard cutoff):

```python
from most_queue.theory.utils.sla import deadline_violation_prob

p = deadline_violation_prob(res.w[0], deadline=3.0)  # should be small
```

### Accuracy and scope

The discipline is DES-exact. The **work-conservation law**
(`sum_k rho_k*E[W_k] = rho*E0/(1-rho)`) is exact only when reneging is negligible (verified to
converge to it as the miss probability shrinks) — it is used as a low-reneging regression test,
not a general cross-check. No exact per-class waiting-time distribution/mean formula is provided;
see [`docs/roadmaps/edf_scheduling_roadmap.md`](../roadmaps/edf_scheduling_roadmap.md) §7 for the
reserve (the 2-class "delayed accumulating priority queue" of Mojalal, Stanford, Taylor & Ziedins
is the closest published exact treatment, but its formulas are paywalled/series-based, not yet
ported here).
