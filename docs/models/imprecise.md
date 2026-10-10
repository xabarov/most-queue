# Imprecise computation / controllable processing times

[🇷🇺 Русская версия](imprecise.ru.md) · [← Model catalog](../models.md)

**In plain words:** under overload, something has to give. The models elsewhere in this catalog
give up *timeliness* (jobs wait longer) or *admission* (jobs are turned away). This one gives up
**answer quality** instead: each job has a mandatory part that must finish, and an optional part
that improves the answer and may be cut short. What is left unexecuted is the job's *error*.

That makes quality a control variable rather than a fixed property of the job — which is exactly
the lever available in an inference pipeline, where a request can be answered with fewer decoding
tokens, fewer refinement passes, or a cheaper model, instead of being delayed or dropped.

### The model

One machine, preemption free. Each task `j` has a release time, a deadline, a mandatory
requirement `m_j` that **must** complete inside its window, an optional requirement `o_j` on top
of it that may be truncated, and a weight `w_j`. The error is the unexecuted optional work, and
the two classical objectives are

- **total weighted error** `Σ w_j e_j` (Shih, Liu & Chung 1991);
- **maximum weighted error** `max_j w_j e_j` (Shih & Liu 1995), plus the lexicographic
  combination — best total first, smallest maximum among the schedules achieving it.

If the mandatory parts alone do not fit, there is no schedule at all; that is reported separately
rather than folded into the error.

### Exact offline optimum

**Class:** `ImpreciseComputationScheduler` (`most_queue.theory.imprecise`)

```python
from most_queue.theory.imprecise import ImpreciseComputationScheduler, ImpreciseTask, verify_schedule

tasks = [
    ImpreciseTask(release=0, deadline=10, mandatory=1, optional=4, weight=1),
    ImpreciseTask(release=0, deadline=10, mandatory=1, optional=4, weight=3),
    ImpreciseTask(release=2, deadline=8,  mandatory=2, optional=3, weight=2),
]

scheduler = ImpreciseComputationScheduler().set_tasks(tasks)
scheduler.is_feasible()                 # can every mandatory part be completed?

result = scheduler.solve("total")       # or "max", or "lexicographic"
result.total_error, result.max_error
result.errors                           # unexecuted optional work, per task
result.schedule                         # [(start, end, task index), ...]
verify_schedule(tasks, result.schedule) # independent re-check of the schedule
```

On that instance the three objectives give genuinely different answers:

| objective | total error | max error | per-task errors |
|---|---|---|---|
| `total` | 6.00 | 4.00 | 4.0, 0.0, 1.0 |
| `max` | 8.18 | 2.73 | 2.73, 0.91, 1.36 |
| `lexicographic` | 6.00 | 4.00 | 4.0, 0.0, 1.0 |

Minimising the total lets one task absorb all the damage; minimising the maximum spreads it at a
35% higher total. The lexicographic objective is the one to use when both matter — here it cannot
improve on the `total` solution, but when it can, it does (two identical tasks forced to shed two
units get `1.0 + 1.0` rather than `2.0 + 0.0`).

**Method.** Implementation of Shih W.-K., Liu J.W.S., Chung J.-Y. & Gillies D.W., *Scheduling
tasks with ready times and deadlines to minimize average error*, ACM SIGOPS OSR 23(3):14–28, 1989,
[doi:10.1145/71021.71022](https://doi.org/10.1145/71021.71022); Shih, Liu & Chung, *Algorithms for
scheduling tasks to minimize total error*, SIAM J. Computing 20(3):537–552, 1991; Shih & Liu,
IEEE Trans. Computers 44(3):466–471, 1995 — reviewed and unified with the controllable-processing-
times literature by Shioura A., Shakhlevich N.V. & Strusevich V.A., EJOR 266(3):795–818, 2018,
[doi:10.1016/j.ejor.2017.08.034](https://doi.org/10.1016/j.ejor.2017.08.034). **The model and the
solved problems are theirs.**

The literature states the problem as a minimum-cost maximum-flow network. This implementation
solves the equivalent linear program instead — the data here are real-valued rather than integral,
and the LP extends to the max-error objective without rebuilding the network. Cut the timeline at
every release and deadline; let `x[j][k]` be the time task `j` spends in interval `k`, allowed
only when the whole interval sits inside `[release_j, deadline_j]`. The interval-capacity rows are
exactly Horn's feasibility condition, so every LP-feasible solution **is** a real schedule, and
the schedule is emitted and re-checked.

**Validation:** exhaustive search over unit-slot assignments on small integer instances (value,
feasibility verdict and schedule all agree); a minimum-cost maximum-flow solved by network simplex
— the literature's own formulation, so agreeing with it confirms the LP encodes the same problem;
closed forms in the degenerate corners. See
[EPIC-080](../epics/EPIC-080-imprecise-computation.md).

### Online policies, scored against the optimum

**Class:** `ImpreciseComputationSim` (`most_queue.sim.imprecise`)

```python
from most_queue.sim.imprecise import ImpreciseComputationSim, compare_with_offline

sim = ImpreciseComputationSim(policy="mandatory_first", seed=1)
sim.set_sources(1.5, "M")            # Poisson arrivals, rate 1.5
sim.set_servers(5.0, 5.0, "M")       # mandatory and optional work, rates (means 0.2 each)
sim.set_deadlines(2.0, "D")          # deterministic relative deadline

jobs = sim.generate_jobs(150)
compare_with_offline(sim, jobs)      # the online error against the exact optimum, same jobs
```

Because the offline problem is solved exactly, an online rule can be measured against *the best
any scheduler could have done on the very same jobs* — not against another heuristic, not against
an asymptotic bound. For the simple mandatory-first rule (mandatory work in deadline order first,
optional work only when nothing mandatory is pending), pooled over 10 seeds × 150 jobs:

| `λ` | comparable instances | total online error | total offline optimum | ratio |
|---|---|---|---|---|
| 1.0 | 9/10 | 9.70 | 3.62 | **2.68** |
| 1.5 | 9/10 | 24.11 | 13.33 | **1.81** |
| 2.0 | 9/10 | 49.29 | 33.05 | **1.49** |
| 2.5 | 8/10 | 65.95 | 50.61 | **1.30** |

The gap is *widest at light load*, which is the opposite of the intuition that overload is where
scheduling matters. The reason is that when there is slack, an optimal schedule can place work so
that almost nothing is lost, so the naive rule's small absolute loss is a large multiple of a
small number; under heavy load everyone has to shed a lot and the relative gap closes. If a system
is sized for the mean and the quality loss is small in absolute terms, that is exactly where a
better policy buys the most proportionally.

**One deliberate counterexample.** The second policy, `full_edf`, serves whole jobs in deadline
order without prioritising mandatory work. It therefore *misses mandatory deadlines* — a hard
failure in this model — and then reports a **lower** optional error than the offline optimum,
because it bought that error by skipping work it was not allowed to skip.
`compare_with_offline` detects this (`online_mandatory_misses`, `comparable`) and refuses to
report a ratio rather than printing a flattering number. It is kept precisely as the concrete
demonstration of why every policy in the literature serves mandatory parts first.

**Scope.** The two online policies are the obvious operating points, **not** the literature's
optimal online algorithms (Shih & Liu, RTSS 1992, give those). Parallel machines, periodic task
sets and non-linear error functions are all studied in the cited literature and are not
implemented here.

### Related models in this library

- [EDF scheduling](edf.md) — deadline-aware service order, but the job is all-or-nothing; here
  it can be answered partially.
- [Deadline-aware admission control](admission-control.md) — gives up *admission* under overload
  instead of quality.
- [SLA / deadline-violation probability](sla.md) — measures missed deadlines without changing
  what the system does about them.
- [Size-based scheduling](size-based.md) — reorders by size; the work itself is fixed.
