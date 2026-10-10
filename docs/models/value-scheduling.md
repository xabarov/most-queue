# Value-based scheduling under overload

[🇷🇺 Русская версия](value-scheduling.ru.md) · [← Model catalog](../models.md)

**In plain words:** once the offered load passes one, not every job can make its deadline, and the
interesting question stops being "how long do jobs wait" and becomes "which jobs are worth
running". Each job carries an importance **value**, collected in full if it finishes by its
deadline and lost entirely otherwise. The figure of merit is the value banked, not any waiting
time.

That reframing matters because the obvious discipline is the wrong one. Earliest-deadline-first is
optimal while the system is underloaded and collapses once it is not: it keeps serving the most
urgent job even when that job, and several behind it, are already doomed — the *domino effect*.
Under overload a scheduler has to decide what to abandon, and EDF has no opinion about that.

### The model

One machine, preemption free, arrivals unannounced. Each job has a worst-case computation time
`C`, an actual computation time no larger than it, an absolute deadline, and a value `V`. A job
that misses its deadline banks nothing, so running it further is pure waste and it is abandoned
the moment the deadline passes. The metric is the **Hit Value Ratio** — cumulative value banked
over the total value of the task set.

Note the distinction between *nominal* load, computed from worst-case times, and *actual* load.
With actual times averaging half the worst case, a nominal load of two is an actual load of one,
so genuine overload begins only above a nominal load of two. The exact optimum below confirms
this: up to nominal load two, nothing need be lost at all.

### Twelve algorithms: four priority rules × three guarantee mechanisms

**Class:** `ValueSchedulingSim` (`most_queue.sim.value_scheduling`)

Which ready job runs now:

| rule | priority | ignores |
|---|---|---|
| `edf` | earliest deadline | value |
| `hvf` | highest value | urgency |
| `hdf` | highest value **density**, `V` over remaining worst-case time | — |
| `mix` | `alpha * V - (1 - alpha) * deadline` | — |

What happens when the ready set cannot all make it:

| mechanism | behaviour |
|---|---|
| `plain` | nothing is rejected; overload is absorbed by missed deadlines |
| `guaranteed` | an acceptance test at every arrival rejects the **newly arrived** job if it predicts an overflow — regardless of how valuable the newcomer is |
| `robust` | the same test, but it rejects the **least valuable** job whose removal clears the overload, and parks rejects rather than discarding them: when a job finishes early, a parked job is reclaimed if it now fits |

The product gives the paper's names — `edf` + `guaranteed` is GEDF, `hdf` + `robust` is RHDF, and
so on; the `algorithm` property spells it out.

```python
from most_queue.sim.value_scheduling import ValueSchedulingSim

sim = ValueSchedulingSim(priority="hdf", guarantee="robust", seed=1)
jobs = sim.generate_task_set(nominal_load=3.0)   # the paper's task-set generator
result = sim.run_on(jobs)

sim.algorithm                 # 'RHDF'
result.hit_value_ratio        # fraction of the total value banked
result.completed, result.missed, result.rejected
result.reclaimed              # parked jobs brought back
```

`run_on` leaves the task set untouched, so every algorithm can be scored on the same realisation.

**The acceptance test** lays the ready jobs out in the order the algorithm will actually run them,
charging worst-case remaining times, and asks whether any finishes late. For `edf` this is
necessary and sufficient — all ready jobs are available immediately and future arrivals are
unknowable to an on-line scheduler, so Horn's optimality of EDF applies. For a value-based
ordering it answers the narrower question "will the schedule I am about to produce miss a
deadline". Either way, since actual times never exceed worst-case ones, an admitted job always
makes its deadline: the guarantee classes trade missed deadlines for rejections.

**Method.** Implementation of Buttazzo G.C., Spuri M. & Sensini F., *Value vs. deadline scheduling
in overload conditions*, Proc. 16th IEEE Real-Time Systems Symposium (RTSS), Pisa, 1995,
pp. 90–99, [doi:10.1109/REAL.1995.495198](https://doi.org/10.1109/REAL.1995.495198). **The model,
the twelve algorithms and the experimental design are theirs.**

### What the experiments say

The paper's task-set generator, reproduced by `generate_task_set`: 100 streams, worst-case times
uniform on [50, 350], laxity uniform on [150, 1850], relative deadline `C + laxity`, actual
execution time uniform on [0, `C`], stream interarrival exponential with mean `N·C / load`. Values
are either independent of everything else (`value_mode="random"`) or equal to the relative
deadline (`value_mode="linear"`, the hard case — the valuable jobs are also the ones most likely
to miss). Below: mean HVR over 10 runs, 100 streams, horizon 30 000.

**Plain class — value density wins, and EDF collapses:**

| nominal load | EDF | HVF | HDF | MIX |
|---|---|---|---|---|
| 1.0 | 1.000 | 0.993 | 0.996 | 0.999 |
| 2.0 | 0.954 | 0.942 | 0.961 | 0.973 |
| 2.5 | 0.799 | 0.888 | **0.917** | 0.909 |
| 3.0 | 0.655 | 0.818 | **0.865** | 0.840 |
| 3.5 | 0.544 | 0.746 | **0.814** | 0.757 |

EDF loses 46 points between light load and a nominal load of 3.5; HDF loses 18. Note also that
MIX beats *both* of the rules it blends at every load — its behaviour is not an average of theirs.

**Guaranteed class — the acceptance test reverses the ranking:**

| nominal load | GEDF | GHVF | GHDF | GMIX |
|---|---|---|---|---|
| 2.0 | **0.941** | 0.829 | 0.843 | 0.890 |
| 2.5 | **0.848** | 0.738 | 0.755 | 0.800 |
| 3.0 | **0.744** | 0.668 | 0.662 | 0.704 |
| 3.5 | **0.657** | 0.590 | 0.586 | 0.607 |

GHDF, the best plain rule, becomes the worst guaranteed one. Rejecting the newcomer regardless of
its value is what does the damage, and EDF is the one rule that never cared about value anyway.

**Robust class — everything recovers, and the four rules converge:**

| nominal load | REDF | RHVF | RHDF | RMIX |
|---|---|---|---|---|
| 2.0 | **0.981** | 0.939 | 0.958 | 0.974 |
| 2.5 | **0.923** | 0.882 | 0.914 | 0.919 |
| 3.0 | **0.864** | 0.820 | 0.863 | 0.859 |
| 3.5 | 0.793 | 0.746 | **0.808** | 0.785 |

REDF leads until a nominal load of about 3, past which RHDF edges ahead — when the demand is far
beyond the available time, value density is the better guide.

**Read along the rows instead** (nominal load 3.0) and the mechanisms separate cleanly:

| priority | plain | guaranteed | robust |
|---|---|---|---|
| EDF | 0.655 | 0.744 | **0.864** |
| HVF | 0.818 | 0.662 | 0.820 |
| HDF | **0.865** | 0.662 | 0.863 |
| MIX | 0.840 | 0.704 | 0.859 |

The bare acceptance test *hurts* every value-aware ordering and only helps EDF. Reclaiming repairs
the damage completely — the robust column matches or beats the plain one everywhere.

**Varying the actual load without touching the nominal one.** `unused_ratio` fixes the paper's
`beta = 1 - actual/worst`, so a nominal load of 3 spans actual loads from 2.6 down to 0.4:

| `beta` | actual load | EDF | GEDF | REDF |
|---|---|---|---|---|
| 0.125 | 2.62 | 0.234 | 0.462 | **0.616** |
| 0.25 | 2.25 | 0.300 | 0.542 | **0.686** |
| 0.5 | 1.50 | 0.588 | 0.758 | **0.867** |
| 0.75 | 0.75 | **1.000** | 0.992 | **1.000** |
| 0.875 | 0.38 | **1.000** | 0.999 | **1.000** |

The crossover is the practical lesson: admission control is worth a great deal under overload and
becomes a liability in underload, where GEDF rejects jobs that would have finished in time because
the test judged them on pessimistic worst-case figures. The robust class, which can take those
rejections back, is the only column that is never wrong.

### The exact clairvoyant optimum

**Class:** `ClairvoyantValueScheduler` (`most_queue.theory.value_scheduling`)

The HVR above is normalised by the *total* value of the task set, which under overload no
scheduler can collect. So an HVR of 0.8 might be poor or nearly perfect, and the table cannot tell
you which. This part is **not** in the paper, which remarks that the robust algorithms come "close
to the best achievable by an on-line algorithm" without measuring the best achievable.

Fixing the denominator means solving the off-line problem exactly: choose a subset of jobs to
complete, maximising total value, knowing the whole arrival sequence and every actual execution
time in advance.

```python
from most_queue.sim.value_scheduling import ValueSchedulingSim, compare_with_clairvoyant

sim = ValueSchedulingSim(priority="edf", guarantee="robust", seed=0)
jobs = sim.generate_task_set(3.0, num_streams=20, horizon=5000.0)

report = compare_with_clairvoyant(sim, jobs)
report["hit_value_ratio"]              # the paper's metric
report["clairvoyant_hit_value_ratio"]  # the best any scheduler could do here
report["competitive_ratio"]            # banked / optimal -- never above 1
```

| nominal load | jobs | clairvoyant HVR | EDF | REDF | RHDF | RMIX |
|---|---|---|---|---|---|---|
| 1.0 | 27 | 1.000 | 1.000 | 1.000 | 0.996 | 0.997 |
| 2.0 | 60 | 0.998 | 0.982 | 0.992 | 0.952 | 0.977 |
| 2.5 | 78 | 0.984 | 0.926 | 0.965 | 0.938 | 0.955 |
| 3.0 | 91 | 0.958 | 0.795 | 0.943 | 0.902 | 0.929 |
| 3.5 | 109 | 0.920 | 0.709 | 0.906 | 0.906 | 0.898 |

Two things the paper's normalisation hides. First, the clairvoyant HVR stays at 1.000 through
nominal load two, which pins the onset of real overload exactly where the paper's reasoning puts
it. Second, the robust class is genuinely close to optimal — REDF banks 94% of the achievable
value at nominal load 3, where its raw HVR of 0.864 looks far more modest. Plain EDF, by contrast,
leaves a fifth of the achievable value on the table.

**Method.** For a *fixed* subset, schedulability on one preemptive machine with release dates and
deadlines is decided by **Horn's condition**: the subset is feasible if and only if, for every
pair of times `t_a < t_b`, the jobs whose windows nest inside `[t_a, t_b]` demand no more than
`t_b - t_a` of machine time. Only pairs where `t_a` is a release and `t_b` a deadline can bind, so
with a binary `y_j` per job,

```
maximise    sum_j  value_j * y_j
subject to  sum_{j: t_a <= r_j, d_j <= t_b}  p_j * y_j  <=  t_b - t_a     for every such pair
            y_j in {0, 1}
```

is an exact reformulation, not a relaxation: its feasible points are precisely the schedulable
subsets. The selection is NP-hard — it contains knapsack — which is what the integrality is for;
the *scheduling* half of the problem is what Horn's condition dissolves entirely.

Two exact reductions keep it tractable. The instance is split into blocks separated by points no
job's window spans, since a Horn row crossing such a cut splits into two rows whose capacities
add up and is therefore implied. Within a block, rows selecting the same job set are deduplicated
to the tightest capacity, and a row whose jobs all together fit is dropped — no subset of them can
violate it. At nominal load 3.5 that leaves about 3 500 rows for 110 jobs, solved in under a
second. A few hundred jobs per block is the practical ceiling, and `max_constraints` refuses
larger instances rather than thrashing.

Because the constraints *are* Horn's condition, an actual schedule follows: the selected jobs are
handed to the interval linear program of
[`most_queue.theory.imprecise.offline`](imprecise.md) as entirely-mandatory tasks, and re-checked
by that module's independent `verify_schedule`.

**Validation:** exhaustive search over all subsets on small instances, with each subset's
feasibility decided by **running preemptive EDF** rather than by Horn's condition — keeping the
reference independent of the formulation under test; agreement with the un-reduced row set, so the
two speedups are shown not to change answers; the emitted schedule re-checked by a separately
written verifier; and, across 300 runs of all twelve algorithms, no competitive ratio above one.
Closed forms in the degenerate corners (one job, two jobs competing, a knapsack corner where the
optimum is not greedy by value). See
[EPIC-081](../epics/EPIC-081-value-vs-deadline-scheduling.md).

**Scope.** No on-line algorithm can guarantee better than 1/4 of the clairvoyant optimum under
overload, whatever the value density (Baruah S. et al., *On the competitiveness of on-line
real-time task scheduling*, Real-Time Systems 4(2):125–144, 1992,
[doi:10.1007/BF00365406](https://doi.org/10.1007/BF00365406)). That bound is adversarial; on these
task sets the measured ratios are far above it, which is the useful thing to know and is not
something the bound can tell you. One machine only; the twelve algorithms are the paper's, not the
optimal on-line algorithms of the competitive-analysis literature.

**One deviation from the paper.** On the *linear* task set in the guaranteed class, GMIX edges
GEDF at nominal loads above 3 (+2.9σ over 30 runs at 3.5), where the paper's figure 3b shows GEDF
ahead everywhere. The cause is a degeneracy of the MIX rule rather than a disagreement about EDF:
`alpha * V - (1 - alpha) * d` adds a value to a time, so it is not invariant to how the two are
scaled, and when the value *is* the relative deadline, `alpha = 0.5` makes the key
`0.5(d - a) - 0.5 d = -0.5 a` — the rule collapses exactly to first-come first-served and stops
consulting value or urgency at all. "GMIX" on the linear set is guaranteed FCFS. Observation 2
holds at every load on the random set, and against both genuinely value-based rules on the linear
set.

### Related models in this library

- [EDF scheduling](edf.md) — the same deadline-driven discipline without values, and without any
  overload mechanism; this page is what happens to it when the load passes one.
- [Deadline-aware admission control](admission-control.md) — accept/reject at arrival with exact
  analysis for exponential deadlines; the `guaranteed` class here is the value-aware, worst-case
  version of the same idea.
- [Imprecise computation](imprecise.md) — the third response to overload: degrade the answer
  rather than drop the job. Supplies the interval LP used here to exhibit schedules.
- [SLA / deadline-violation probability](sla.md) — measures missed deadlines without changing what
  the system does about them.
