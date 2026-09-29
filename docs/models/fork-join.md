# Fork-Join systems

[🇷🇺 Русская версия](fork-join.ru.md) · [← Model catalog](../models.md)

![Fork-Join diagram](../figures/fork_join.png)

**In plain words:** on arrival, a job splits (fork) into several parts, the parts are served
in parallel on different servers, and the result is ready only when all parts are collected
(join). This is how parallel computing, RAID arrays, and distributed queries (map-reduce) work.
The response time is determined by the *slowest* part — which is why the mean sojourn time
grows with the number of parts even for the same total amount of work.

### M/M/c Fork-Join

**Description:** A system where a job splits into several parts served in parallel and then rejoined.

**Calculator class:** `ForkJoinMarkovianCalc`

**Example:**

```python
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc

calc = ForkJoinMarkovianCalc(n=5, k=2)  # 5 servers, 2 required
calc.set_sources(l=1.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/G/c Split-Join

**Description:** A Split-Join system with a general service time distribution.

**In plain words:** the strict variant of Fork-Join — the next job does not begin service until
the previous one has been fully reassembled (a synchronous batch pipeline).

**Calculator class:** `SplitJoinCalc`

**Example:** See the test `test_fj_sim.py`

### Heavy-tailed sub-task service time (Pareto) — exact, not approximated

**In plain words:** the light-tailed path above approximates the max of n sub-task service times
by moment-matching an H2/Gamma/Erlang distribution and numerically integrating (Gauss-Laguerre
quadrature). For **Pareto**-distributed sub-tasks that approximation is both unnecessary and wrong
— the max of n iid Pareto(α, K) has an exact closed form (substituting `U = F(X) ~ Uniform(0,1)`
turns the max of n Pareto into the max of n Uniform, whose moments are a Beta-function integral):

```
P(max_n > x) = 1 - (1 - (K/x)^α)^n                (exact tail/CDF, any α > 0)
E[max_n^k]   = K^k · n · B(n, 1 - k/α)             (exact moments, only for k < α)
```

Moments of order ≥ α do not exist (same as for a single Pareto) — `pareto_max_moments` raises a
clear error rather than returning `NaN`/`inf`. `SplitJoinCalc(approximation="pareto")` composes
this exact maximum with the exact M/G/1 Pollaczek-Khinchine formula for the full sojourn time —
valid when α > 2 (finite variance, required by P-K). For α ≤ 2, use `pareto_max_tail` directly for
the exact tail of the slowest sub-task; the full-queue sojourn-time asymptotic in that
infinite-variance regime ("one big jump" — Boxma et al., Queueing Systems 2022) is not yet
implemented, see [`docs/research/fork-join-heavy-tail-2026.md`](../research/fork-join-heavy-tail-2026.md).

**Functions/classes:** `pareto_max_moments`, `pareto_max_tail`
(`most_queue.theory.utils.max_dist`); `SplitJoinCalc(calc_params=CalcParams(approx_distr="pareto"))`

**Example:**

```python
from most_queue.random.utils.params import ParetoParams
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fork_join.split_join import SplitJoinCalc

calc = SplitJoinCalc(n=4, calc_params=CalcParams(approx_distr="pareto"))
calc.set_sources(l=0.3)
calc.set_servers(ParetoParams(alpha=3.5, K=1.0))   # alpha > 2: finite variance
results = calc.run()   # exact sojourn-time moments
```

**See also:** [SLA / deadline-violation probability](sla.md) — its H2/Gamma fit assumes finite
variance and is **not valid** for α ≤ 2; use `pareto_max_tail` (exact) instead for heavy-tailed
fork-join deadlines.

### Heterogeneous branches and series-parallel task DAGs

**In plain words:** the models above all assume every branch has the *same* service-time
distribution. Real task graphs rarely do — a MapReduce job's mappers run on different-sized
shards, a microservice call fans out to endpoints with different latency profiles. This section
drops the "identical branches" assumption in two independent ways: branches with *different*
distributions in one fork-join level, and a *task graph* (nested series/parallel composition, not
just one flat level).

**Description:** `n` independent branches, each its own family/parameters (any mix of the
library's Pareto/Gamma/H2/Erlang). No closed form exists in general (unlike i.i.d. Pareto above),
but the moments are still numerically exact given each branch's assumed family — via the standard
identity `E[max^k] = k∫t^(k-1)P(max>t)dt`, `P(max>t) = 1 - Π(1-tail_i(t))`, evaluated with
`scipy.integrate.quad`. Reduces exactly to `pareto_max_moments` when all branches happen to be the
same Pareto (regression-tested).

**Functions:** `heterogeneous_max_moments`, `branch_tail` (`most_queue.theory.utils.max_dist`)

```python
from most_queue.theory.utils.max_dist import heterogeneous_max_moments
from most_queue.random.utils.params import ParetoParams

branches = [
    ("pareto", ParetoParams(alpha=5.0, K=1.0)),
    ("pareto", ParetoParams(alpha=6.0, K=1.5)),
    ("gamma", [2.0, 6.0, 24.0]),   # raw moments -- fitted internally
]
moments = heterogeneous_max_moments(branches, num=3)  # E[max], E[max^2], E[max^3]
```

**DAG composition:** `ForkJoinDAGCalc` recursively composes a small nested-tuple spec —
`("leaf", family, spec)`, `("series", [node, ...])` (sequential, sum via `conv_moments`),
`("parallel", [node, ...])` (fork-join, max via `heterogeneous_max_moments`) — matching the
well-known **series-parallel-reducible** class of task graphs (MapReduce stages, microservice call
chains, CI/CD pipelines — the overwhelming majority of real task graphs; see
[`docs/research/fork-join-dag-heterogeneous-2026.md`](../research/fork-join-dag-heterogeneous-2026.md)).
Exact at a `parallel` node whose children are raw leaves; a `parallel` node whose children are
themselves composite subtrees needs one extra moment-fit step (same standard `SplitJoinCalc`
already uses for its non-Pareto i.i.d. case — not a new source of inexactness, cross-validated
against direct Monte Carlo sampling of the graph). The root's moments are the per-job service time
of a Split-Join (blocking) queue, wrapped in the exact M/G/1 (Pollaczek–Khinchine) formula.

```python
from most_queue.theory.fork_join.dag import ForkJoinDAGCalc
from most_queue.random.utils.params import ParetoParams

dag = (
    "series",
    [
        ("parallel", [("leaf", "pareto", ParetoParams(alpha=6.0, K=1.0)), ("leaf", "gamma", [2.0, 5.0, 14.0])]),
        ("parallel", [("leaf", "pareto", ParetoParams(alpha=7.0, K=1.2)), ("leaf", "pareto", ParetoParams(alpha=6.0, K=0.8))]),
    ],
)
calc = ForkJoinDAGCalc(dag)
calc.set_sources(l=0.1)
res = calc.run()   # res.v, res.w -- sojourn/wait moments of the whole DAG
```

**Not covered (reserve):** general (non-series-parallel) precedence DAGs are provably #P-hard to
solve exactly in general (Dodin, Operations Research, 1985) — would need bounding techniques or
Monte Carlo, not an exact/quasi-exact calculator.

### (n,k)-fork-join over heterogeneous branches

**In plain words:** the models above all require *every* branch to finish (n-of-n). Real systems
often only need *some* of them — a replicated read needs only the fastest quorum to respond, a
RAID/erasure-coded write is done once enough shards are written, a MapReduce reduce phase with
speculative execution only waits for enough of the redundant mapper copies. This is a k-out-of-n
join, generalized here to branches that need not be identical.

**Description:** the k-th order statistic (ascending: k=1 the fastest/minimum, k=n the
slowest/maximum, i.e. the models above) of `n` independent branches. `P(X_(k) <= x) = P(at least k
of the n branches have finished by x)` — a standard order-statistics fact, and numerically the same
problem as k-out-of-n:G system reliability with heterogeneous component lifetimes: computed via
the classical Rushdi (1985) O(n²) Poisson-Binomial DP over each branch's CDF at `x`, then the same
tail-moment quadrature as the heterogeneous maximum above (`k=n` reduces to it exactly). For i.i.d.
Pareto branches there's again an exact closed form (`pareto_kth_order_moments`), generalizing
`pareto_max_moments` via the `k`-th order statistic of Uniform(0,1) being `Beta(k, n-k+1)`
distributed — used here as the regression anchor for the heterogeneous numerical path. See
[`docs/research/fork-join-nk-heterogeneous-2026.md`](../research/fork-join-nk-heterogeneous-2026.md).

**Functions:** `pareto_kth_order_moments`, `heterogeneous_kth_order_moments`
(`most_queue.theory.utils.max_dist`)

```python
from most_queue.theory.utils.max_dist import heterogeneous_kth_order_moments
from most_queue.random.utils.params import ParetoParams

branches = [
    ("pareto", ParetoParams(alpha=6.0, K=1.0)),
    ("pareto", ParetoParams(alpha=7.0, K=1.2)),
    ("gamma", [2.0, 6.0, 24.0]),
]
moments = heterogeneous_kth_order_moments(branches, k=2, num=2)  # 2nd of 3 to finish
```

Any `("parallel", ...)` node in a `ForkJoinDAGCalc` DAG can take an optional third element, the
required count `k` (defaults to `len(children)`, i.e. plain fork-join):

```python
dag = ("parallel", [("leaf", "pareto", p1), ("leaf", "pareto", p2), ("leaf", "pareto", p3)], 2)
```
