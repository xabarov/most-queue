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

### General (Erlang-fitted) batch-service time

**Description:** `BulkServiceMM1Calc` above assumes **exponential** batch-service time. This
closes that reserve for a **general** distribution — fitted to an Erlang(`k`, `rate`) phase-type
representation from its raw moments (the same moment-fitting convention used throughout the
library: `MaxDistribution`, the SLA layer). Erlang naturally represents low-CV (`≤1`) batch-service
times — a realistic regime for GPU/LLM batching, where per-batch processing time is fairly
predictable rather than highly variable. The CTMC state gets an extra phase dimension:
`(batch size i, Erlang phase p, waiting j)`. `k=1` (Erlang collapses to Exponential) reduces
**exactly** to `BulkServiceMM1Calc` — the primary regression check.

**A units bug caught before shipping:** the phase-transition rate was first set to `k*rate`
(intending an aggregate rate across `k` phases), which actually gives total mean `1/rate` instead
of the correct `k/rate` — off by a factor of `k²`. This coincidentally still passed the `k=1`
regression check (`k*rate=rate` when `k=1`) but diverged sharply from an independent DES at `k=3`
— caught before committing to the derivation. See
[`docs/research/bulk-service-general-erlang-2026.md`](../research/bulk-service-general-erlang-2026.md)
for the full account, including why the classical embedded-chain/PGF method (Neuts 1975,
Chaudhry-Templeton) was set aside in favor of this lower-risk, phase-type approach.

**Calculator class:** `BulkServiceErlangCalc` (`most_queue.theory.batch.bulk_service_erlang`)

```python
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc

calc = BulkServiceErlangCalc(a=1, b=4, k=3)   # Erlang(3, rate): CV = 1/sqrt(3)
calc.set_sources(1.0)
calc.set_servers(rate=1.5)                     # mean batch service = k/rate = 2.0
res = calc.run()                                # res.v[0] mean; res.w -- exact moments at a=1 (see below)

# exact raw moments of W (not just the mean), a=1 only:
w_moments = calc.get_w(num=4)                   # [E[W], E[W^2], E[W^3], E[W^4]]

# or fit (k, rate) from raw moments directly:
calc2 = BulkServiceErlangCalc(a=1, b=4, k=1)   # k is overwritten by the fit
calc2.set_sources(1.0)
calc2.set_servers_from_moments([2.0, 4.5])     # [mean, E[S^2]] of batch-service time

# rate may also be a callable rate(batch_size) -- k (phase count) stays fixed,
# only the per-phase rate varies with the size of the batch currently in service:
calc3 = BulkServiceErlangCalc(a=1, b=4, k=3)
calc3.set_sources(1.0)
calc3.set_servers(lambda size: 3.0 + 0.2 * size)  # bigger batches slower per phase (LLM/GPU-style)
```

**Accuracy and scope:** exact given the Erlang-fitted family. `get_w()` gives EXACT raw moments of
`W` at `a=1` (EPIC-043, porting EPIC-032's PASTA/hypoexponential-decomposition technique to this
phase-augmented case — the remaining service of the batch an arrival finds in progress is
`Erlang(k-p, rate)`, `p` = the phase found, plus `j//b` full batches ahead each `Erlang(k, rate(b))`,
convolved); `run()` uses this exact mean for `a=1` and falls back to a busy-time-weighted-average
approximate mean for `a>1` (same `a>1` restriction as `BulkServiceMM1Calc`, see
[`docs/research/bulk-service-waiting-moments-2026.md`](../research/bulk-service-waiting-moments-2026.md)
for why). Not covered (reserve): H2 fitting for CV`≥1` combined with exact moments (EPIC-036's H2
calculator is still mean-only); batch-size-dependent phase COUNT (`k`) — EPIC-042 ported the
exponential model's callable-rate convention to this phase-type case, but `k` itself stays fixed
across batch sizes (a materially harder extension, same class of difficulty as EPIC-041's
per-server phase counts).

### General (H2-fitted) batch-service time (CV ≥ 1)

**Description:** the CV`≥1` complement to the Erlang case above — batch-service time fitted to an
H2(`p1`,`mu1`,`mu2`) (hyperexponential, 2-phase mixture) representation from its raw moments, for
batch-service times that are more variable than exponential. Unlike Erlang (a sequential chain of
phases), H2 is a **branch**: at the moment a batch starts service, one of two exponential phases is
chosen once (phase 0 w.p. `p1`, rate `mu1`; phase 1 w.p. `p2=1-p1`, rate `mu2`) and the batch stays
in that phase until it completes — no mid-service phase transitions. The CTMC state gets an extra
phase dimension: `(batch size i, phase ∈ {0,1}, waiting j)`; every "batch starts" transition splits
into two weighted sub-transitions (`×p1`, `×p2`) instead of Erlang's single deterministic one.
`p1=1` (H2 collapses to `Exp(mu1)`) reduces **exactly** to `BulkServiceMM1Calc` — the primary
regression check, verified to full float64 precision. Unlike the Erlang case, the general
(genuinely two-phase) case matched an independent DES on the first attempt — no bug found, likely
because "choose-once-at-start" branching is structurally simpler to get right than Erlang's
sequential-phase-advance rate convention. See
[`docs/research/bulk-service-general-h2-2026.md`](../research/bulk-service-general-h2-2026.md) for
the full account.

**Calculator class:** `BulkServiceH2Calc` (`most_queue.theory.batch.bulk_service_h2`)

```python
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc

calc = BulkServiceH2Calc(a=1, b=4)
calc.set_sources(1.0)
calc.set_servers(p1=0.5, mu1=1.0, mu2=3.0)     # mean batch service = p1/mu1 + p2/mu2
res = calc.run()                                # res.v[0], res.w[0] -- mean only (see below)

# or fit (p1, mu1, mu2) from raw moments directly:
calc2 = BulkServiceH2Calc(a=1, b=4)
calc2.set_sources(1.0)
calc2.set_servers_from_moments([2.0, 12.0, 100.0])   # [mean, E[S^2], E[S^3]], CV >= 1

# p1/mu1/mu2 may each be a callable f(batch_size) -- the two-branch structure stays
# fixed, only the branch probability/rates vary with the size of the NEW batch starting:
calc3 = BulkServiceH2Calc(a=1, b=4)
calc3.set_sources(1.0)
calc3.set_servers(lambda size: 0.3 + 0.1 * size, mu1=1.5, mu2=3.0)
```

**Accuracy and scope:** exact given the H2-fitted family (mean-only, same starting scope as the
Erlang case above). Not covered (reserve): exact raw moments beyond the mean (needs a PASTA
argument that also tracks which phase an arrival finds the batch in); batch-size-dependent
branch COUNT (the two-branch structure itself stays fixed — EPIC-042 only made the per-branch
probability/rates batch-size-dependent).

### Auto-dispatch (don't compute CV by hand)

**Description:** `fit_bulk_service_calc(a, b, moments, family="auto")`
(`most_queue.theory.batch.bulk_service_general`) picks between the two calculators above
automatically from the CV of the given raw moments — `cv ≤ 1` → `BulkServiceErlangCalc`, `cv > 1`
→ `BulkServiceH2Calc` — mirroring `theory.utils.sla.fit_from_moments`'s `family="auto"` convention.
Returns a calculator with `set_servers()` already applied from the moments; the caller still calls
`set_sources()` then `run()`. Explicit `family="erlang"`/`"h2"` overrides validate CV-feasibility
and raise `ValueError` rather than silently producing a wrong fit.

```python
from most_queue.theory.batch.bulk_service_general import fit_bulk_service_calc

calc = fit_bulk_service_calc(a=1, b=4, moments=[2.0, 4.5])        # cv < 1 -> Erlang
calc = fit_bulk_service_calc(a=1, b=4, moments=[2.0, 12.0, 100.0])  # cv > 1 -> H2 (needs 3 moments)
calc.set_sources(1.0)
res = calc.run()
```

**See also:** [SLA / deadline-violation probability](sla.md) — turn these moments into a deadline-violation probability or SLO quantile.
