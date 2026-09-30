# Roadmap: queueing-inventory with general c heterogeneous servers

> Context: `docs/research/queueing-inventory-heterogeneous-servers-general-c-2026.md`.
> Base: EPIC-025 (canonicalize-based mechanical CTMC, `theory/priority/preemptive/mm2_heterogeneous.py`),
> EPIC-028 (`theory/inventory/mmc_inventory.py`, stacked-boundary-superblock QBD),
> EPIC-033 (`theory/inventory/mm2_heterogeneous_inventory.py`, the c=2 special case this generalizes).

## 1. State space

Servers indexed `0..c-1`, ordered by assignment priority (index 0 = first choice when multiple
idle -- "fastest preferred" convention, matches EPIC-033's "server 1 preferred when both idle").

- **Boundary** (`n = 0..c-1` customers in system, all in service, none waiting): phase =
  `(S, i)` where `S` is the subset of busy servers, `|S| = n`, and `i` is the stock level
  (`0..s_max`). Number of subset-phases at level `n` is `C(c, n)`; total boundary phase count is
  `sum_{n=0}^{c-1} C(c,n) * m = (2^c - 1) * m`, `m = s_max + 1`.
- **Repeating** (`n >= c`): all c servers busy (`S = {0,...,c-1}` always, since any freed server
  immediately picks up the next waiting customer -- no idle server can coexist with a nonempty
  queue), phase = stock level `i` only, `m` states per level, aggregate service rate
  `mu_sum = sum(mu_k)`.

Same `m`-phase repeating part as `MMcQueueingInventoryCalc`/`MM2QueueingInventoryHeterogeneousCalc`
-- only the boundary super-block size changes with c.

## 2. Mechanical transition construction (canonicalize-style, per EPIC-025)

Three small pure functions build every transition -- no hand-written case-by-case matrix:

```python
def arrival_target(subset: tuple[int, ...], c: int) -> tuple[int, ...]:
    """Lowest-index idle server joins (deterministic 'fastest/highest-priority idle first')."""
    idle = [k for k in range(c) if k not in subset]
    return tuple(sorted(subset + (min(idle),)))

def departure_targets(subset: tuple[int, ...], mus: list[float]) -> list[tuple[int, float, tuple]]:
    """One (departing_server, rate, resulting_subset) per busy server -- each fires independently."""
    return [(k, mus[k], tuple(s for s in subset if s != k)) for k in subset]

def subset_service_rate(subset: tuple[int, ...], mus: list[float]) -> float:
    """Total outflow rate for the diagonal -- sum of the busy servers' rates."""
    return sum(mus[k] for k in subset)
```

Stock-level sub-structure per subset (arrivals `i -> i` no-op except diagonal, service completion
`i -> i-1` at `i>=1` only [blocked at zero stock, library-wide convention], replenishment `i -> S`
at rate `theta` for `i <= s`) is identical to every other model in this family and reuses
`diag_block`/`down_block`/`arrival_diag` from `bulk_service_erlang.py`'s sibling pattern in
`mm2_heterogeneous_inventory.py` -- just parametrized by `subset_service_rate(subset)` instead of
a single `mu_local`.

**Boundary blocks** (`b00`, `b01`, `b10` for `QBDSolver`), built by iterating every
`(n, subset)` pair via `itertools.combinations(range(c), n)`:
- `b00` diagonal sub-block at `(S,S)`: `diag_block(subset_service_rate(S))`.
- `b00` arrival sub-block `(S, arrival_target(S))` for `n < c-1` (targets stay in the boundary):
  `arrival_diag`.
- `b00` departure sub-blocks `(S, S\{k})` for each `k` in `S`: `down_block(mu_k)` (one sub-block
  per busy server, NOT a single aggregate -- this is the genuinely new piece beyond EPIC-033's
  binary case, generalized from "2 possible departures" to "`|S|` possible departures").
- `b01` (boundary `n=c-1` subsets -> repeating level `n=c`): for each subset `S` with `|S|=c-1`,
  the one idle server joins deterministically -- `arrival_diag` (no combinatorial choice needed,
  since `arrival_target` is forced once only one idle slot remains).
- `b10` (repeating level `n=c` -> boundary `n=c-1`): split by which of the c servers finishes --
  for each `k in range(c)`: `down_block(mu_k)` into boundary subset `full_set \ {k}`. Exact
  generalization of EPIC-033's `b10` split (there: 2 sub-blocks; here: c sub-blocks).

`c=2` reduces this construction exactly to EPIC-033's hand-built `b00`/`b01`/`b10` (subsets `()`,
`(0,)`, `(1,)` map to EPIC-033's "n=0", "config=1", "config=2" respectively) -- the primary
structural regression check, in addition to the numerical `mu_1=...=mu_c` reduction to
`MMcQueueingInventoryCalc`.

## 3. Aggregate metrics (generalizing EPIC-033's per-n formulas)

Because boundary states of the same `n` are stored contiguously (all `C(c,n)` subsets of level
`n` grouped together), per-n aggregation is just a slice-sum, no need to unpack which subset:

- `P(n)` for `n < c`: sum over that level's contiguous slice of `pi0`.
- `E[N]`: `sum_{n=0}^{c-1} n * P(n) + [c * p_ge_c + mean_extra]` (repeating-part terms identical in
  shape to EPIC-033's, just `c` instead of `2`).
- Stock marginal `P(stock=i)`: sum every boundary subset's `i`-th phase (reshape each n-block to
  `(C(c,n), m)` and sum over the subset axis) plus `pi1 @ (I-R)^{-1}`.
- `E[S]` (mean service time actually experienced), the flow-balance shortcut generalizes cleanly:
  `sum_k P(busy_k)` = (number of busy servers, summed over states) = `sum_{n=1}^{c-1} n * P(n,
  i>=1) + c * P(N>=c, i>=1)` -- note this sum does NOT need the per-server breakdown EPIC-033 used
  (it only needed `p_busy1, p_busy2` separately to report symmetric stock breakdowns, which this
  epic does not add); it only needs `n`, which is already known per boundary level. Then
  `E[S] = sum_k P(busy_k) / lambda_eff`, reducing to `1/mu` at `mu_1=...=mu_c=mu` (division by
  `c` cancels against `c` busy servers, same algebra as EPIC-033's `mu1=mu2` case).

## 4. API

```python
class MMcQueueingInventoryHeterogeneousCalc(BaseQueue):
    def __init__(self, c: int, s_max: int, s: int = 0, policy: Policy = "backorder",
                 calc_params: CalcParams | None = None): ...
    def set_sources(self, l: float): ...
    def set_servers(self, mus: list[float], theta: float): ...  # len(mus) == c, priority order
    def run(self, num_levels: int | None = None) -> QueueingInventoryResults: ...
```

`set_servers(mus=[mu, mu], theta=theta)` with `c=2` must exactly match
`MM2QueueingInventoryHeterogeneousCalc.set_servers(mu1=mu, mu2=mu, theta=theta)`.

## 5. DES simulator

`MMcQueueingInventoryHeterogeneousSim` (`most_queue/sim/inventory.py`), generalizing
`MM2QueueingInventoryHeterogeneousSim`'s "race between competing exponentials" event loop: track
an explicit `busy: tuple[int,...]` only while `n < c` (computed as `range(c)` implicitly whenever
`n >= c`, never stored) -- critically, after a departure that leaves `n still >= c`, the freed
server is immediately re-occupied by the next queued customer (same server index stays busy), so
`busy` is only updated when the departure causes `n` to drop *below* `c` for the first time. This
mirrors exactly why the theory repeating part needs no subset tracking.

## 6. Tests

| File | What we check |
|---|---|
| `tests/units/test_mmc_queueing_inventory_heterogeneous.py` (new) | `c=2` matches `MM2QueueingInventoryHeterogeneousCalc` exactly (structural regression); `mu_1=...=mu_c` matches `MMcQueueingInventoryCalc(c=c,...)` for c=3,4,5 (numerical regression); QBD residual negligible; speeding up one server does not increase `E[W]`/stockout; probability vectors valid (`P(n)` sums to 1, stock distribution sums to 1); invalid params rejected (`len(mus) != c`, `c < 1`). |
| `tests/test_mmc_queueing_inventory_heterogeneous_sim.py` (new) | Theory vs. independent DES (`MMcQueueingInventoryHeterogeneousSim`) for c=3 with 3 distinct rates. |

## 7. Documentation

- `docs/models/inventory.md`+`.ru.md` -- new subsection generalizing the existing c=2
  heterogeneous subsection.
- `docs/models.md`/`.ru.md` -- new row.

## 8. Effort

| Stage | Complexity | Days |
|---|---|---|
| Mechanical CTMC construction (theory) | medium (combinatorial bookkeeping, not new math) | 1.0 |
| Aggregate-metrics generalization | low-medium | 0.5 |
| DES simulator | low-medium | 0.5 |
| Tests (structural + numerical regressions, DES cross-check) | medium | 0.75 |
| Documentation | low | 0.25 |
| **Total** | | **3.0** |

---

**Next step:** `most_queue/theory/inventory/mmc_heterogeneous_inventory.py`, class
`MMcQueueingInventoryHeterogeneousCalc` per §2-4.
