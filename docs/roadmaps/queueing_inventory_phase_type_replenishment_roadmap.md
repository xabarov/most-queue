# Roadmap: queueing-inventory with Erlang-fitted replenishment lead time

> Context: `docs/research/queueing-inventory-phase-type-replenishment-2026.md`.
> Base: `theory/inventory/mm1_inventory.py` (the Exp(theta) model this extends).

## 1. State space

`i` = stock level, `0..S`. `s` = reorder point. Servers/customers unaffected -- this
augmentation is purely on the stock/replenishment axis.

- **Plain zone** (`i in {s+1,...,S}`, no order pending): `m_plain = S - s` states.
- **Pending zone** (`i in {0,...,s}`, order in transit, Erlang phase `p in {0,...,r-1}`):
  `m_pending = (s+1) * r` states.
- Total phase count per QBD level: `M = m_plain + m_pending`.

Indexing: `plain_idx(i) = i - (s+1)` for `i in s+1..S`; `pending_idx(i,p) = m_plain + i*r + p`
for `i in 0..s`, `p in 0..r-1`.

## 2. Transitions (all local to one QBD level's phase block)

```
Arrival (rate lam, or 0 if lost_sales and i==0): phase unchanged, level n -> n+1.

Service completion (rate mu, blocked at i=0):
  plain i, i > s+1:        plain(i) -> plain(i-1)                  [level n -> n-1]
  plain i == s+1:           plain(s+1) -> pending(s, phase=0)        [ORDER PLACED]
  pending (i,p), i >= 1:    pending(i,p) -> pending(i-1,p)           [phase preserved]
  pending (0,p):            blocked (no transition)

Replenishment phase-advance (rate = erlang_rate, per-phase, same for every phase):
  pending (i,p), p < r-1:   pending(i,p) -> pending(i,p+1)           [same level, same i]
  pending (i,r-1):          pending(i,r-1) -> plain(S)                [ORDER COMPLETES -> restock]
```

`r=1`: every pending state immediately routes via the "p=r-1" branch, reproducing the original
`Exp(theta)`-style "jump to S from every i<=s" transition exactly -- `theta` <-> `erlang_rate`.

## 3. Validation

- **`r=1` structural regression**: reduces exactly (float64 precision) to `MM1QueueingInventoryCalc`
  with `theta = erlang_rate`. Verified for both `backorder` and `lost_sales`.
- **`r>1` vs. independent DES**: matched at `r=2,3,5`, **mean lead time held fixed**
  (`rate = r / mean_lead`) -- an earlier attempt that held `rate` fixed while varying `r` produced
  an apparently unstable/divergent QBD (log-reduction overflow, non-decaying tail mass in a
  truncated cross-check), which turned out to be a genuinely different (much slower) mean lead
  time, not a bug -- see research doc.

## 4. API

```python
class MM1QueueingInventoryErlangReplenishmentCalc(BaseQueue):
    def __init__(self, s_max: int, s: int = 0, policy: Policy = "backorder",
                 calc_params: CalcParams | None = None): ...
    def set_sources(self, l: float): ...
    def set_servers(self, mu: float, r: int, rate: float): ...  # mean lead time = r/rate
    def set_replenishment_from_moments(self, mu: float, moments: list[float]): ...  # fits (r,rate) via ErlangDistribution.get_params
    def run(self, num_levels: int | None = None) -> QueueingInventoryResults: ...
```

## 5. Tests

| File | What we check |
|---|---|
| `tests/units/test_mm1_queueing_inventory_erlang_replenishment.py` (new) | `r=1` exact regression to `MM1QueueingInventoryCalc`; `r>1` at fixed mean lead time vs. independent DES (backorder + lost_sales); QBD residual; valid probability vectors; invalid params rejected. |

## 6. Documentation

- `docs/models/inventory.md`+`.ru.md` -- new subsection.
- `docs/models.md`/`.ru.md` -- new row.

## 7. Reserve

- H2 replenishment (CV>=1).
- Port to `MMcQueueingInventoryCalc`/heterogeneous-server calculators (EPIC-038/039) -- the
  phase-block augmentation is local and doesn't touch the server-busy dimension, so should port
  directly.

---

**Next step:** `most_queue/theory/inventory/mm1_inventory_erlang_replenishment.py`, class
`MM1QueueingInventoryErlangReplenishmentCalc` per §2-4, translating the validated prototype.
