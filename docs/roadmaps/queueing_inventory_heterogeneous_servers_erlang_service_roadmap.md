# Roadmap: queueing-inventory, c heterogeneous servers with Erlang-fitted service

> Context: `docs/research/queueing-inventory-heterogeneous-servers-erlang-service-2026.md`.
> Base: EPIC-038 (`theory/inventory/mmc_heterogeneous_inventory.py`, subset-tracking),
> EPIC-039 (`theory/inventory/mmc_heterogeneous_h2_inventory.py`, the H2 sibling this
> complements), EPIC-035 (Erlang phase-augmentation for bulk-service).

## 1. State space

Server `k` has its own `ErlangParams(r_k, rate_k)`. Per-server state: idle (`-1`) or busy at
phase `p in {0,...,r_k-1}`. Config = length-`c` tuple.

- **Boundary** (`n=0..c-1`): `prod_k(1+r_k) - prod_k(r_k)` configs (each server contributes either
  "idle" (1 way) or "busy at some phase" (`r_k` ways); summed over all occupied-counts `n<c`).
- **Repeating** (`n>=c`, all servers busy): `prod_k(r_k)`-sized phase space.

## 2. Transitions -- the key correction vs. a naive EPIC-038/039 port

Split each config's outflow into two DIFFERENT rate components, unlike EPIC-038/039's single
`mu_local`:

```python
def departure_rate(cfg):  # servers at their LAST phase -- consumes stock, BLOCKED at i=0
    return sum(rate_of(cfg,k) for k busy where cfg[k] == r_k - 1)

def advance_rate(cfg):    # servers NOT at last phase -- doesn't touch stock, NEVER blocked
    return sum(rate_of(cfg,k) for k busy where cfg[k] < r_k - 1)
```

`diag_block(dep_rate, adv_rate)` (generalizes EPIC-038/039's single-argument version):

```
i == 0:          -(arrival(0) + adv_rate + theta)            [dep_rate excluded -- blocked]
i in 1..s:        -(arrival(i) + dep_rate + adv_rate + theta)
i in s+1..S:       -(arrival(i) + dep_rate + adv_rate)
```

`down_block(dep_rate)` unchanged (i>=1 only, as always). **New**: within-level phase-advance,
added directly to `a1`/`b00`'s off-diagonal for each server `k` not at its last phase:
`block[src, dst] += rate_k * I_m` (identity in stock -- phase advance never touches `i`), where
`dst` = config with `k`'s phase incremented by one, UNCONDITIONAL on `i` (this is exactly the term
a naive port of EPIC-038/039's `diag_block(mu_local)` silently drops, since it would fold
`advance_rate` into the blocked-at-`i=0` `mu_local` instead of keeping it always-active).

Arrival (server starts a fresh service): always phase 0, deterministic (no branch split, unlike
H2) -- single target `config_with(k, 0)`.

Departure (only from LAST-phase completion): `config_with(k, -1)` (boundary, drop) or
`config_with(k, 0)` (repeating, instant refill at phase 0) -- same two-case split as EPIC-038's
departure logic, just gated on "is this server's CURRENT phase its last one".

`r_k=1` for all `k`: every busy server is trivially "at its last phase" (`r_k-1=0`), so
`advance_rate` is always 0 and `departure_rate` collapses to EPIC-038's plain `mu_local` exactly.

## 3. Validation

- **`r_k=1` structural regression**: exact (float64) match to `MMcQueueingInventoryHeterogeneousCalc`.
- **`r_k>1` vs. independent DES, mean rates held meaningful** (not a mean-drift trap like
  EPIC-040 -- here `rate_k` is chosen so `r_k/rate_k` matches a fixed reference mean per server,
  same discipline): matched within Monte Carlo tolerance at `(r_1,r_2)=(2,3)` and `(3,4)`, both
  policies, AFTER fixing the `diag_block` outflow-splitting bug described in the research doc.
  **Caught via this exact non-degenerate check, not the `r_k=1` one** -- reinforces the
  session-wide lesson that a degenerate-case regression alone is insufficient.

## 4. API

```python
class MMcQueueingInventoryHeterogeneousErlangCalc(BaseQueue):
    def __init__(self, c: int, s_max: int, s: int = 0, policy: Policy = "backorder",
                 calc_params: CalcParams | None = None): ...
    def set_sources(self, l: float): ...
    def set_servers(self, servers: list[ErlangParams], theta: float): ...
    def set_servers_from_moments(self, moments_per_server: list[list[float]], theta: float): ...
    def run(self, num_levels: int | None = None) -> QueueingInventoryResults: ...
```

## 5. Tests

| File | What we check |
|---|---|
| `tests/units/test_mmc_queueing_inventory_heterogeneous_erlang.py` (new) | `r_k=1` for all `k` matches `MMcQueueingInventoryHeterogeneousCalc` exactly; `r_k>1` (genuinely non-degenerate, at least one server with `r_k>=2` AND a second with a different `r_k`) -- QBD residual negligible (catches the row-sum bug class directly); valid probability vectors; invalid params rejected. |
| `tests/test_inventory.py` (extended) | Theory vs. independent DES at `c=2`, distinct `r_k`. |

## 6. Documentation

- `docs/models/inventory.md`+`.ru.md` -- new subsection next to the H2 one.
- `docs/models.md`/`.ru.md` -- new row.

## 7. Effort

| Stage | Complexity | Notes |
|---|---|---|
| Prototype + validation | medium-high | Found and fixed the departure/advance-rate split bug before shipping. |
| Production implementation | medium | Direct translation of the validated prototype. |
| Tests | medium | Non-degenerate `r_k` case is mandatory, not optional. |
| Documentation | low | |

---

**Next step:** `most_queue/theory/inventory/mmc_heterogeneous_erlang_inventory.py`, class
`MMcQueueingInventoryHeterogeneousErlangCalc` per §2-4.
