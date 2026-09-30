# Roadmap: queueing-inventory, c heterogeneous servers with H2-fitted service

> Context: `docs/research/queueing-inventory-heterogeneous-servers-h2-service-2026.md`.
> Base: EPIC-036 (`theory/batch/bulk_service_h2.py`, H2 branch-choice phase-type technique),
> EPIC-038 (`theory/inventory/mmc_heterogeneous_inventory.py`, the plain-exponential general-c
> heterogeneous model this extends).

## 1. State space

Each server `k` has its own H2 distribution `(p1_k, mu1_k, mu2_k)`. Per-server state: idle (`-1`),
or busy having chosen branch 0 (rate `mu1_k`) or branch 1 (rate `mu2_k`) at the start of its
CURRENT service -- fixed until that customer departs (H2's defining "choose once" property).

A **config** is a length-`c` tuple over `{-1, 0, 1}`.

- **Boundary** (`n = 0..c-1`): configs with exactly `n` non-`-1` entries, `(i)` stock. Number of
  such configs: `C(c,n) * 2^n` (choose which `n` servers are busy, times 2 branch choices each);
  total boundary phase count `sum_{n=0}^{c-1} C(c,n)*2^n = 3^c - 2^c` (binomial theorem,
  `sum_{n=0}^{c} C(c,n)*2^n = 3^c`, minus the `n=c` term `2^c`).
- **Repeating** (`n >= c`): configs with ALL `c` entries busy (`2^c` branch combinations), `(i)`
  stock. Unlike EPIC-038 (scalar phase, since exponential rate didn't depend on any persistent
  choice), the repeating part here needs a genuine `2^c`-sized phase space, because which branch
  each server is currently running determines its completion rate.

## 2. Mechanical transition construction

Reuses EPIC-038's canonicalize-style pure functions, extended with the branch dimension:

```python
def rate_of(config, k, servers):
    p1, mu1, mu2 = servers[k]
    return mu1 if config[k] == 0 else mu2

def arrival_slot(config, c):
    """Lowest-index idle server -- the one that will start service (branch decided by caller)."""
    return min(k for k in range(c) if config[k] == -1)

def config_with(config, k, value):
    cfg = list(config); cfg[k] = value; return tuple(cfg)
```

**Boundary blocks:**
- Diagonal (`b00`): `diag_block(sum(rate_of(config,k) for k busy))`, same as EPIC-038.
- Arrival (`n < c-1`, stays boundary): `arrival_slot` finds the joining server `j`; SPLITS into
  two weighted sub-transitions -- `config_with(config,j,0)` at rate `lam*p1_j`,
  `config_with(config,j,1)` at rate `lam*(1-p1_j)` (same "batch/service starts -> split by H2
  branch" pattern as EPIC-036).
- Arrival (`n = c-1`, crosses into repeating level 0): same split, target is a repeating-phase
  config (`b01` block).
- Departure (any busy `k`): SINGLE target `config_with(config,k,-1)` (no branch redraw -- the
  server goes idle, no refill, since boundary means no queue) at rate `rate_of(config,k)`.

**Repeating blocks** (phase space = `2^c` configs, all entries busy):
- `a0` (arrival, level `j -> j+1`): SAME config (arrival doesn't touch any server), `arrival_diag`.
- `a1` (diagonal): `diag_block(sum of all c servers' current rates)` -- no within-level
  transitions exist (H2's key simplifying property: a busy server's branch never changes except
  at departure), so `a1` off-diagonal is all zero -- much simpler than an Erlang-service version
  would be.
- `a2` (departure, level `j -> j-1`, `j >= 1`, queue still nonempty after the departure): for each
  busy `k`, SPLITS into two weighted sub-transitions -- the departing server `k` is immediately
  refilled from the queue with a FRESH branch draw: `config_with(config,k,0)` at rate
  `rate_of(config,k)*p1_k`, `config_with(config,k,1)` at rate `rate_of(config,k)*(1-p1_k)`.
- `b10` (departure from repeating level `j=0`, i.e. `n=c`, into boundary `n=c-1`): for each busy
  `k`, SINGLE target `config_with(config,k,-1)` (no redraw -- queue is now empty) at rate
  `rate_of(config,k)`.

`p1_k = 1` for all `k` (H2 degenerates to `Exp(mu1_k)`) must reduce exactly to
`MMcQueueingInventoryHeterogeneousCalc` (EPIC-038) -- the primary regression anchor. Validated
numerically to full float64 precision before writing any production code.

## 3. Aggregate metrics

Same per-n-level aggregation as EPIC-038 for `E[N]`/`P(n)` (boundary configs grouped contiguously
by `n`, same fix for the `(c-1)*p_ge_c + mean_extra` identity EPIC-038 already derived and
validated -- this part of the derivation is untouched by adding branches, since it only depends on
occupied-count `n`, not on WHICH branch). `E[S]` via the same flow-balance shortcut
(`sum_{n=1}^{c-1} n*P_n_active + c*p_ge_c_active) / lambda_eff`) -- also untouched, since "server k
busy" doesn't care which branch it's in.

## 4. API

```python
class MMcQueueingInventoryHeterogeneousH2Calc(BaseQueue):
    def __init__(self, c: int, s_max: int, s: int = 0, policy: Policy = "backorder",
                 calc_params: CalcParams | None = None): ...
    def set_sources(self, l: float): ...
    def set_servers(self, servers: list[H2Params], theta: float): ...  # len(servers) == c
    def set_servers_from_moments(self, moments_per_server: list[list[float]], theta: float): ...
    def run(self, num_levels: int | None = None) -> QueueingInventoryResults: ...
```

`set_servers_from_moments` fits each server's `(p1_k, mu1_k, mu2_k)` independently from its own
raw moments via `H2Distribution.get_params` (`fit_h2`, Aliev's method -- real by construction,
never `fit_h2_clx`, per the established convention -- see research doc for why complex-valued H2
parameters cannot be used to build a CTMC).

## 5. DES simulator

`MMcQueueingInventoryHeterogeneousH2Sim` (`most_queue/sim/inventory.py`): unlike EPIC-038's
simulator (which only tracked `busy` explicitly while `n < c`), this one must track each busy
server's CURRENT branch always (even deep in the repeating region), since completion rates depend
on it. State: `config: list[int]` of length `c` over `{-1,0,1}`, updated on every arrival
(branch drawn for the newly-busy server) and departure (server goes idle if `n<c` after the
departure, else immediately redraws a fresh branch).

## 6. Tests

| File | What we check |
|---|---|
| `tests/units/test_mmc_queueing_inventory_heterogeneous_h2.py` (new) | `p1_k=1` for all `k` matches `MMcQueueingInventoryHeterogeneousCalc` exactly (structural regression) at c=2,3; QBD residual negligible; probability vectors valid; invalid params rejected. |
| `tests/test_inventory.py` (extended) | Theory vs. independent DES at c=2 and c=3, genuinely heterogeneous H2 per server, both policies. |

## 7. Documentation

- `docs/models/inventory.md`+`.ru.md` -- new subsection after the general-c heterogeneous
  subsection, explicitly addressing why complex-valued H2 (`fit_h2_clx`) is not used.
- `docs/models.md`/`.ru.md` -- new row.

## 8. Effort

| Stage | Complexity | Days |
|---|---|---|
| Numeric prototype + validation (done) | medium | 0.5 |
| Production implementation | medium-high (bigger QBD blocks, branch-split bookkeeping) | 1.25 |
| DES simulator | medium | 0.5 |
| Tests | medium | 0.75 |
| Documentation | low | 0.25 |
| **Total** | | **3.25** |

---

**Next step:** `most_queue/theory/inventory/mmc_heterogeneous_h2_inventory.py`, class
`MMcQueueingInventoryHeterogeneousH2Calc` per §2-4, translating the already-validated prototype.
