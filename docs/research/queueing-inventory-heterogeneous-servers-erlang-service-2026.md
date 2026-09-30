# Queueing-inventory, c heterogeneous servers with Erlang-fitted service — literature review (2026)

Complement to EPIC-039 (H2, CV≥1): each of the `c` heterogeneous servers gets its own
Erlang(`r_k`, `rate_k`) (CV≤1) service-time distribution instead of H2. Closes the reserve
EPIC-039 flagged ("Erlang-per-server service ... needs within-level phase-advance transitions ...
the exact shape of problem that caused EPIC-035's k² bug").

## Literature

Same classical foundation as every other phase-type queueing-inventory epic this session
(EPIC-035/036/039/040) -- no new literature search needed; purely a mechanical combination of two
already-validated techniques (EPIC-038's heterogeneous-server subset-tracking, Erlang
phase-augmentation as in EPIC-035).

## Why this is genuinely harder than EPIC-039 (H2) -- and where the risk actually landed

H2's defining property (a server's branch is chosen once at service start and never changes until
departure) means the repeating part of the QBD has *no* transitions within a level -- every
transition either goes up (arrival) or down (departure) a level. Erlang's sequential-phase-advance
breaks that: a busy server's phase can advance (`p -> p+1`) *without* a departure, which is a
same-level ("stay") transition. This means, for the first time in this session's inventory epics,
the repeating part's `a1` block needs real off-diagonal entries beyond the diagonal decay term.

**The actual bug this produced (caught via the `r>1` DES cross-check, not the `r=1` degenerate
check):** the existing `diag_block` helper (reused unmodified from EPIC-038/039) assumes a single
undifferentiated "service rate" that is *entirely* blocked at stock `i=0` (the library-wide "no
unit to consume, no service" convention). For Erlang, that convention is only correct for the rate
component belonging to a server's *last* phase (whose completion actually consumes a stock unit).
A server's *intermediate* phase advances (`p < r_k - 1`) do not consume stock and must **not** be
blocked at `i=0` -- but the reused `diag_block` blocked the server's *entire* rate (both
intermediate and last-phase components lumped into one "mu_local" number) whenever `i=0`,
silently dropping outflow from the generator's diagonal. The resulting matrix failed a basic
row-sum-to-zero check (confirmed directly: several rows summed to `+4.8`/`+2.4` instead of `~0`)
and made the QBD log-reduction diverge (overflow) for any parametrization with `r_k > 1` -- while
the `r_k = 1` degenerate case (where "last phase" and "the whole server" are the same thing by
construction) passed by coincidence, exactly the `k=1`-hides-the-bug pattern from EPIC-035 and
EPIC-038. **Fix:** split the per-config outflow into `departure_rate` (sum of rates of servers at
their *last* phase -- blocked at `i=0`, feeds `down_block`) and `advance_rate` (sum of rates of
servers *not* at their last phase -- never blocked, feeds the new within-level `a1`/`b00`
off-diagonal `rate * I_m` term). Re-verified both `r_k=1` (still exact) and `r_k>1` (now matches
an independent DES) after the fix.

## State space

Per-server state: idle (`-1`) or busy at Erlang phase `p in {0,...,r_k-1}` (heterogeneous `r_k`
per server, unlike EPIC-038/039 where every server shared the same family/CV class). A "config" is
a length-`c` tuple. Boundary phase count (occupied count `n = 0..c-1`): `prod_k(1+r_k) -
prod_k(r_k)`. Repeating part (`n>=c`, all servers busy): `prod_k(r_k)`-sized phase space --
non-scalar, like EPIC-039, but now WITH within-level phase-advance transitions EPIC-039 never
needed.

## Scope

**Core deliverable:** `MMcQueueingInventoryHeterogeneousErlangCalc` -- each server has its own
`ErlangParams(r_k, rate_k)`; `r_k=1` for all `k` reduces exactly to
`MMcQueueingInventoryHeterogeneousCalc` (EPIC-038); mean-only (`E[V]`, `E[W]`), same scope as every
other model in this family.

**Reserve:** mixed Erlang/H2 families per server; exact moments beyond the mean.

Full derivation: `docs/roadmaps/queueing_inventory_heterogeneous_servers_erlang_service_roadmap.md`.
