# Queueing-inventory, c heterogeneous servers with H2-fitted (non-exponential) service — literature review (2026)

Direct combination of two already-shipped techniques: EPIC-038's general-c heterogeneous-server
subset-tracking and EPIC-035/036's phase-type-fitted (non-exponential) batch-service time. Picked
up after an explicit user request to make queueing-inventory service times more realistic than
exponential, and to specifically consider H2 (since the general case needs complex-valued H2
fits discussed and rejected below).

## Literature

- "A Survey on the Queueing Inventory Systems with Phase-type Service Distributions" (2016,
  doi:10.1145/3016032.3016033) -- directly on-topic survey, confirms phase-type service in
  queueing-inventory is an established (not novel) modeling direction; this epic is the first
  time this library combines it with per-server heterogeneity (distinct heterogeneous servers,
  each with its own general service-time distribution) rather than a single shared phase-type
  service process.
- "Detailed proof of ergodicity condition for the multi-server retrial queueing system with
  heterogeneous..." (2026, doi:10.67268/1812-5093-2026-34-1-118-124) -- confirms heterogeneous
  multi-server queueing (with retrial) is still an active 2026 research topic, same signal as the
  2024 Scientific Reports paper already cited in EPIC-038.
- "Batch service systems with heterogeneous servers" (2020, doi:10.1007/s11134-020-09654-y,
  Queueing Systems) -- confirms combining heterogeneity with non-exponential/batch service
  structure is a recognized combination in the broader (non-inventory) queueing literature.

As with every epic this session, full text was inaccessible (paywalls); the derivation below is
independent, grounded only in these papers' existence/title/abstract as confirmation the direction
is real and not a dead end, not as a transcribed method.

## Why H2, not Erlang, and not complex-valued H2 (`fit_h2_clx`)

The user specifically asked about H2 with complex parameters. **Complex-valued H2 fits cannot be
used here.** `fit_h2_clx` solves a cubic in raw moments and can return complex `(p1, mu1, mu2)`
outside H2's strictly-real-feasible region -- useful only as an approximate CDF/tail fit for the
SLA layer (`theory.utils.sla`), never as an input to building an actual CTMC generator: a Markov
generator's rates and branch probabilities must be real and non-negative by construction, or the
"chain" is not a valid stochastic process. Every place this library augments a CTMC with a
phase-type distribution (EPIC-035, EPIC-036) uses `fit_h2` (Aliev's method) specifically because
it is real by construction and degenerates gracefully (to a single exponential phase) outside the
H2-feasible region, instead of returning physically meaningless complex rates. This epic follows
the same convention.

**H2 over Erlang for this specific combination, deliberately:** combining phase-type service with
heterogeneous-server subset-tracking (EPIC-038) requires each busy server to carry its own phase
state through the state space. H2's "choose once at service start, fixed until departure" branch
structure (already noted as structurally simpler in EPIC-036's writeup) means a busy server's
phase NEVER changes except at departure -- so the repeating part of the QBD has no within-level
("stay") transitions at all, only between-level ones, exactly mirroring the plain-exponential
EPIC-038 structure with a larger phase dimension. Erlang's sequential-phase-advance would instead
require within-level phase-advance transitions for every busy server independently, a materially
harder QBD to construct correctly (and was exactly the source of EPIC-035's k² bug) -- deferred as
a genuinely separate, harder reserve item.

## Approach: per-server config vector

Generalizes EPIC-038's per-server busy/idle flag to a 3-valued per-server flag: each server slot
is idle (`-1`), or busy having chosen H2 branch 0, or busy having chosen H2 branch 1. A "config"
is a length-`c` tuple over `{-1, 0, 1}`. Boundary phase count (levels `n=0..c-1`) becomes
`3^c - 2^c` (vs. EPIC-038's `2^c - 1`); the repeating part's phase count becomes `2^c` (branch
combinations across all `c` servers, since every server is always busy there) instead of EPIC-038's
single scalar phase -- a genuinely new piece: the repeating part of the QBD is no longer
homogeneous-scalar, it needs its own `2^c`-sized phase space, with departures **splitting** (like
every H2 "batch/service starts" transition in EPIC-036/037) into two weighted sub-transitions
(redraw the departing-then-instantly-refilled server's branch) when the queue stays nonempty, or
collapsing into the boundary (no redraw, server goes idle) when it does not.

Validated numerically before writing any production code (same discipline as every other epic):
(a) `p1_k=1` for all servers (H2 degenerates to `Exp(mu1_k)`) reproduces
`MMcQueueingInventoryHeterogeneousCalc` (EPIC-038) to full float64 precision at `c=2`; (b) the
genuinely heterogeneous two-phase case matches an independent DES at `c=2` and `c=3`, both
`backorder` and `lost_sales` policies, within Monte Carlo tolerance -- no bugs found in the
prototype, consistent with H2's now-repeated track record (EPIC-036, EPIC-037) of the
"choose-once-at-start" structure being easy to get right the first time, versus Erlang/
sequential-phase conventions which have twice produced off-by-constant/off-by-k² bugs
(EPIC-035, EPIC-038's own `_mean_in_system`).

## Scope

**Core deliverable:** `MMcQueueingInventoryHeterogeneousH2Calc(c, s_max, s, policy)` -- each
server has its OWN H2 distribution (`p1_k, mu1_k, mu2_k`, not just a scalar rate), mean queue/
inventory moments, exact QBD (not an approximation, given the assumed H2-per-server family).
`set_servers_from_moments` fits each server's H2 independently from its own raw moments via the
existing `H2Distribution.get_params` (`fit_h2`).

**Reserve, not in this epic:** Erlang-per-server service (needs within-level phase-advance
transitions, materially harder QBD); mixed families (some servers Erlang, some H2); phase-type
replenishment lead time (the cheaper of the two extensions discussed with the user, deferred in
favor of this harder one per explicit request); exact moments beyond the mean.

Full derivation: `docs/roadmaps/queueing_inventory_heterogeneous_servers_h2_service_roadmap.md`.
