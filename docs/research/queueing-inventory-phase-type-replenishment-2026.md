# Queueing-inventory with phase-type (Erlang) replenishment lead time — literature review (2026)

Closes the reserve EPIC-040 was queued for: every `queueing-inventory` model in this library
(EPIC-024/026/027/028/033/038/039) uses `Exp(theta)` replenishment lead time. EPIC-027's own
note explains why this was previously easy: "memoryless lead time means no extra state bit
needed" — an order in transit is fully determined by `stock <= s`, no need to track how long it
has been pending. A phase-type (non-memoryless) lead time breaks that shortcut: Erlang and H2 both
require tracking *how far along* the current order is.

## Literature

Classical foundation already cited across the queueing-inventory epics (Schwarz & Daduna 2006,
Saffari/Haji/Hassanzadeh 2013) already discusses "general lead times" as the natural generalization
beyond exponential — phase-type lead time is the same "fit to a phase-type family, augment the
CTMC" technique this library has used repeatedly (EPIC-035/036/039), just applied to the supply
side instead of the service side. No new literature search was needed beyond what's already cited
in `docs/roadmaps/queueing_inventory_general_sS_roadmap.md`.

## Scope: Erlang first, M/M/1 base case

Following the established "Erlang before H2" sequencing (EPIC-035 before EPIC-036, same
low-risk-first rationale), and the established "base case (M/M/1) before generalizing to
multi-server/heterogeneous" sequencing (EPIC-024 before EPIC-028/033/038/039): this epic covers
`MM1QueueingInventoryErlangReplenishmentCalc` only. H2 replenishment and porting to the
multi-server/heterogeneous calculators are reserve, tracked as follow-up epics.

## Key state-space insight

Unlike every phase-type augmentation so far in this library (which added a phase dimension to
*every* stock/service state), phase-type replenishment only needs a phase dimension for stock
levels `i <= s` (an order is in transit) — levels `i > s` have no pending order and need no phase
at all. The augmented "stock+order" phase space splits into a **plain zone** (`i in {s+1,...,S}`,
`S-s` states, no phase) and a **pending zone** (`i in {0,...,s}`, each with `r` Erlang sub-phases,
`(s+1)*r` states). A service completion crossing from the plain zone's lowest level (`i=s+1`) into
the pending zone (`i=s`) is exactly the moment an order is placed (phase initialized to 0); all
other service completions within the pending zone preserve the current phase (the order's progress
is unaffected by further stock depletion); the only route back to the plain zone is a *replenishment
completion* (last phase finishes), which jumps stock directly to `S` regardless of how low it had
drifted meanwhile. Because none of this touches the arrival/service *level* structure of the QBD
(number of customers `n`), the augmentation is entirely local to each level's phase block
(`diag_block`/`down_block`/`arrival_diag`) — the surrounding `a0`/`a1`/`a2`/`b00`/`b01`/`b10` level
structure is untouched, unlike EPIC-039 where the repeating part itself needed a bigger phase space.

**Bug-shaped trap avoided by validating at a non-degenerate parameter immediately (per the
EPIC-035/038 lesson):** the first numeric prototype held Erlang `rate` fixed while varying `r`,
which (correctly, by the `mean = r/rate` identity) changes the *mean* lead time, not just its CV —
producing an apparently "unstable" system (`pi`'s tail mass not decaying, QBD log-reduction
overflowing) that looked like an implementation bug but was actually a correctly-modeled, genuinely
much-slower-replenishment regime. Re-testing with `rate = r/mean_lead` (holding the mean fixed
across `r`) resolved it immediately and confirmed the construction is correct — a useful reminder
that an "instability" during validation is not automatically a modeling bug; check parameter
correspondence first, exactly as EPIC-035's `k` vs `k*rate` mean confusion.

## Scope

**Core deliverable:** `MM1QueueingInventoryErlangReplenishmentCalc` — `r=1` reduces exactly to
`MM1QueueingInventoryCalc`; validated against an independent DES for `r>1` at matched mean lead
time, both backorder and lost-sales.

**Reserve:** H2 replenishment (CV≥1); porting to `MMcQueueingInventoryCalc` and the heterogeneous-
server calculators (EPIC-038/039) — same local-phase-block technique should port directly since the
augmentation doesn't interact with the server-busy dimension at all, but not attempted here.

Full derivation: `docs/roadmaps/queueing_inventory_phase_type_replenishment_roadmap.md`.
