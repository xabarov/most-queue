# Queueing-inventory with heterogeneous servers — literature review (2026)

Combines two already-closed epics: EPIC-023/025 (state-splitting for two heterogeneous exponential
servers, Krishnamoorthi 1963 technique) and EPIC-028 (`MMcQueueingInventoryCalc`, `c` *identical*
servers sharing a stock pool, stacked-boundary QBD). The gap: `c=2` servers with *different* rates
`mu1 != mu2` sharing one stock pool.

## Literature

- Krishnamoorthy B., **On Poisson Queue with Two Heterogeneous Servers**, Operations Research,
  1963, doi:10.1287/opre.11.3.321 — the foundational state-splitting technique, already reused in
  EPIC-023 (machine repair) and EPIC-025 (priority + heterogeneous servers).
- **Modeling of Junior Servers Approaching a Senior Server in the Retrial Queueing-Inventory
  System**, Mathematics (MDPI), 2023, doi:10.3390/math11224581 (5 cites), published follow-on
  **Analysis of junior servers approaching a senior server in the multi-server queueing-inventory
  system**, Scientific Reports, 2025, doi:10.1038/s41598-025-99748-5 (3 cites) — direct precedent:
  heterogeneous ("junior"/"senior", i.e. differently-rated) servers sharing inventory, confirming
  this is a small but real, active niche (not a solved textbook case, low citation counts because
  the sub-area itself is small, not because it's stale — both papers are 2023/2025).
- **A finite source retrial queueing inventory system with stock dependent arrival and
  heterogeneous servers**, Scientific Reports, 2024, doi:10.1038/s41598-024-81593-7 (6 cites) —
  same server-heterogeneity idea combined with retrial + finite-source arrivals (a different,
  more complex combination than what's targeted here).
- **Performance Analysis of Synergistic Servers in a Multi-Server Queueing-Inventory System with
  Encouraged...**, Journal of the Indian Society for Probability and Statistics, 2025,
  doi:10.1007/s41096-025-00255-7 — further confirms continued 2025 activity in this specific
  sub-niche.

The cited papers were not accessible in full (paywalled/blocked), so the block derivation below is
independent, not transcribed from them — grounded instead in directly recombining this repo's own
already-validated techniques (EPIC-023/025's state-splitting, EPIC-028's stacked-boundary QBD).

## Scope decision: c=2 only

State-space growth from "who exactly is busy" (not just "how many") is combinatorial in `c`: for
`c=2` only one boundary level (`n=1`) needs a disambiguating "which server" bit (2 configurations);
for general `c` every level `n=1,...,c-1` would need to track the specific *subset* of busy
servers (up to `C(c, n)` configurations per level), a much bigger design/implementation task with
higher error risk for limited practical benefit (real systems rarely have more than 2-3
meaningfully different server speeds). `c=2` is scoped here as the practical case, matching
EPIC-023/025's own `c=2` scoping decision; general `c` is left as a reserve item.

## Block derivation sketch (verified against DES cross-validation, not the papers above)

State: `(n, config, i)` — `n` = customers in system, `i` = stock level, `config ∈ {1,2}` labels
*which* server is the sole busy one, needed only when `n=1` (for `n=0` nobody is busy; for `n>=2`
both servers are occupied, unambiguous). Same blocking convention as the whole queueing-inventory
family (EPIC-024 onward): service is blocked entirely at `i=0`, consuming one stock unit *at
completion*, not reservation at start.

- Level `n=0`: phase = stock `i` only (`m = S+1` states).
- Level `n=1`: phase = `(config, i)` (`2m` states) — an arrival is assigned to whichever server was
  idle (tie-break convention: prefer server 1; label the faster server as server 1 if a specific
  preference matters for your use case).
- Level `n>=2`: phase = stock `i` only (`m` states again — both servers always occupied) — this is
  the QBD's homogeneous repeating part, `A0/A1/A2` exactly like `MMcQueueingInventoryCalc` with
  `c*mu -> mu1+mu2`.
- The one genuinely new piece: the departure transition from level `n=2` down to level `n=1` must
  **split** by which server fired — rate `mu1` leaves server 2 as the sole survivor (`config=2`),
  rate `mu2` leaves server 1 (`config=1`) — the same "disambiguate on departure" step EPIC-023/025
  already use, now feeding into a QBD `B10` block instead of a plain CTMC transition list.

`c=2`, `mu1=mu2` must reduce exactly to `MMcQueueingInventoryCalc(c=2, ...)` (EPIC-028) — the
primary regression anchor, in the same spirit as every previous reduction check this session.

Full derivation: `docs/roadmaps/queueing_inventory_heterogeneous_servers_roadmap.md`.
