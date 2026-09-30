# Queueing-inventory with general c heterogeneous servers — literature review (2026)

Direct generalization of EPIC-033 (`MM2QueueingInventoryHeterogeneousCalc`, exactly c=2
heterogeneous servers) to an arbitrary number of heterogeneous servers c — the reserve item
flagged in EPIC-033's own roadmap and deferred through EPIC-036/037 in favor of faster wins.

## Literature

Multi-server heterogeneous queueing-inventory remains an active research area:

- Krishnan, K. & Elango, C., *Two server Markovian inventory systems with server interruptions:
  Heterogeneous vs. homogeneous servers*, Mathematics and Computers in Simulation, 2018,
  doi:10.1016/j.matcom.2018.03.001 -- the direct c=2 precedent already used to ground EPIC-033;
  explicitly frames heterogeneous vs. homogeneous servers as the interesting comparison, same
  framing this epic extends to general c.
- A 2024 *Scientific Reports* paper, "A finite source retrial queueing inventory system with stock
  dependent arrival and heterogeneous servers", doi:10.1038/s41598-024-81593-7 -- confirms
  multi-server heterogeneous queueing-inventory (explicitly "joint probability distribution of
  number of inventory and busy servers") is a live 2024-2025 research topic; that paper adds
  finite-source and retrial on top, which is out of scope here, but the core "how many/which
  servers are busy interacts with stock level" modeling question is the same one this epic answers
  exactly for the classical (infinite-source, no retrial) case.
- "Modeling of Junior Servers Approaching a Senior Server in the Retrial Queueing-Inventory
  System" (2023 preprint, doi:10.20944/preprints202310.1176.v1) -- another 2023-2024 signal that
  heterogeneous-server assignment policy (who gets picked first) in queueing-inventory is an
  active modeling question; motivates this epic's explicit "fastest/highest-priority idle server
  first" assignment convention (same tie-break convention EPIC-033 already established for c=2).

Full-text of these journal articles was not accessible (MDPI/Nature paywalls, as with essentially
every journal source in this session); grounding is from title/abstract/DOI metadata only, same as
every other epic this session -- the actual derivation below is independent, not transcribed from
any of these papers.

## Why this was deferred three epics running (EPIC-035/036/037 picked instead)

State-splitting for HETEROGENEOUS servers (as opposed to identical servers, where only the *count*
of busy servers matters) requires tracking *which* servers are busy, not just how many. For c=2
(EPIC-033) this is a binary disambiguation (server 1 busy alone, or server 2 busy alone) --
cheap. For general c, the number of distinct "busy configurations" at boundary level n (n < c
customers in service) is `C(c, n)` (choose which n of the c servers are the busy ones), so the
total boundary phase space is `sum_{n=0}^{c-1} C(c,n) = 2^c - 1` times the stock-level count `m`.
This is genuinely exponential in c -- the "combinatorics blow up past c=2" risk flagged in
EPIC-033's own memory note. Tractable for realistic c (say up to 6-8, `2^c` in the tens to low
hundreds), which covers essentially every real queueing-inventory deployment (very few systems
have dozens of distinguishable heterogeneous servers); the implementation below is built to scale
to whatever c the state space can hold in memory, not hand-capped to a specific small c.

## Approach

Same combination-of-proven-techniques strategy as every other epic in this "general N" family:
1. The canonicalize-based **mechanical CTMC construction** pattern from EPIC-025 (build the
   transition structure from small pure functions operating on a busy-server subset, rather than
   hand-deriving a transition table) -- directly solves the "too many branches to safely hand-
   derive" problem this epic would otherwise hit for c>=3.
2. The **stacked-boundary-superblock QBD trick** from EPIC-028/033 (stack all boundary levels
   n=0..c-1 into one super-block for `QBDSolver`; repeating part starts at n=c, homogeneous with
   aggregate rate `sum(mu_k)`, exactly as in the identical-server case).

`c=2` must reduce exactly (same boundary-block shapes, same transition rates) to
`MM2QueueingInventoryHeterogeneousCalc` -- the primary regression anchor, in addition to the
`mu_1=...=mu_c` reduction to `MMcQueueingInventoryCalc` that EPIC-033 already established for c=2
and this epic extends to general c.

## Scope

**Core deliverable:** `MMcQueueingInventoryHeterogeneousCalc(c, s_max, s, policy)` -- exact QBD,
mean queue/inventory moments, `(s,S)` replenishment, backorder or lost-sales, same accuracy level
as every other calculator in this family (exact given the CTMC, mean-only for waiting/sojourn
time -- V=W+S is still exact via the flow-balance `E[S]` shortcut EPIC-033 derived, generalized
here to `sum_k P(busy_k)` over an arbitrary subset structure rather than two named servers).

**Reserve, not in this epic:** retrial (the 2024 Scientific Reports paper's extension);
finite-source; non-exponential heterogeneous service times (would need combining this state-
splitting with the Erlang/H2 phase-augmentation technique from EPIC-035/036 -- a genuinely new
combination, not attempted here); priority classes layered on top of heterogeneous servers for
c>2 (EPIC-025 only covers c=2).

Full derivation: `docs/roadmaps/queueing_inventory_heterogeneous_servers_general_c_roadmap.md`.
