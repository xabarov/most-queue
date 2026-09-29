# Multi-server queueing-inventory systems — literature review (2026)

Direct continuation of EPIC-024/026/027 (`docs/research/queueing-inventory-2026.md`,
`queueing-inventory-lost-sales-2026.md`, `queueing-inventory-general-sS-2026.md`): the single-server
`M/M/1` queueing-inventory model (`(s,S)` policy, backorder or lost sales, exact QBD) generalized to
`c > 1` identical servers, each service still consuming one shared stock unit.

## Key literature

- **Yue D., Zhao G., Yue W.**, *Analysis of a multi-server queueing-inventory system with
  non-homogeneous Poisson arrivals*, Proc. 11th Int. Conf. on Queueing Theory and Network
  Applications, 2016, doi:10.1145/3016032.3016037. Continuous-review `(s,S)` policy, exponential
  lead time, formulated directly as a QBD process; stability conditions and the joint
  queue-length/inventory-level stationary distribution obtained via the matrix-geometric method —
  **exact**, not an approximation. Confirms the model family stays QBD-solvable with `c` servers.
- **Krishnamoorthy A., Manikandan R., Dhanya B.**, *Analysis of a Multiserver Queueing-Inventory
  System*, Advances in Operations Research, 2015, doi:10.1155/2015/747328. Derives the exact
  steady-state distribution of the `M/M/c` queueing-inventory system with positive service time;
  works out `c=2` homogeneous servers in full detail as the base case before generalizing.
- **Krishnamoorthy A. et al.**, *The multi server M/M/(s,S) queueing inventory system*, Annals of
  Operations Research, 2013, doi:10.1007/s10479-013-1405-5 (34 cites). **Different model** —
  servers themselves *are* the inventory (a server is "consumed"/removed when its customer
  finishes and replenished via `(s,S)`), not the shared-stock-per-service model this epic
  implements. Cited for completeness/disambiguation only; not used as a direct reference for the
  block derivation below.
- Survey: **Salini K., Arya P.S., Manikandan R.**, *Queueing-Inventory Systems: A Survey*, arXiv
  2308.06518, 2023 — section 2.3 "Multi-Server QIS" confirms the Yue et al. QBD/matrix-geometric
  treatment is the standard exact approach for this model family (2016–2023 literature); more
  recent extensions (vacations, retrial, MAP arrivals, heterogeneous customers) stack additional
  structure on top of the same base, out of scope here.

## State-space structure (the key design question)

For `c=1` (already implemented), level = number of customers `n` (unbounded), phase = stock level
`i ∈ {0,...,S}`, and the boundary is exactly level `n=0` (dimension `S+1`) — `QBDSolver`'s
single-boundary-level layout handles it directly.

For `c>1`, the number of *busy* servers is `min(n, c)`, so the state does **not** need an extra
dimension to track "servers busy" explicitly — it is a deterministic function of `n` alone, exactly
as in the ordinary `M/M/c` queue. What changes is that departure rate depends on `n` for
`n = 0, ..., c-1` (rate `n·μ`, not yet capacity-saturated) and only becomes level-independent
(`c·μ`) once `n ≥ c`. That means there are `c` distinct "boundary" levels, not just one, before the
repeating (homogeneous) part of the QBD kicks in.

`QBDSolver` supports a boundary block of *any* dimension `m0` different from the repeating
dimension `m`, as long as it is a single block — so the fix is to **stack the `c` boundary levels
into one super-block** of dimension `m0 = c·(S+1)` (phases = stock level `i`, nested inside
customer-count sub-level `n = 0,...,c-1`), and let the repeating region (`n ≥ c`, dimension
`S+1`) start only once all `c` servers are saturated. This is a standard QBD-folding trick, not
something the reviewed papers spell out mechanically, but it is exactly what "formulated as a QBD
process" implies in Yue et al. 2016. Setting `c=1` collapses the super-block to a single level and
must reduce exactly to the already-implemented `(0,S)`/general-`(s,S)` model — a strong regression
check for the implementation (mirrors the `s=0 → (0,S)` reduction check from EPIC-027).

Full block derivation: `docs/roadmaps/queueing_inventory_multiserver_roadmap.md`.

## Recommendation

Implementable **exactly** (matrix-geometric QBD, no approximation), reusing the existing
`QBDSolver` unchanged — only the block-construction logic in a new calculator class is new. Combine
with the already-generalized `s` (reorder point) and `policy` (backorder/lost-sales) parameters
from EPIC-026/027 rather than resetting to the `(0,S)`-only special case, since the block
derivation carries both through with no extra cost. Main implementation risk (per the "what's next"
tradeoff discussion): `get_p()`/mean-metric helpers need custom boundary-block-splitting logic
(the existing `_phase_marginal`/`_mean_in_system` helpers assume a single boundary level and must
be adapted), not just a parameter threaded through unchanged formulas as in EPIC-026/027.
