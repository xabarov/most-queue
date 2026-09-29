# Fork-Join with heterogeneous branches and series-parallel task DAGs — literature review (2026)

Direct continuation of EPIC-022 (`docs/research/fork-join-heavy-tail-2026.md`): that epic made the
max-of-n computation *exact* for Pareto branches and *numerically exact given the assumed family*
otherwise, but every branch was assumed **i.i.d.** — same distribution, same parameters. This epic
removes that restriction in two independent directions: (1) `n` parallel branches with
**independent but not identically distributed** service times, and (2) a **task graph** (DAG) of
series/parallel composition, not just one flat fork→n→join level.

## Current state in most_queue

- `SplitJoinCalc`/`MaxDistribution` (`theory/fork_join/split_join.py`, `theory/utils/max_dist.py`):
  flat `n`-way fork-join, all branches **i.i.d.**; exact for Pareto (Beta-function moments,
  EPIC-022), fit-based (H2/Gamma/Erlang moment-matching + quadrature) otherwise.
- `ForkJoinSim` (`sim/fork_join.py`): flat `(n,k)` fork-join DES, technically accepts a different
  Kendall spec per channel already (`set_servers` takes a list) — heterogeneity was never a DES
  limitation, only a theory-side one.
- No DAG/task-graph structure anywhere in the fork-join stack — every model is exactly two levels
  (split, then join).

## Literature

- **Heterogeneous order statistics (the "max of non-identical branches" building block):**
  David H.A., Nagaraja H.N. and earlier — recurrence relations for order statistics of independent,
  non-identically distributed (INID) random variables: Balasubramanian K. & Beg M.I., *Recurrence
  relations for order statistics from n independent and non-identically distributed random
  variables*, Annals of the Institute of Statistical Mathematics, 1988,
  doi:10.1007/bf00052344; Balasubramanian K. et al., *Recurrence relations among moments of order
  statistics from two related sets of INID random variables*, 1989, doi:10.1007/bf00049399. These
  confirm the moments of a maximum of independent-but-different random variables are computable via
  recursion/integration of the product of survival functions — no closed form in general (unlike
  the i.i.d. Pareto case), but **numerically exact given the assumed per-branch family**, exactly
  the same standard this library already applies to the non-Pareto i.i.d. case.
- **Heterogeneous fork-join specifically:** *Overcome Heterogeneity Impact in Modeled Fork-Join
  Queuing Networks for Tail Prediction*, 2019 IEEE ICCNC, doi:10.1109/iccnc.2019.8685575 —
  confirms this is a recognized, still-open practical gap (heterogeneous nodes in real clusters),
  not a solved textbook case.
- **Series-parallel task graphs / stochastic PERT networks:** Dodin B., *Bounding the Project
  Completion Time Distribution in PERT Networks*, Operations Research, 1985,
  doi:10.1287/opre.33.4.862 (132 cites) — establishes that **general** PERT/precedence networks
  need *bounds*, not exact distributions (this is the well-known #P-hardness of general stochastic
  activity networks); Al-Ghanim A., *Reliability Analysis of a Flow Network with a
  Series-Parallel-Reducible Structure*, IEEE Trans. Reliability, 2016, doi:10.1109/tr.2015.2499962
  — confirms **series-parallel-reducible** graphs (built by repeatedly composing sub-graphs in
  series or in parallel — covers the overwhelming majority of real task graphs: MapReduce stages,
  microservice call chains, CI/CD pipelines) *do* reduce exactly via recursive
  series(=convolution)/parallel(=max) composition, with no combinatorial blowup.
- Book-length treatment: Rizk A., Poloczek F. et al. (eds.), *Analysis of Fork-Join Systems*,
  CRC Press, 2022, doi:10.1201/9781003150077 — chapters on relaxed/general fork-join networks;
  used here only to confirm the field's own boundary between exactly-tractable (series-parallel)
  and generally-intractable (arbitrary precedence) cases matches the scoping below, not as a
  source of specific formulas (not accessed in full — paywalled monograph).

## Scope decision

1. **Heterogeneous max (new building block, `heterogeneous_max_moments`):** `n` independent
   branches, each its own family/parameters (any mix of Pareto/Gamma/H2/Erlang already supported by
   the library). Raw moments via the standard identity
   `E[max^k] = k * integral_0^inf t^(k-1) * P(max > t) dt`, `P(max>t) = 1 - prod_i(1-tail_i(t))`,
   evaluated with `scipy.integrate.quad` (already used elsewhere in the library, e.g.
   `theory/srpt/utils/predictor.py`) — numerically exact given each branch's assumed family
   (same standard as the existing i.i.d. non-Pareto case), reduces to the closed-form
   `pareto_max_moments` in the i.i.d.-Pareto special case (regression test).
2. **Series-parallel DAG composition (`ForkJoinDAGCalc`):** a small recursive spec
   (`leaf`/`series`/`parallel` nodes) composed via `conv_moments` (series = sum, already exists,
   used elsewhere in the library) and `heterogeneous_max_moments` (parallel = fork-join). **Exact
   only for a single composition level whose leaves are raw distributions** (same as building
   block #1); a `parallel` node whose children are themselves composite subtrees needs a
   moment-fit step to turn the subtree's computed raw moments back into a tail function for the
   next `quad` integration — this is the *same* fit-based approximation `SplitJoinCalc` already
   uses for its non-Pareto i.i.d. case, not a new source of inexactness. The whole DAG's root
   moments are then wrapped in the existing exact M/G/1 (Pollaczek–Khinchine) formula
   (`MG1Calc`), reusing the Split-Join **blocking** semantics of EPIC-022 (no job overlap) so no
   new queueing-level machinery is needed.
3. **Reserve, not in this epic:** general (non-series-parallel) precedence DAGs — genuinely
   #P-hard in general per Dodin 1985; would need bounding techniques (Dodin's own bounds, or
   Monte Carlo) rather than an exact/quasi-exact calculator.

Full derivation: `docs/roadmaps/fork_join_dag_heterogeneous_roadmap.md`.
