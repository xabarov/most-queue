# (n,k)-Fork-Join over heterogeneous/DAG branches — literature review (2026)

Direct continuation of EPIC-030 (`docs/research/fork-join-dag-heterogeneous-2026.md`): that epic
generalized the fork-join *maximum* (n-of-n, every branch required) to heterogeneous branches and
series-parallel DAGs. This epic drops the "every branch required" assumption: **k-of-n** — the
job is done once *any* k of the n branches finish, matching `ForkJoinMarkovianCalc`'s existing
(n,k) semantics but generalized to heterogeneous branches (and, via the DAG, to k-of-n nodes
nested inside a larger task graph).

## Current state in most_queue

- `ForkJoinMarkovianCalc` (`theory/fork_join/m_m_n.py`): (n,k) fork-join, but only **i.i.d.
  exponential** branches (Varma/Nelson-Tantawi interpolation) and **mean only**, not raw moments.
- `heterogeneous_max_moments` (EPIC-030, `theory/utils/max_dist.py`): exact-given-family moments
  of the **maximum** (n-of-n = k=n special case) of heterogeneous branches.
- `ForkJoinDAGCalc` (EPIC-030): DAG composition, but every `parallel` node requires *all* its
  children (no partial/k-of-n join anywhere in the DAG).
- **Gap:** no k-of-n (k<n) treatment anywhere for heterogeneous branches, and no raw moments
  (only means) even for the homogeneous case.

## Literature

- **Order statistics of independent non-identical (INID) random variables — the general building
  block:** David H.A. & Nagaraja H.N.'s classical framework; specific applications: *Order
  statistics from non-identical exponential random variables and some applications*,
  Computational Statistics & Data Analysis, 1994, doi:10.1016/0167-9473(94)90172-4 (38 cites);
  *Computing moments of discrete order statistics from non-identical distributions*, Journal of
  Computational and Applied Mathematics, 2017, doi:10.1016/j.cam.2017.07.017; *Computing the
  moments of order statistics from nonidentically distributed phase-type random variables*,
  Journal of Computational and Applied Mathematics, 2010, doi:10.1016/j.cam.2010.11.012 —
  particularly relevant since PH distributions are already a first-class citizen of this library's
  MAP/PH stack.
- **k-out-of-n reliability with heterogeneous components — the same math under a different name:**
  computing "P(at least m of n independent events occur)" for heterogeneous per-item probabilities
  is *exactly* the k-out-of-n:G system reliability problem (a system works iff at least k of n
  components work), solved by a classical O(n²) DP recursion: Rushdi A.M., *Recursive algorithm
  for reliability evaluation of a k-out-of-n:G system*, IEEE Transactions on Reliability, 1985,
  doi:10.1109/tr.1985.5221975 (38 cites) — the Poisson-Binomial-CDF recursion this epic reuses;
  modern review: Chen S.X. & Liu J.S. et al., *The Poisson Binomial Distribution — Old & New*,
  Statistical Science, 2022, doi:10.1214/22-sts852 (27 cites); *Exact equation and an algorithm
  for reliability evaluation of K-out-of-N:G system*, Reliability Engineering & System Safety,
  2002, doi:10.1016/s0951-8320(02)00046-7.
- **Heterogeneous fork-join, very recent activity (confirms this is a live topic, not solved
  territory):** *On the Busy Cycle Maxima in a Heterogeneous Fork-Join Queue*, ACM SIGMETRICS
  Performance Evaluation Review, 2025, doi:10.1145/3764944.3764972; *Stability of Fork-Join
  Systems with Redundancy and Heterogeneous Servers*, SSRN, 2026, doi:10.2139/ssrn.7432278;
  *Delay-Optimal Policies in Partial Fork-Join Systems with Redundancy and Random Slowdowns*,
  SIGMETRICS/Performance abstracts, 2020, doi:10.1145/3393691.3394181.

## A bonus exact closed form (not previously in the repo)

The i.i.d.-Pareto **maximum** already has an exact closed form (`pareto_max_moments`, EPIC-022,
via the `U=F(X)~Uniform(0,1)` substitution and the Beta function). The *k-th order statistic* of
n i.i.d. Pareto generalizes cleanly the same way: `X_(k) = K/(1-U_(k))^(1/alpha)` where
`U_(k) ~ Beta(k, n-k+1)`, so `1-U_(k) ~ Beta(n-k+1, k)` and

```
E[X_(k)^m] = K^m * B(n-k+1 - m/alpha, k) / B(n-k+1, k)
```

(convergent for `m < alpha*(n-k+1)`, which reduces exactly to the existing `m < alpha` condition
at `k=n`). This is a genuine new exact building block (`pareto_kth_order_moments`), not just an
approximation — and doubles as the regression anchor for the heterogeneous numerical case (set all
branches to the same Pareto and confirm the DP+quadrature result matches this closed form).

## Scope decision

1. **`pareto_kth_order_moments(params, n, k, num)`** (new, exact closed form) — i.i.d. case,
   generalizes `pareto_max_moments` (`k=n` special case, regression-tested against it).
2. **`heterogeneous_kth_order_moments(branches, k, num)`** (new, numerically exact given each
   branch's family) — `P(X_(k) <= x) = P(at least k of the n branches have finished by x)`
   (standard order-statistics fact: the k-th smallest is `<=x` iff at least `k` values are),
   computed via the Rushdi/Poisson-Binomial O(n²) DP over each branch's CDF at `x`, then the same
   `E[Y^m] = m∫x^(m-1)(1-P(Y<=x))dx` quadrature already used for `heterogeneous_max_moments`
   (`k=n` reduces to it, regression-tested; an earlier draft of this doc had `n-k+1` instead of
   `k`, swapping min/max -- caught by the `k=n` regression test itself during implementation, see
   the epic's "Результаты").
3. **`ForkJoinDAGCalc` `parallel` node gets an optional `k`** (defaults to `len(children)`,
   preserving EPIC-030's existing all-required semantics) — lets any node in a series-parallel DAG
   express k-of-n partial completion, not just the DAG's top level.
4. **Reserve, not in this epic:** purging vs. non-purging semantics (whether the `n-k` unfinished
   branches keep consuming server capacity after the k-th finishes — `ForkJoinSim`'s DES already
   distinguishes these for the homogeneous case; the heterogeneous *theory* side here computes the
   join time only, agnostic to what happens to the leftover branches after — a non-purging-style
   answer by construction, matching `ForkJoinMarkovianCalc`'s convention).

Full derivation: `docs/roadmaps/fork_join_nk_heterogeneous_roadmap.md`.
