# EDF (Earliest Deadline First) as a service discipline — literature review (2026)

Distinct from EPIC-021 (`most_queue/theory/utils/sla.py`): that layer computes a deadline-violation
probability *on top of* an already-computed waiting-time distribution for an existing discipline
(FCFS, priority, ...) — it never changes *who gets served next*. This epic is about EDF as the
actual **service discipline**: the job with the smallest deadline is served next, so deadlines
directly determine scheduling order, not just a post-hoc SLA metric.

## Literature

Classical and modern EDF queueing literature is overwhelmingly **asymptotic** (heavy-traffic,
fluid-limit, many-server law-of-large-numbers), not exact finite-queue results:

- Panwar S., Towsley D., **Real-time queues in heavy traffic with earliest-deadline-first queue
  discipline**, Annals of Applied Probability, 2001, doi:10.1214/aoap/1015345295 (145 cites) —
  the foundational heavy-traffic EDF result; state-space collapse, not an exact stationary
  distribution.
- Doytchinov B., Lehoczky J., Shreve S., **Real-time queues in heavy traffic with EDF discipline**
  and follow-ups (accuracy of state-space collapse, 2006, doi:10.1214/105051605000000809; law of
  large numbers for many-server EDF, 2017, doi:10.1016/j.spa.2017.09.009) — same asymptotic family.
- Bhattacharya S., Ephremides A. and others, **On queues with impatience: stability, and the
  optimality of Earliest Deadline First**, Queueing Systems, 2013, doi:10.1007/s11134-013-9342-1 —
  EDF's *optimality* (minimizes deadline-miss rate among all disciplines) under reneging, but
  again a stability/asymptotic-optimality result, not a finite-queue formula.
- Liu C.L., Layland J.W. (1973) and Leung J., Whitehead J. (1982, deadline-monotonic priority) —
  classical **hard real-time** schedulability theory: establishes that EDF strictly dominates any
  *fixed*-priority assignment (deadline-monotonic included) in schedulable utilization. Cited here
  because it directly **rules out** one of the two shortcuts explored below (see "Rejected
  hypotheses").
- Mojalal M., Stanford D., Taylor P., Ziedins I. et al., **the delayed accumulating priority
  queue**: low-priority waiting time (INFOR, 2019, doi:10.1080/03155986.2019.1624473) and
  high-priority waiting time via a conservation law (INFOR, 2022,
  doi:10.1080/03155986.2022.2038962, also arXiv:2001.06054) — the closest **exact** treatment
  found. Result: a 2-class M/G/1 queue where the low-priority class only starts accumulating
  priority at rate `b` after an initial delay `d` (so is strictly dominated by the high-priority
  class until then) has an exact expected-waiting-time algorithm via a classical conservation law,
  but the paper itself describes the general formula as requiring **truncated infinite sums**, not
  closed-form — and the source is paywalled (MathML-rendered, not extractable). See §3 below for
  why this model is the mathematically correct target and why it was not fully re-derived here.

## Rejected hypotheses (tested numerically, both wrong — recorded to save re-deriving them)

Before settling on the scope below, two "surprisingly this reduces to an existing model" shortcuts
were tried, in the same style as prior epics' memoryless-property findings (EPIC-023/027) — both
were **numerically disproven** via direct DES comparison (not just an oversight; the discrepancy
is systematic, not Monte Carlo noise — confirmed across 3 seeds and 1.5M events each):

1. **"EDF with exponential relative deadlines ≡ random-order-of-service (SIRO)"** — the idea:
   memorylessness makes each waiting customer's residual-time-to-deadline `Exp(θ)` regardless of
   elapsed wait, so "who has the smallest deadline" seems exchangeable. **Disproven**: simulated
   FCFS, SIRO, and true EDF (deadline = arrival + `Exp(θ)`, dropped on expiry) on the same M/M/1+M
   system (`λ=0.8, μ=1, θ=0.3`) — FCFS and SIRO matched each other almost exactly (`mean_n≈1.19-1.20`,
   `drop_prob≈0.213`, as expected: aggregate count dynamics are order-invariant for
   memoryless non-priority disciplines), but EDF was clearly different (`mean_n=1.35`,
   `drop_prob=0.186`) — EDF genuinely reduces the miss rate by actively favoring near-deadline
   jobs, something SIRO cannot do since it ignores deadlines.
2. **"2-class exponential deadlines ⇒ exact aggregate `(n1,n2)` CTMC via a race between class-level
   exponential clocks"** — the idea: since each class's residual-to-deadline is (marginally) fresh
   `Exp(θ_k)` at every instant, the probability the next job served is class 1 should be
   `n1·θ1/(n1·θ1 + n2·θ2)` (comparing minima of `n1` vs `n2` i.i.d. exponentials), turning EDF into
   a finite `(n1,n2)`-level CTMC with no per-job deadline tracking. **Disproven**: simulated true
   EDF (per-job absolute deadline comparison) against this aggregate race-based model on the same
   parameters (`λ1=λ2=0.4, μ=1, θ1=0.3, θ2=0.6`) — consistent, non-noise gap across 3 seeds
   (`n1: 0.64-0.65` true vs `0.59` race-model; `drop1: 0.177-0.179` true vs `0.20-0.21` race-model).
   Root cause: the *marginal* fresh-exponential-residual property holds per job, but the *joint*
   selection process is not memoryless at the population level — a job's survival to the current
   instant is correlated with the specific rivals it has already out-waited, which aggregate counts
   alone don't capture. This is the same obstruction the literature above reflects: exact EDF
   analysis needs more state than simple counts (in general, the full sorted deadline list).
3. **"The classical work-conservation law `Σ ρ_k E[W_k] = ρ E0/(1-ρ)` holds exactly for EDF with
   reneging"** — this was the scope's original planned "exact invariant" (see roadmap draft); also
   numerically disproven once reneging is non-negligible (`λ1=λ2=0.4, μ=1, θ1=0.3, θ2=0.6`,
   `miss_prob≈0.18/0.27`): measured LHS/RHS ratio `≈0.35`, nowhere near 1, even after substituting
   the *effective* (served, not offered) arrival rate for `λ_k` in both `ρ_k` and `E0`. Root cause,
   understood after the fact: the classical proof that "total unfinished work" is
   discipline-invariant assumes every arriving job's *entire* service requirement eventually gets
   consumed by the server; reneging removes a job's full (not-yet-started) service demand from the
   system as a "negative jump" whose probability itself depends on the discipline (a job might
   renege under FCFS but be served in time under EDF, so different disciplines see different
   *effective* sample paths of arriving work, not just different scheduling of the same work) —
   this breaks the invariance argument at its root, not just the arithmetic. Re-tested with
   deadlines rare enough that reneging is negligible (`θ→` very small, `miss_prob<0.002`): the
   ratio climbs toward 1 as reneging shrinks (`0.90` at `miss_prob≈1%`, `0.97` at `miss_prob≈0.1%`),
   confirming the law *is* asymptotically valid in the no-reneging limit and that the simulator
   itself is not buggy — it's the invariant's applicability that was overclaimed. **Consequence for
   scope**: the conservation-law regression test is only meaningful in a low-reneging
   configuration (generous deadlines); it cannot cross-validate the lossy/high-miss-rate regime,
   which is exactly the regime EDF is most interesting in. That regime is validated by DES-vs-DES
   comparison instead (EDF vs FCFS on identical parameters, see roadmap §2).
4. **"K fixed deadline classes ⇒ exactly static non-preemptive priority by class"** — ruled out
   *analytically*, not numerically, by the classical Liu-Layland/Leung-Whitehead result above:
   EDF is strictly more efficient than any fixed-priority (deadline-monotonic included) policy,
   which would be a contradiction if the two were identical. Concretely: a low-priority job that
   has already waited past `(d_low - d_high)` should jump ahead of a *freshly arriving*
   high-priority job under true EDF, but static priority never lets that happen — this is the
   textbook reason EDF dominates fixed priority.

## Scope decision

Given the exact finite-state/closed-form case is a genuinely open-ended research topic (confirmed
independently by the literature and by two failed shortcuts above), this epic is scoped to what
**is** rigorously deliverable without overclaiming exactness the model doesn't have:

- **DES simulator** (`most_queue/sim/edf.py`): exact-by-construction (no approximation in the
  discipline itself) for `K` classes, Poisson arrivals, exponential service, per-class deadline
  drawn from any distribution the library already supports (`create_distribution`) — covers both
  the deterministic-per-class-deadline case and the exponential-relative-deadline case explored
  above.
- **Exact cross-check, not a full theory calculator**: the work-conservation law
  `sum_k rho_k * E[W_k] = rho * E0 / (1 - rho)` (E0 = sum_k lambda_k E[S_k^2]/2) holds for *any*
  non-idling discipline, EDF included — this is a genuine exact invariant, used as a regression
  test on the DES output rather than a hand-wavy "close enough" DES/theory match.
  See `docs/roadmaps/edf_scheduling_roadmap.md` §2.
  - **Composition with EPIC-021's SLA layer**: per-class deadline-violation probability is
  computed by feeding the DES-estimated waiting-time moments into
  `most_queue.theory.utils.sla.deadline_violation_prob` (the existing fit-and-tail approach),
  exactly the same composition pattern EPIC-021 already uses for other disciplines — the novelty
  here is that the discipline *itself* is now deadline-aware, not just the post-hoc metric.
- **Reserve** (future epic, contingent on someone porting the Mojalal/Stanford/Taylor/Ziedins
  formulas from the paywalled papers or re-deriving them from scratch with more time budget):
  exact closed-form/series mean-waiting-time calculator for the 2-class delayed-APQ special case.
