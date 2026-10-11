# Phase-aware stability and infinite delay continuation: EPIC-096

2026-10-11. [Preregistered protocol](../epics/EPIC-096-phase-aware-stability.md),
[complete results](phase-aware-stability-results-2026.json),
[reproduction](../../examples/phase_aware_stability_experiment.py).
The original grid has **48/48 bounded cases**, including all three sufficient-
stability refusals of EPIC-095. This is a certificate for the specified synthetic
iid token queue, not for production GPU serving, correlated traces or SLO tails.

## Model and exact reachability

The finite-support joint iid law of prompt/output lengths, decode-first token
allocation, FIFO prefill/admission and affine iteration cost remain those of
[EPIC-093](stage-aware-ttft-validation-2026.md). Decoder canonicalization and
deferred waiting-mark disclosure remain those of [EPIC-094](stage-aware-ttft-reduction-validation-2026.md).
Costs/rate are the exact represented binary inputs. Positive input probabilities
are normalized **exactly as rational numbers**; numerical mean evaluation uses
their floating representations. This explicit normalization also applies to B1.

Write a for active stages, h for waiting count, B for pool cap. Start at idle,
then one fresh iid job. At any active iteration let s(a) be survivors and
v(a)=B-|s(a)|. Explore all admissions f=0,...,v(a) and all positive iid type words
of length f. Every finite active-only path is physically realizable with h=0:
choose exactly f Poisson arrivals in that iteration. Conversely every physical
active transition has this form. Therefore this BFS enumerates the exact
reachable active set, without choosing a numerical count truncation.

For h>=B, the pool is full and all v(a) vacancies are filled. Its transition
matrix P is independent of h and lambda. The count increment is

`h' - h = J - v(a),  J ~ Poisson(lambda*d(a))`,

with next phase determined by independent iid admission words. All reachable
full phases are closed under full refill. Each is attainable at arbitrarily
large h by adding arrivals in the final realizing transition. For B1 a later
completion/refill provides the same access. Every finite physical state can
clear to idle with positive probability, and states reachable from idle form
one communicating class. This last fact concerns the **open queue**; P may
have several closed classes and transient phases.

## Multi-class drift certificate

For each closed SCC C solve its stationary probability pi_C, without an
aperiodicity assumption. Define

`capacity_C = (pi_C v)/(pi_C d)`,
`g(a)=lambda*d(a)-v(a)`, `delta_C=pi_C g`.

The denominator is physical time, not number of iterations. Extend delta to
transient phases by `(I-P_TT)delta_T=P_TC delta_C`. Solve

`(I-P)b = g-delta`

with one zero anchor per closed SCC. Class centering makes this finite Poisson
system consistent; the transient block is nonsingular. Float solves only
propose b. For every row, evaluate **exactly using Fraction**

`D(a)=lambda*d(a)+sum_j P(a,j)b(j)-b(a)-v(a)`.

Thus the proof does not depend on a small residual or accurate numerical pi.
If every D(a)<0, epsilon=-max D(a)>0 and
`V(h,a)=h+b(a)-min b+1` has drift <=-epsilon outside the finite core h<B.
V is nonnegative; the core has finite one-step moments. The accessible open
chain satisfies Foster's criterion. Bounded positive holding times and finite
mean regenerative return imply positive recurrence in physical time as well.

If a reachable closed class has every D(a)>0, continue its additive process
without the lower count boundary. Its martingale increments have uniformly
bounded second moments, so the martingale divided by n tends to zero; corrected
count then grows at least linearly. Its path minimum is finite almost surely.
For some sufficiently high reachable start there is positive probability never
to enter h<B. The original open queue has a possible escape and cannot be
positive recurrent. This is not inferred from a finite cohort that eventually
drains in a DES replay.

The implementation accepts these **strict row inequalities** away from a
relative 1e-10 numerical capacity band. Critical/numerically ambiguous cases
are explicitly unresolved; it does not classify null recurrence at equality.
Probability underflow cannot silently remove a positive graph edge. State and
word budgets can produce an explicit refusal, not an approximate certificate.

## Full-cycle mean continuation

An exact drift certificate alone does not bound mean-delay truncation error.
Let `c(a)=b(a)-min b+1`, `Y=h+c(a)` and epsilon be the negative drift margin.
For `K=J-v(a)+b(a')-b(a)`, let `M2=max_a E[K^2|a]`, explicitly

`E[K^2|a]=lambda*d(a)+sum_j P(a,j)(lambda*d(a)-v(a)+b(j)-b(a))^2`.

Then `L Y<=-epsilon`, `L Y^2<=-2*epsilon*Y+M2`, and the nonnegative potential

`G=(Y^2+(M2/epsilon)*Y)/(2*epsilon)` satisfies `L G<=-Y<=-h`.

Stopped before entering h<B, expected step count is <=Y/epsilon and expected
sum of waiting counts is <=G. Put

`dmax=alpha+B*(beta*Smax+gamma*Lmax)`, `Lmax=max(S+O)`.

This overbounds **partial as well as full** pool holding times. Physical time
is bounded by dmax*Y/epsilon. Waiting area is bounded by
`dmax*G+(lambda*dmax^2/2)*Y/epsilon`. Population and not-yet-first-output areas
are bounded by `dmax*G+(B*dmax+lambda*dmax^2/2)*Y/epsilon`.

In the core there are <=2B-1 jobs. With no arrivals, at most
`L0=(2B-1)*Lmax` iterations clear them: at least one token advances per nonempty
iteration. A trial lasting at most `tau=L0*dmax` succeeds with probability at
least `p0=exp(-lambda*tau)`. If it fails, return from outside to the core using
the exterior potentials, then try again. At trial end waiting count is dominated
by `B-1+Poisson(lambda*tau)` using arrivals over a deterministic enclosing time
window. With `cmax=max c`, set

`EY=B-1+lambda*tau+cmax`, `EY2=EY^2+lambda*tau`,
`EG=(EY2+(M2/epsilon)*EY)/(2*epsilon)`.

Uniform full return bounds from the core are

`Tcore=(tau+dmax*EY/epsilon)/p0`,
`Rcore=(tau*(2B-1+lambda*tau)+dmax*EG+(B*dmax+lambda*dmax^2/2)*EY/epsilon)/p0`.

The first-trial cost and conditional success bound hold uniformly on every
restart, giving the geometric factor 1/p0 without assuming independent trials.
Overflow or excessively loose bounds can refuse means while the stability
certificate itself remains valid.

At a killed-domain exit use **unconditional overshoot moments**, not a capped
boundary: `p=Pr(exit), m1=E[H;exit], m2=E[H^2;exit]` from EPIC-095. Replace Y
moments by `m1+cmax*p` and `m2+2*cmax*m1+cmax^2*p`, apply exterior potentials,
then add `Tcore*p` / `Rcore*p`. This covers the entire continuation to idle,
including subsequent excursions. For an exit already in the core, the extra
exterior terms are merely nonnegative overbounds.

The killed-kernel cycle-time/reward ratios, actual within-iteration waiting
tail and B1 PK shortcut are unchanged. Numerical N is **not a loss buffer**.
Only the phase drift inequalities are rationally verified. Poisson tails,
linear solves and all mean-bound arithmetic use ordinary floats;
`roundoff_certified=False`. Narrow intervals are not machine-certified digits.

## Frozen results and independent checks

Original 48 cases: 24 B1 PK and all 24 batched cases bounded; all 48 have strict
negative drift certificates. All three EPIC-095 refusals are recovered without
changing the load/grid or width target. Selected original-grid N<=32, maximum
701 count/phase states. All W/V/TTFT widths satisfy 0.1%+1e-8. All attempts and
exact row drifts are saved in JSON, including numerical corrector residuals.

Extended stability-only 40 cases use eight B2 configurations at saturated
loads .6/.85/.95/1/1.05: **24 positive recurrence, 8 possible escapes, 8 critical
unresolved**. These capacity-designed rates are diagnostic probes, not an
independent capacity oracle. In the homogeneous S=O2,M4 case two reachable
closed three-cycles independently give `2/(3alpha+8beta+18gamma)` jobs/time;
M2 has two closed classes and one transient phase. B1 capacity equals
`1/E[isolated service]`. A synthetic multichain unit arbiter checks differing
class capacities and transient absorption, not an invented observed model.

Fresh independent DES: mixed law, original first cost set, M2/M4, saturated
load .6/.85, 50000 arrivals per seed, warmup20%, three seeds96000..96002.
All 12 mean predictions (W/V/TTFT across four configurations) fall inside the
replicate t95 intervals. At load .85 selected N128, up to2717 states; conservative
workload loads reach1.4051(M2) /1.6869(M4), so this is beyond the old certificate.
Only three replicates give wide sampling intervals, not a coverage guarantee.

| Token budget | Saturated load | Old workload load | Selected N | Predicted W | Predicted TTFT | Predicted V |
|---|---|---|---|---|---|---|
| 2 | .60 | .9918 | 32 | .18282 | .55959 | .62077 |
| 2 | .85 | 1.4051 | 128 | .79329 | 1.20187 | 1.26517 |
| 4 | .60 | 1.1908 | 32 | .16232 | .47602 | .53977 |
| 4 | .85 | 1.6869 | 128 | .63353 | .96557 | 1.03210 |

Costs and delays use the same synthetic time unit; no hardware timing is fitted.
The first seed alone differs by 6–7% in high-load W; a one-seed 3.5% test failed.
The integration test therefore uses the **preregistered three-seed mean**, with
the unchanged prior 3.5% tolerance. No seed or experimental criterion was changed.

Unit arbiters compare phase reachability with the independently constructed
N0 killed count kernel; test periodic SCCs, exact row fractions, physical-time
capacity, B1 PK, critical and positive drift, complexity/underflow refusals,
quadratic potential inequalities, refinement containment and overlap with old
mean intervals. [Optional SymPy reproduction](../../examples/phase_aware_stability_symbolic.py)
checks the three identities first verified before production computation.

## Attribution and next decision

Final verification after overflow guards: 27 new unit/integration tests passed;
112 stage-aware family tests passed. With `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`, `.venv/bin/pytest tests/ -m 'not slow' -n auto -q`:
3332 passed,16 existing warnings,389.16s; `.venv/bin/pytest tests/ -n auto -q`:
3350 passed,16 existing warnings,470.21s. These are **tests-directory counts**,
not the previous root counts that also included misc tests. Black/isort and
diff checks clean; affected library/examples pylint10/10. Full-library
pylint9.99/10 retains unrelated pre-existing warnings, no new phase-module
warnings. Private archive:108 synchronized snapshots/SHA,7 tests passed;
439 Markdown files/2814 local links/0 missing. Historical GL3/095 JSON and
the original87393-byte journal prefix hashes are unchanged.

The Poisson-corrector/martingale mechanism is classical; see
[Glynn–Infanger2022v1](https://arxiv.org/html/2202.10404v1), §1 equations1.1–1.2
and selected beginning of §2. Its irreducible-chain results must be applied per
closed class, not to this whole multichain. The comparison/Foster framework is
[Glynn–Meyn1996](https://web.stanford.edu/~glynn/papers/1996/GM96.pdf), §2,
equation8 and Theorems2.1–2.2, pp919–920. The paper treats general irreducible
chains and explicit Lyapunov bounds; our finite phase/core construction checks
the particular hypotheses here. Regenerative ratio methodology remains
[Glynn–Zheng2025v1](https://arxiv.org/html/2505.03157v1), §2 equations3–12.
These source mechanisms are not our novelty. Exact reachability, multichain
normalization and the clearing-trial continuation above are model-specific
derivations, not a completed publication novelty search.

Recommended next candidate: **EPIC-097, tighter phase-aware continuation bounds**
(finite-core fundamental matrix and phase-specific exit accounting), with a new
preregistered near-critical width/runtime grid. It would address the current
very loose global core constants, not assert novelty automatically. Alternatives
remain real-support compression, verified mean arithmetic, correlated marks or
hardware/SLO validation. None is launched or authorized by completing EPIC-096.
