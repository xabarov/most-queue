# Bulk-service SLA: corrections and new experiments, 2026-10-12

This revision supersedes the affected numerical/contribution claims in
EPIC-066–071, the original SLA reports and the dominant-pole follow-up.
Historical JSON has not been overwritten. This is a correction and reproducible
research result, not a claim of fully certified infinite-queue probabilities.

## Corrections

1. At a>1 and gamma>0, customers behind the tag also abandon. Their live
   population cannot be exactly saturated at a−1. The shared chain now includes
   both R*gamma and K*gamma transitions and selects a numerical K cap using a
   hitting-before-patience bound. The independent a=b=2, lambda=mu=gamma=1,
   R=0 anchor gives P(abandon)=(1+exp(-1))/2=.6839397205857212; the old answer
   was 2/3. Merely adding deaths at C=1 would instead give 11/16, also wrong.
2. H2 waiting mean without abandonment now uses E[Lq]/lambda. Subtracting a
   busy-time-weighted batch duration from customer sojourn was invalid for
   size-dependent service. A departure-flow customer-weighted check is included.
3. Inoue's W denotes **sojourn**, not queue waiting. Theorem 2's phi0 denominator
   is 1+lambda*alpha. The corrected comparison uses deterministic service on
   both sides and does not apply the unbounded bound to a finite cap.
4. The old service-rate exponential surrogate is withdrawn. It did not implement
   Claeys' poles/residues and did not identify the infinite-queue tail exponent.
   The module refuses to regenerate that result. Even M/M/1 uses mu−lambda.
5. Minimum-cap selection scans every integer candidate and tests that candidate,
   without requiring all larger caps to pass. No general monotonicity is proved.
6. Setters invalidate stationary and tagged caches. `get_tagged_wait()` exposes
   reusable moments, tails, the zero atom, all-arrival outcomes and diagnostics.
   H2's historical `get_w()` gamma=0 restriction remains for compatibility;
   the shared tagged representation supplies those moments when explicitly used.

## Mathematical scope

Until tag dispatch, the live behind population is an immigration/death process:
lambda births, k*gamma deaths. No preceding batch can consume behind customers
without consuming the tag. If H_C is its first transition C→C+1 and A~Exp(gamma)
is independent tag patience, suppressing births at C changes outcomes only on
H_C<A. For M with diagonal lambda+(k+1)gamma, upper−lambda, lower−k gamma,

    B_n = E[A^n; H_C<A]
        = n! sum_{j=0}^n e0^T M^(-j-1) lambda eC / gamma^(n-j).

B0 bounds absolute unconditional event errors; Bn bounds unnormalized
dispatched-wait moment errors. Conditional ratios require denominator bounds.
Omitted initial mass delta adds delta or delta*n!/gamma^n respectively.
At gamma=0 the behind saturation at a−1 is exact, and initial states are not
discarded; at a=1 the behind counter is irrelevant. These local bounds exclude
stationary truncation and floating-point rounding. The implementation is not
interval arithmetic and does not certify the full infinite stationary queue.

The model remains an infinite FIFO queue. Suppressed arrivals at stationary N
are a numerical approximation, not an assumption of a physical loss buffer.
The shared chain rejects the nonstationary gamma=0 domain lambda*E[S_b]>=b.

## Frozen protocol and reproducible results

Runner: [examples/bulk_sla_revision.py](../../examples/bulk_sla_revision.py).
Use one of `--panel patience`, `--panel load`, `--panel decision`, `--panel inoue`.
Use `--output-dir recomputed` to preserve the original files when rerunning.
The runner refuses to overwrite existing observations. The
[frozen protocol](bulk-sla-revision-protocol-2026.md) is also included in the
article supplement; all grids are explicit in code and all cases are listed below.

All experiments use affine mean g(s)=.1438s+1.8874 in abstract time units;
the original Inoue calibration was in milliseconds, not LLM seconds.
E3/H2 patience panel: lambda=.8,a2,b8,gamma0/.02/.05/.1/.25/.5.
Load panel: E10/E3/M/H2,b8,a1/2/4,lambda*g(8)/8=.3/.6/.85,gamma0.
Decision panel: E3/H2,lambda=.8,gamma0/.1,b2..16,a1..min(4,b),D5/10/20.
H2 uses balanced-means CV1.5. Inoue panel: deterministic g(s),lambda.2/.8,
a1,gamma0,cap4/8/16/32/unbounded. Thresholds and all seeds are fixed in code.

Stationary attempts N200/400/800/1600, minimum two successive values, boundary
mass<=1e-10 and absolute refinement difference<=1e-7. The implementation checks
the **first two moments**, probabilities, and tails; this is more conservative
than the protocol's prose about means, and the difference is reported in the
paper. Local behind/initial-mass tolerances are 1e-13; state budget 300000/C4096.
No tolerance or budget was loosened after observing outcomes.

| Panel | Scenarios | Outcome |
|---|---:|---|
| [Patience](bulk-sla-revision-patience-2026.json) | 12 | 12 diagnostic convergence passes |
| [Load](bulk-sla-revision-load-2026.json) | 36 | 33 passes; 3 high-load H2 unresolved |
| [Decisions](bulk-sla-revision-decision-2026.json) | 228 | 228 passes |
| [Inoue](bulk-sla-revision-inoue-2026.json) | 10 | Matched deterministic DES, 2 unbounded chain references |

All attempts/refusals, boundaries, local bounds, elapsed times and provenance
hashes are retained. N refinement is diagnostic, not an a posteriori certificate.
Tiny computed values below 1e-7 are not treated as demonstrated tail differences.
Runtime is descriptive, not a benchmark speedup claim.

Independent DES uses 12 replicates of 100000 arrival-cohort jobs after 20000 warmup
arrivals; arrivals continue until every cohort outcome is resolved. Seeds12000..
12011 for the12 patience scenarios and13000..13011 for10 deterministic cases:
264 runs,26.4million recorded arrivals. Replicate-t95 intervals account for
within-run dependence, but are pointwise, not simultaneous. Two of 36 patience
checks (mean wait, abandon, on-time atD5) fall outside their intervals: H2 gamma.02
mean2.486360 vs[2.486605,2.540306], on-time.811045 vs[.807621,.810932].
One of 2 unbounded deterministic sojourn references falls outside its interval:
lambda.8 chain3.243011 vs[3.239147,3.242448]. These misses are preserved, not
hidden by retuning seeds. Rare probabilities are not DES-certified.

## Decision findings

For E3,a2,b8,lambda.8, increasing gamma0→.5 reduces conditional mean wait
1.32444→.65806, while all-arrival timely dispatch atD5 falls.97970→.62763.
H2 likewise shows no improving timely-dispatch hump on this chosen grid.
Abandonment both changes congestion and selects the eventually dispatched jobs.

Two-moment gamma fitting was tested with and without preserving the zero atom.
Both use identical calculated moments, not noisy estimated moments.

| Criterion, 2052 tests per row | Baseline | False acceptance | False rejection |
|---|---|---:|---:|
| Conditional tail | Gamma | 20 | 61 |
| Conditional tail | Atom + gamma | 34 | 20 |
| All-arrival failure | Gamma | 1 | 6 |
| All-arrival failure | Atom + gamma | 1 | 3 |

Preserving the atom reduces total conditional classification errors 81→54, but
increases false acceptance 20→34: **not a uniformly conservative improvement**.
No decisions are within the declared 1e-7 indistinguishability band. For E3, a=1,
gamma=0, D=10, target=.001, the reference-chain minimum candidate is b=5,
whereas atom fitting chooses b=4. No physical GPU-memory/TTFT or universal cap
monotonicity claim follows.

## Attribution and limits

[van Ommeren et al.](https://doi.org/10.1007/s11134-020-09654-y) already use
ahead/behind counters, threshold races and PH waiting distributions. PASTA and
the absorption formula are not new. [Claeys et al.](https://biblio.ugent.be/publication/3033378)
is openly available and uses actual characteristic roots and residues.
[Inoue](https://arxiv.org/abs/1912.06322) supplies the unbounded model and bound.
[Chaudhry et al.](https://doi.org/10.3390/math11051142) use a physical finite buffer
and Padé inversion in their waiting-time procedure; it is not an exact numerical
oracle for the infinite queue here. Claims of priority are deliberately limited.

Only exponential patience, independent Poisson input and gated atomic service
are covered. General patience, correlated input and continuous LLM batching
require separate work. Earlier EPIC-095–097 mean/stability certificates do not
automatically certify this model's SLA tails.

## Validation and reproducibility audit

The final full `tests` run passed **3409 tests**, with 16 warnings in unchanged
parts of the repository (553.17 s, eight workers). All 32 new
[revision regressions](../../tests/units/test_bulk_service_revision.py) pass;
the earlier expanded affected-model run passed 70 tests. Black/isort and
`git diff --check` are clean; Pylint rates the 11 affected Python files 10.00/10.

The first complete run had 3407 passes and one failure in the unchanged,
unseeded `test_mph1_two_class_across_cv[8.0]`: theory 19.99967 versus DES 23.61330
at relative tolerance .15. Repeating that unchanged test file passed all four
tests, and the final full run passed. Its test, simulator, solver, tolerance
and seed handling were not modified to obtain a pass. The first full run
preceded collection of the final integer-patience regression.

All four panels were independently rerun with the finalized computational
implementation. Every scientific field of all **286 scenarios**, including
the 264 DES replicates and all refusals, matched the original observations
at rtol=1e-10, atol=1e-12. Execution times are excluded from the comparison.
Post-run source hashes differ because of formatting, parameter validation,
documentation, integer-gamma handling and portable output/protocol paths;
the original provenance hashes have not been rewritten.

The private article handoff contains the original and repeated data, comparison
report, source manifest, PDF, editable Word and a whitelisted reproducibility
archive. The extracted archive independently passes all 32 regression tests
and rebuilds the same derived results. Original historical results remain
unchanged; the dissertation archive records the correction separately.
