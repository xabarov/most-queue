# Finite core and exit phase continuation for infinite TTFT

2026-10-11. [EPIC-097 protocol](../epics/EPIC-097-core-aware-continuation.md),
[all attempts and results](core-aware-continuation-results-2026.json),
[reproduction](../../examples/core_aware_continuation_experiment.py).
Finite-core continuation recovers four near-critical accuracy/budget refusals:
**28/32 bounded instead of24/32**. The frozen twofold state-reduction target
fails: only1/32, not the required16/32. The same iid queue uses tighter return
bounds, not a new scheduler or a finite-loss approximation. Means remain float.

## Exterior and phase accounting

Use the EPIC-096 corrector c(a)=b(a)-min b+1, epsilon>0, M2 and holding/area
constants. For h>=B the active pool is full. At an exit from any domain N>=B-1,
every vacancy is filled, so the next phase a' is the canonical full iid refill
of the known survivors. Conditional on this exit it is independent of the
Poisson count J. No waiting mark was disclosed or used for admission.

Instead of cmax, compute c1=E[c(a')], c2=E[c(a')²]. For unconditional exit
moments p,m1,m2 of H, the potential moments are

`EY=m1+c1*p`, `EY²=m2+2*c1*m1+c2*p`.

With `n=EY/epsilon`, `g=(EY²+(M2/epsilon)*EY)/(2*epsilon)`, bound the excursion
until first return to C={idle or h<B} by

`e=(dmax*n, dmax*g+Cpop*n, dmax*g+Cwait*n, dmax*g+Cpop*n)`.

Coordinates are physical time, first-output area, waiting area and population
area. These are the unchanged EPIC-096 exterior comparison bounds integrated
over the **unbounded** overshoot and actual refill law. Since c1<=cmax and
c2<=cmax², they weakly improve the global-shift bounds in exact arithmetic.

## Finite return system and repeated excursions

Choose an auxiliary domain F={reachable (a,h):h<=K}, K>=B-1. Remove idle z
from its genuine killed transition matrix, obtaining Q. Let r be the four
within-iteration rewards, pexit the one-step exit probability, and e the
unconditional exterior reward above, **without any core continuation term**.
Solve using a sparse factorization, not an explicitly inverted dense matrix:

`A=(I-Q)^-1(r+e)`, `q=(I-Q)^-1 pexit`,
`s=(I-Q)^-1 P(.,z)`.

There is a positive no-arrival path to idle from every finite state; thus Q
is transient and q+s=1. A bounds reward until idle, or exit followed by return
to C. The true full return values U are finite by the EPIC-096 clearing bound.
Let M=max_{i in C\{z}} U_i coordinatewise. Strong Markov decomposition gives
`U_i<=A_i+q_i*M`. At a core maximizer,

`M<= max_{i in C\{z}} A_i/(1-q_i)=max_i A_i/s_i`.

Therefore use this uniform core bound for **all subsequent excursions**, not
just the first one. Take its coordinatewise minimum with the independently
valid old clearing-trial bound. Production exit continuation is `e+p*M`.
The usual killed-cycle time/reward ratio bounds are otherwise unchanged.

The row-wise ratio bound is <= `max_i A_i / min_i s_i`, the separate-max
geometric estimate. This is a direct algebraic inequality, not a claim of a
new general theorem. A strict improvement requires maxima and minima to differ.
At a common numerical N, unchanged retained rewards and smaller continuation
give nested W/V/TTFT intervals relative to EPIC-096 in exact arithmetic.

The scalar M still discards the actual distribution of return phases. The
exterior quadratic bound, dmax and M2 remain global. No assertion of optimal
bounds or near-critical complexity rate follows from this construction.

## Attribution and numerical scope

Regeneration, exterior Lyapunov bounds and finite-domain excursion accounting
are classical; [Glynn–Zheng2025v1](https://arxiv.org/html/2505.03157v1), §2,
equations13–21 and Theorem2.4, provide the closest checked framework.
The phase-mixture and row-wise core closure above apply it to our token policy.
Publication novelty requires a separate comparison, not just tighter numbers.

Checks include finite rewards, positive idle probabilities, mass and absorption
balance, relative linear residuals, combined count-system budgets and explicit
refusal on singular/noncontracting systems. Checks are numerical diagnostics,
not outward-rounded enclosures. `roundoff_certified=False` remains mandatory.
The old clearing bound is still constructed as a fallback; its overflow can
refuse the new calculator too. Critical/rate/support restrictions are unchanged.

## Experiments and verification

K8, all three ablations, original48 and near-critical32, 0.1%+1e-8 widths,
combined20000state/2000000kernel-word budgets and fresh five-seed token replay
were frozen before the run. No threshold, seed, cost or domain sequence was
changed to improve the outcome. All failed domain attempts remain in JSON.

| Grid | Global | Exit phase only | Finite core |
|---|---|---|---|
| Original48 | 48bounded, selected N<=32 | 48bounded, N<=32 | 48bounded, N<=16 |
| Near-critical32 | 24bounded,8word-budget refusals | 24bounded,8word-budget refusals | 28bounded,4word-budget refusals |

All219 common-domain checks pass:56original and163near-critical, checking
phase within global and core within phase. The standalone phase ablation
does not reduce selected N on this grid: most improvement comes from the
core bound. Original maximum main states701→365, but the auxiliary work gives
maximum **total562**, and low-load cases can use twice as many states as before.

Of24 jointly bounded near-critical cases,16 reduce total selected states,
eight increase them. The global/core state ratio has median1.726 and range
0.871–3.441. Only one satisfies >=2; recovery of four baseline refusals is
separate and is not counted as infinite speedup. The target >=16/32 therefore
**does not pass**. Coarse powers-of-two N and auxiliary overhead matter.

Representative mixed-law cases, costs(.1,.02,.003):

| M | Saturated load | Global selected N / states | Core N / main+auxiliary states |
|---|---|---|---|
| 2 | .85 | 128 /2717 | 64 /1373+197=1570 |
| 2 | .95 | word-budget refusal | 256 /5405+197=5602 |
| 4 | .85 | 128 /1941 | 64 /981+141=1122 |
| 4 | .95 | word-budget refusal | 256 /3861+141=4002 |

All four mixed-law cases at load.98 still refuse at N512 because of kernel
word budgets; their strict subcritical drift certificate is not withdrawn.
Homogeneous cases at.98 remain bounded at N512 by all three methods.
Max selected total count states for the new near-critical grid is5602.

### Independent token replay

Fresh seeds97000..97004,100000jobs each,20%warmup, mixed law, first cost set,
M2/4 and saturated loads.85/.95. All12 W/V/TTFT predictions lie inside the
five-replicate t95 intervals. At.95 the empirical mean waiting differs from
the analytic midpoint by8.17%(M2)/8.28%(M4), and TTFT by7.19%/7.25%; these
differences and broad sampling intervals are preserved, not hidden by retuning.
The narrow analytic interval bounds truncation, **not DES sampling error**.
Five replicates and finite warmup do not establish statistical coverage or
accuracy on production traces. No new hardware fitting or decisions were tested.

### Artifact and resource accounting

Every successful and failed refinement is retained. Elapsed times are a
single-run descriptive measurement, not a repeated paired speed benchmark;
some early original-grid attempts overlapped targeted unit checks.
CSR bytes sum matrix data/indices/indptr arrays for both count kernels, not
peak RSS or sparse-factorization storage. Kernel-word counts cover their
arrival/admission enumeration, not every phase-mixture arithmetic operation.
Saturated phase certification has the separate EPIC-096 budget and timing.

To keep the artifact under the existing2MiB file limit, certificates common
across attempts are stored in `phase_certificates`; each result records its
`phase_certificate_id` and the actual `phase_certificate_duration`.
`unpack_results` restores the original dictionaries. Exact JSON round-trip
equality was checked on the complete run; no attempt or diagnostic was dropped.
Compact storage rewriting does not rerun or change measurements.

### Tests and next step

25new tests pass;137 stage-aware family tests pass. Independent arbiters check
direct Poisson/admission-word sums, a finite chain with an explicit exterior
state, repeated excursions, periodic/partial pools, B1 PK, old interval
containment, reconfiguration and all48 unchanged legacy selected answers.
Dropping exterior rewards demonstrably underbounds the independent finite
arbiter. Bad mass, singular/noncontracting solves and combined budgets refuse.
Repository-wide verification uses the tests directory, not root/misc counts.

Fast tests-directory suite:3359passed,16existing warnings,333.54s.
The first full suite had3376passed/1failure,16warnings,516.64s:
`test_mgn_h2_delay_cold_warm`, an unchanged unrelated vacation-queue test,
compares four moments against one100000-job replay without a fixed seed.
Its fourth waiting moment was11788.9503 versus7201.53194 (rtol.5/atol2).
The test does not call the new calculator. Three separate unchanged reruns
passed in1.37/1.92/3.60s; the fast suite also passed it. This is consistent
with sampling variability, not a proof that the legacy approximation is exact.
The second unchanged full run passed **3377tests**,16existing warnings,518.71s.
No tolerance/seed/test was edited. Black/isort/diff checks pass; affected
library/examples pylint10/10, wholelibrary9.99/10 with unchanged unrelated
warnings. Private archive110snapshots/SHA and7archive tests pass; local links
are checked, the94931-byte historical journal prefix and all four prior
GL3/095/096 JSON files are unchanged. EPIC-097 is complete as a research pilot.

EPIC-097 completes a research pilot, not the failed engineering target or a
publication novelty claim. Next choices remain controlled active-phase
compression, dependent-input modeling and independent configuration validation.
Unresolved high-load budgets reinforce the compression need. Automatically
tuning K or changing precision to make this pilot pass is not a next result.

Here the surviving refusals hit **transition-word**, not active-state budgets.
For near-critical load alone, structured Poisson count transitions/finite-domain
solves are another candidate; phase compression is not proved to remove this
count-enumeration bottleneck. Its primary motivation remains larger length
support. Either extension requires a new protocol and separate comparison.
