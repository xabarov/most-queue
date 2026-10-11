# Infinite stage-aware TTFT: regenerative truncation bounds

2026-10-11 · [EPIC-095](../epics/EPIC-095-infinite-stage-aware-ttft.md) ·
[model/API](../models/stage-aware-ttft.md) ·
[all measurements and refusals](infinite-stage-aware-ttft-results-2026.json).

## Outcome and scope

`InfiniteStageAwareTtftCalc` computes the infinite waiting-room model, not
the stationary finite-loss model with small blocking. The iid deferred-mark
reduction and decoder symmetry from EPIC-094 remain exact under their conditions.
The numerical domain kills a regenerative trajectory at its first count exit;
upper bounds account for its entire remaining cycle, including overshoot and
possible repeated excursions. No discarded arrivals or stochasticized rows.

**Analytic truncation bounds are evaluated in ordinary floating point.**
`roundoff_certified=False`: these are not machine-verified enclosures of every
rounding/linear-solver error. A residual is a diagnostic, not an arithmetic
certificate. Do not interpret very small printed widths as validated significant
digits. Hardware cost/model error and simulation sampling error are separate.

## 1. Infinite transitions and rewards

At an iteration boundary the state is `(a,h)`: active attained-token phases and
number of waiting requests. Given duration d and survivors s, arrivals
J~Poisson(lambda*d) are **uncapped**. Fill f=min(B-|s|,h+J); reveal f iid
joint marks, and transition to canonical `(s+new,h+J-f)`.
Every waiting mark remains unobserved and iid conditional on this count/active
history. No length-aware admission, serial dependence, cancellation or KV loss.

For U=h+number of active phases that have not produced first decode, the
unconditional iteration rewards are

\[
T=d,\quad R_F=Ud+\lambda d^2/2,\quad
R_W=hd+\lambda d^2/2,\quad R_V=(h+|a|)d+\lambda d^2/2.
\]

The `q==input_length` phase still has no first output. Idle holding is 1/lambda
and has zero area; the first arrival starts immediately. The genuine infinite
accepted rate is lambda, blocking0. B1 preserves the exact M/G/1 PK shortcut;
its unused truncation/tail/cycle/domination diagnostics are `None`, not zero tail.

## 2. Sufficient stability by service credits

Assign to attained token j the positive credit
`c_j=alpha+beta+gamma*j`. For L=S+O a request's full reserved work is

\[
D(S,O)=(\alpha+\beta)L+\gamma L(L+1)/2.
\]

An iteration advancing a block q+1,...,q+k costs beta*k+gamma*(q+k)
for that request. Its token credits dominate this contribution, since k>=1
and all coefficients are nonnegative. Every nonempty iteration advances at
least one token: with M>=B the existing decoders cannot consume all capacity
when there is a prefill. Thus the sum of consumed alpha credits also covers
the one shared overhead alpha. Requests with zero progress incur no separate
gamma cost in this contract. Total consumed credits dominate iteration d.

Couple a work-conserving M/G/1 workload Y to the same Poisson arrivals and D
marks. At boundaries Y dominates the total remaining credits: over an
iteration Y drains d, while real remaining credits drop by at least d.
During that iteration active credits at its start cover the whole d, so
Y cannot empty before it finishes and always dominates the waiting credits.
Admission only transfers credits between waiting and active. This induction
holds until real idle; the virtual busy period cannot end earlier.
Starting at an exit boundary, give all active jobs their **full** credits as
an upper bound on the remaining ones, and all h waiting jobs iid full D.

Consequently rho_D=lambda*E[D]<1 guarantees finite mean return to idle;
bounded support gives the additional moments used below. This proves positive
recurrence on the class reachable from empty without assuming irreducibility
of a saturated active-phase chain. It is deliberately conservative. It does
not establish instability when rho_D>=1; serialized isolated-service load is
also not the exact batching threshold. Saturated phases can have multiple
closed/periodic classes, so solving a single arbitrary phase stationary law
would not justify a general threshold.

A small structural check makes the last warning concrete. With one type
S=O=2, B2/M4, the reachable saturated attained-token pairs have two closed
cycles: `(0,0)->(2,2)->(3,3)->(0,0)` and
`(2,0)->(2,3)->(3,0)->(2,0)`, after canonicalizing decoders. At
costs(.1,.02,.003) both cycles take.514 time and complete2 jobs, so their
capacities happen to agree at3.89105058. This verifies nonirreducibility,
not different capacities or a general stability theorem. The workload
certificate above needs neither choosing a cycle nor a uniqueness assumption.

## 3. Continuation rewards

Let a=1-rho_D>0 and m2=E[D^2]. For a compound-Poisson workload started at x,
the busy-period duration potential and integrated-work potential are

\[
t(x)=x/a,\qquad g(x)=x^2/(2a)+\lambda m_2 x/(2a^2).
\]

For the workload generator `L f=-f'+lambda E[f(x+D)-f(x)]`, direct
substitution gives `L t=-1`, `L g=-x` and t(0)=g(0)=0. Stopped Dynkin
identities/localization give the busy-period expectations; bounded D and
strict negative mean drift justify integrability. This derivation, not an
exponential-service assumption, supplies the continuation bound.
Let dmin=min D. At every physical time waiting count<=Y/dmin and total
population/unfirst count<=Y/dmin+B. The entire remaining real cycle therefore
has duration<=t(x), waiting area<=g(x)/dmin and either other area<=g(x)/dmin+B*t(x).

For an exit state with H waiting jobs, replace all active remaining work by
C=B*max D. Independent unrevealed waiting marks give

\[
E[X\mid H]\le C+H m_1,\quad
E[X^2\mid H]\le(C+H m_1)^2+H\operatorname{Var}(D).
\]

These positive polynomials yield upper bounds even though active marks and
the exit count are dependent. Overshoot is not replaced by N+1.

## 4. Killed-cycle solve and overshoot

Retain states h<=N. From `(a,h)`, put v=B-|s| and t=N-h+v+1.
Only J<t stays inside, and every J>=t fills v slots and exits with
H=h+J-v. Using Poisson tails Q(t)=P(J>=t):

\[
E[J;J\ge t]=\mu Q(t-1),\quad
E[J^2;J\ge t]=\mu^2Q(t-2)+\mu Q(t-1),\qquad \mu=\lambda d.
\]

Expanding H and H² gives the unconditional exit moments used to integrate
t(X), g(X). No infinite arrival-word enumeration is needed. The row's
retained transition mass plus exit probability is1. The nonidle killed
matrix Q is substochastic. Solve `(I-Q.T)v=nu` for expected visits before
idle return or exit; include one initial idle visit. Multiplying by the
unconditional iteration rewards gives retained time T0 and areas R0.
Multiplying by the exit continuation bounds gives ET and ER. Strong Markov
and nonnegative rewards yield true cycle quantities in `[T0,T0+ET]` and
`[R0,R0+ER]`, including paths that exit and later reenter the retained domain.
Flow conservation and regenerative Little's law then give

\[
\frac{R_0}{\lambda(T_0+E_T)}\le E[delay]\le
\frac{R_0+E_R}{\lambda T_0}.
\]

The same ratio logic is classical; we specialize the physical-time rewards
and use a global workload bound instead of a phase-specific exterior drift.
`w[0]`, `v[0]`, `ttft_mean` are interval midpoints, not exact closed forms.
Utilization follows from exactly one idle holding of mean1/lambda per cycle.

`waiting_tail_upper` bounds **time-stationary actual waiting count** P(Wq>N),
not an embedded-epoch distribution and not P(TTFT>x). Before first exit, count
can exceed N within an iteration and fall after admission. Its time is bounded
by `d*P(J>N-h)` for each visited state. Add ET for all post-exit time and
divide by T0. Omitting this within-iteration term would be an incorrect bound.

## 5. Frozen results

Of48 original-grid cases,45 return bounds:24 B1 PK and21 of24 batched cases.
All successful B>1 cases meet the prespecified width criterion <=0.1% for
each W/V/TTFT (plus1e-8 absolute allowance), using N<=32 and <=701 states.
Maximum selected relative widths: TTFT0.014222%, W0.076708%, V0.012464%.
Maximum returned waiting-tail upper bound6.1331e-6; maximum solve residual
2.7756e-16. These are model-bound evaluations, not validated rounding digits.

Three explicit sufficient-stability refusals at serialized load.8 and
costs(.1,.02,.003): single-type M2/M4, rho_D=1.002457, and mixed M4,
rho_D=1.006547. No accuracy or state-budget refusal on this grid. The refusals
are a negative result about this domination certificate, not about real stability.
The protocol was not loosened to recover them.

Fresh independent DES: mixed law, rate1.3, costs(.1,.02,.003),30000jobs,
warmup20%,seeds95000..95002. At N16 the predictor TTFT is.39438136(M2)
and.32765156(M4); DES replicate mean/95% t interval is
.39598145/[.39099709,.40096580] and.32827282/[.32414711,.33239852].
W/V predictions also fall inside their replicate t intervals. Three replicates
are a small stochastic diagnostic, not proof or calibrated asymptotic coverage.
The old48-case DES artifact is reused and explicitly historical, not rerun.

Secondary historical finite-model diagnostic: K6 mean TTFT falls outside the
new infinite intervals in13/21 successful batched cases. Largest absolute
difference is.01125169 (single-type, B2/M2,load.8,held-out costs), despite
finite blocking only.00181231. This is not an error of the exact finite model:
its accepted cohort and physical loss policy differ from the infinite model.
It illustrates why a small blocking probability is not a mean-error certificate.
Both the old finite result and the new interval are retained in the JSON.

## Reproduction and boundaries

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m examples.infinite_stage_aware_ttft_experiment
pytest tests/units/test_infinite_stage_aware_ttft.py
```

31 new tests cover Poisson moment positive-series checks, killed mass, real
infinite rewards, consumed credits, analytic B1 containment, refinement,
low load, independent token replay and explicit guards. Frozen costs are not
fitted to the oracle. EN/RU API/catalogs and private dissertation synthesis
are updated alongside these artifacts. Final suite counts are recorded in the epic.

No general empirical-support scalability, exact stability boundary, correlated
marks, GPU calibration, quantiles/SLO or arithmetic verification is claimed.
The promising next mathematical direction is phase-dependent drift to improve
the conservative condition, with careful treatment of every reachable closed
phase class; publication novelty remains unestablished.

## Primary method attribution

[Infanger–Glynn2022, arXiv2208.14406v1](https://arxiv.org/abs/2208.14406v1),
§2, supplies the cycle-ratio framework; the experimental HTML is incomplete
after the beginning of its block representation. [Glynn–Zheng2025,
arXiv2505.03157v1](https://arxiv.org/html/2505.03157v1), §2 equations3–12,
gives killed-visit and lower/upper cycle-ratio formulas. We do not claim to
implement their full exterior-excursion theorem or reproduce their experiments.
The explicit token-credit application is derived above; regeneration itself
is not a new contribution. Source cards/local HTML and reading depth are in
the private dissertation folder.
