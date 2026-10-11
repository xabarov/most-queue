# Variable token batching analysis and validation

EPIC-085 implements the service mechanism in Yang, Jiao and Xu (2024), retaining
random output lengths instead of replacing random batch durations by their means.
Dynamic, fixed and gated elastic service have separate waiting and sojourn metrics.
The numerical method estimates means, not analytic delay distributions or certified
truncation error bounds. Costs used below are synthetic, not a new GPU calibration.

## Sources and prior work

The full text of [Yang, Jiao, Xu, arXiv:2407.05347v1](https://arxiv.org/html/2407.05347v1)
was read, including equations (1)–(26), experiments and references. Its conditional
batch-cost mechanism and batching policies are attributed to the authors. The
embedded-chain and reward identities come from
[Inoue 2021](https://arxiv.org/html/1912.06322v3), implemented in [L1](dynamic-batching-validation-2026.md).
Neither the mechanism nor numerical truncation is presented as new research.

The targeted check of adjacent mechanisms on 2026-10-11 gives this coverage:

| Area | Source and reading status | Consequence for this work |
|---|---|---|
| Existing local literature | Source cards and classical-result registry; Ryzhikov 2018, `docs/books/RBOOQLan18/Ch2w.tex`, subsection on bulk service | Waiting for N requests is already a classical batching rule, not an LLM invention |
| Fixed-size bulk service | [Chang 2006, DOI 10.1504/IJOR.2006.010212](https://www.inderscience.com/info/e_inarticle.php?artid=10212), publisher abstract, not subscription full text | Embedded analysis and batch-size optimization predate this paper; the inaccessible derivation is not claimed verified |
| Specialized dissertation | [Claeys 2011](https://biblio.ugent.be/publication/1955129), university description, not the complete thesis | Full delay characterization is an established subject of batch-service theory |
| Classical delay distributions | [Claeys et al. 2010](https://biblio.ugent.be/publication/838475), university bibliographic card | Known results include customer-delay moments/tails under full-batch service; no global absence claim |
| ML systems | [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu), author presentation page/abstract | Iteration-level admission differs from the gated elastic simplification of Yang et al. |
| Real-time scheduling | [Online Service with Per-Batch Maximum Delay](https://arxiv.org/abs/2608.18577), abstract only; searches for batch maximum cost and variable processing times | Online dispatch costs are adjacent, not a verified stationary solution of this token model |
| Age of Information | [Peak Age of Information Analysis in Systems with Multiple Time-Correlated Traffic Streams](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902817/), search-level identification; batch-service/AoI mechanism searches | Adjacent freshness models found; their complete coverage was not assessed, so they establish no research gap |

Queries included `bulk service random processing time order statistics maximum`,
`optimal batch size Chang 2006`, `batch service output length real time scheduling`,
and `age of information variable batch service`. They are a targeted check, not a
systematic review proving novelty. Patient users and fixed configured prefill costs
are the scope here; the impatience approximation in paper §III-B and KV-memory
admission are not part of roadmap L2.

## Token and cost conventions

The PMF stores total positive integer output length T. Decode length N=T−1 does
not include the first token, as required by equation (18). For a batch of b iid
marks, S=sum(N_i), M=max(N_i), P_b=k1*b+k2:

```text
Padded duration H = P_b + (k3*b+k4)*M
Elastic duration H = P_b + k3*S + k4*M
Elastic completion C_i = P_b + k3*sum_j min(N_i,N_j) + k4*N_i
```

Arrivals during a batch wait for its final completion under both policies. Elastic
members can leave earlier, reducing both their sojourn and actual occupancy. No
new member joins that batch, so this is not Orca-style continuous batching.

Clipping moves all mass above max_tokens to the cap. It is censoring, not a
renormalized conditional distribution below the limit. Empirical/sparse finite PMFs,
discrete uniform lengths, and explicitly ceiled/censored lognormal and nonnegative
normal variates are supported. These discrete laws are not silently described as
the paper's continuous uniform or Gaussian laws.

## Conditional moments and count law

For sorted support n_j with probabilities p_j and cumulative F_j,
P(M=n_j|b)=F_j^b−F_{j−1}^b. This determines padded moments. Elastic moments need
E[S²]=b Var(N)+b²E[N]² and E[SM], not only E[M]:

```text
E[SM] = b sum_j p_j*n_j*E[max(n_j,M_{b-1})]
E[C_i|b] = P_b + k3*(E[N]+(b-1)*E[min(N_1,N_2)]) + k4*E[N]
```

The latter conditional mean is used as the size-biased job reward. Treating the
last completion time as every elastic member's service time would overstate sojourn.

The count PGF is G_b(z)=E[exp(lambda*(z−1)*H)|b]. For elastic batches, put
t=lambda*(z−1), U_j(t)=sum_{r<=j}p_r exp(t*k3*n_r):

```text
G_b(z) = exp(t*P_b) sum_j exp(t*k4*n_j) [U_j(t)^b-U_{j-1}(t)^b]
```

This keeps the dependence of S and M. An FFT inverts the PGF. Its length is chosen
using the Poisson tail for the largest pathwise duration so aliasing mass is at
most 1e−15 before rounding. Roundoff and stationary truncation still require
diagnostics; this is not an exact symbolic solution. Negative probabilities beyond
roundoff trigger an error. max_fft_points bounds each inversion before allocation.
The count mean/second moment are checked against mixed-Poisson identities.
Finite count survival is summed from the right and its CDF endpoint is exactly
one. A regression at b=23, lambda=3 exposed a generic finite-PMF inverse returning
zero when accumulated rounding left its endpoint below 1−1e−15. That failure
produced false boundary mass and worsening moment errors as truncation grew.
The custom positive-tail inverse removes it without loosening convergence tolerance.

## Dynamic and elastic stationary rewards

Under all-waiting gated dispatch, B'=max(1,A), with A|H~Poisson(lambda H).
Future marks remain iid and independent of the current duration because selection
uses arrival order, not output length. Thus the L1 chain remains applicable:

```text
E[Wq] = E[B(B-1)]/(2*lambda*E[B])
E[V] = E[Wq] + E[B*E[C_i|B]]/E[B]
idle = P(A=0)/E[B]
```

QueueResults.p/pi remain unset in theory: the batch-epoch PMF is not a time-average
queue-length PMF. The simulator separately measures actual occupancy in continuous
time, including early releases.

With finite decode cap d, H<=a*b+c pathwise, a=k1+k3*d, c=k2+k4*d. The check
lambda*a<1 gives a sufficient finite-mean regime, not the sharp necessary condition
for elastic service. An uncapped normal/lognormal maximum has no finite upper
endpoint: F(x)^b tends to zero for every fixed finite x. The paper's description
that a Gaussian maximum becomes fixed cannot justify a global affine envelope.

## General service callbacks transferred from L1

CallbackBatchingCalc accepts moments(b), counts(b,lambda) and optional job_mean(b).
counts supplies a weighted mixture of frozen count distributions implementing
pmf/cdf/sf/ppf/isf/mean/var. The caller also supplies global certificates:

```text
E[H|b] <= a*b+c
E[H²|b] <= u*b²+v*b+w
lambda*a < 1; lambda²*u < 1
```

These bounds make the first/second moment drift contracting outside a finite set.
Finite count caches are limited to twice the matrix entry budget. The solver
validates moment values and consistency at every visited size, but a
finite grid cannot prove the global certificate. Providing valid global bounds
and the true mixed-Poisson law remains the caller's responsibility. Unsupported
laws without such a certificate are rejected, not silently declared stable.
DynamicBatchingSim.set_service_callback independently samples actual durations.

## Fixed batches and formation delay

After departure, Q requests remain. The next batch waits for b−Q arrivals if Q<b.
At its start, R=max(Q−b,0) remain queued; after processing, Q'=R+A. The duration
law is fixed by b, not by Q. The chain is truncated numerically with overflow
folded into its last state, while the physical batch size stays exactly b.

```text
cycle(Q) = E[H|b] + max(b-Q,0)/lambda
formation_area(Q) = [b*(b-1)-Q*(Q-1)]/(2*lambda), if Q<b, else 0
queue_area(Q) = formation_area(Q) + R*E[H|b] + lambda*E[H²|b]/2
E[Wq] = E[queue_area]/(lambda*E[cycle])
E[V] = E[Wq] + E[C_i|b]
```

The fill-area formula follows by summing time spent with Q,Q+1,...,b−1 waiting.
Independent DES explicitly handles those arrivals and does not use this reward.
Fixed results expose a departure-queue PMF, not a time-stationary occupancy PMF.

## Literal paper formulas and independently checked discrepancies

Equation (23), as printed, integrates b*f(x)*F(x)^(b−1) without an x factor. This
integrates to one, not E[M]. Independent order-statistic enumeration establishes
the correct expectation used by the implementation.

Section IV-C replaces random H by E[H|b] to obtain M/D^b/1. This is a surrogate,
not the actual random-token service law. For b=1, its waiting-time deficit is
lambda*Var(H)/(2*(1−lambda*E[H])), a direct M/G/1 limiting-case arbiter.

The literal equation (25) is neither the full customer waiting mean nor its full
sojourn mean for b>1. For b=1, it reduces to H+lambda*H²/(2*(1−lambda*H)), exactly
M/D/1 sojourn. Numerically, its converged series equals fixed deterministic
sojourn minus the mean formation delay (b−1)/(2*lambda). This relation is checked
for b=1,2,3,8 against the departure chain, while the chain's full metrics are
checked by an independent request DES. The default 20 terms are exposed literally;
increasing them is not described as a certified exact root solution.

For equation (26), replacing the realized sum by b*E[N] preserves the unconditional
mean but not H's variance or its dependence on M. This implementation retains both.
These are reproducible model/formula checks, not a claim to have corrected every
result or replicated the paper's measured A100 curves. The fitted GPU coefficients
needed for a faithful numerical replication are not provided as a reproducible
parameter table, so matching its 125-second/optimal-batch-8 example is not claimed.

## Waiting bounds with random durations

Both equation (16) bounds can be justified for a **pathwise** H<=a*B+c envelope.
They are not obtained by asserting that the random service is deterministic.
Writing X=E[H²], m=E[B], z=P(A=0), we have E[B²]=lambda²*X+m:

```text
X <= a²*E[B²]+2*a*c*m+c²
=> E[Wq] <= lambda*(a+c)² / [2*(1-(lambda*a)²)] = phi0

H² <= (a*B+c)*H; B*H <= a*B²+c*B; lambda*E[H]=m-z
=> (1-(lambda*a)²)*X <= (a²+a*c+c/lambda)*m-c*z/lambda
=> E[Wq] <= (lambda*a²+lambda*a*c+c) / [2*(1-(lambda*a)²)] = phi1
```

This derivation also handles elastic durations. A bound on E[H|b] alone would not
suffice. Neither formula is a bound on every customer's sojourn or delay tail.

## Reproduction and validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m examples.token_batching_experiment \
  --output docs/research/token-batching-results-2026.json --simulate --replications 4 --jobs 10000
.venv/bin/pytest tests/units/test_token_batching.py tests/test_token_batching_vs_sim.py
```

The [saved experiment](token-batching-results-2026.json) contains 36 configurations
across four token laws, three rates and three strategies, fixed surrogate/literal
formula comparisons, a finite clipping utility search, and physical-cap DES runs.
Independent seeds, whole-batch warmup and replication standard errors distinguish
sampling uncertainty from correlated within-run requests. Tolerances are in
tests/default_params.yaml. Exhaustive individual-mark enumeration, M/G/1 limits,
constant-length reduction to L1, and literal-series comparisons provide deterministic
arbiters alongside the simulation. Final regression results are recorded in
[EPIC-085](../epics/EPIC-085-variable-token-batching.md).

For the sparse law clipped at 16, lambda=1.5 and synthetic batch costs
(k1,k2,k3,k4)=(0.1,0.2,0.0025,0.004), numerical means are:

| Policy | Waiting, seconds | Sojourn, seconds |
|---|---:|---:|
| Dynamic | 0.07981264 | 0.43555489 |
| Gated elastic | 0.07886536 | 0.42818614 |
| Fixed b=4 | 1.00438139 | 1.72220626 |

The fixed strategy's low-load formation cost outweighs its reduced batch overhead
in this example; this is not a proof that fixed always loses. Across the saved
36-case DES comparison, the largest absolute discrepancy divided by the replication
standard error is 2.65. Four replications are illustrative; the unit/slow validation
uses 8/10 independent replications and longer runs.

For a separate single-request clipping example, the original law is
P(T=1,3,10,100)=(0.3,0.4,0.25,0.05), lambda=1.5, a=c=0.05 seconds and theta=0.95.
Among caps {1,2,3,5,8,10,16,32,64,100}, the finite search selects 16. Mean waiting
falls from 4.10925 seconds at cap 100 to 0.17356195 seconds at cap 16 (95.78%
reduction). Five percent of responses are clipped and expected token-fraction
utility becomes 0.958. This is a configured mathematical utility, not measured
language quality or a universal recommendation.

In the deterministic test law T=(1,3,9), p=(0.2,0.6,0.2), lambda=1 and costs
(0.1,0.2,0.025,0.04), fixed b=1 gives waiting 0.25426641 seconds; its
deterministic-mean surrogate gives 0.22425097 seconds. The 0.03001544-second
difference agrees with the M/G/1 variance correction and is not Monte Carlo noise.

Numerical caps, matrix/FFT budgets and convergence diagnostics are reported, not
relaxed automatically. Analytical delay CDFs, higher delay moments, length-aware
scheduling, impatient users and real KV-memory limits remain outside this model.
