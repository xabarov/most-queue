# LLM-serving SLO: exact deadline-aware admission control — literature review (2026)

Distinct from EPIC-021 (SLA layer: passive `P(W>D)` computed *after* the fact, on top of any
already-fixed discipline) and EPIC-029 (EDF: reorders *service*, discovered to have no exact
finite-state solution in general). This epic targets a third, genuinely different mechanism:
**admission control** — a proactive accept/reject decision made *at arrival*, based on whether the
customer's own (random) deadline is still feasible given the current system state, while keeping
FCFS service order. Rejected customers never join the queue at all (not reneging/abandonment,
which happens *after* joining).

## Literature

- Das S., Jenkins L., Sengupta D., **Analysis of an M/M/1+G queue operated under the FCFS policy
  with exact admission control**, Queueing Systems 75, 169–188 (2013), doi:10.1007/s11134-013-9366-6
  — the direct precedent: M/M/1, FCFS order preserved, each customer carries an i.i.d. relative
  deadline from an arbitrary distribution `G`; admitted iff the deadline is still feasible given
  the current workload. The paper derives an explicit solution to a functional equation for the
  stationary workload distribution, and explicit loss ratio / sojourn-time-distribution formulas.
  Not accessible in full (paywalled) — the derivation below is independent, grounded in a classical
  level-crossing argument for the M/G/1 workload process (Takács), not transcribed from the paper.
- Related admission-control literature confirming this is an established, distinct sub-area:
  *Queueing models with admission and termination control: monotonicity and threshold results*,
  TU/e PhD thesis, 2003, doi:10.6100/ir570263; *Optimal Pricing and Admission Control in a Queueing
  System with Periodically Varying Parameters*, Queueing Systems, 2004,
  doi:10.1023/b:ques.0000035313.20223.3f.
- LLM-serving motivation (2025-2026, confirming the practical relevance rather than providing exact
  math): *Optimal Scheduling Algorithms for LLM Inference: Theory and Practice*, ACM POMACS, 2025,
  doi:10.1145/3771574; *QUARTZ: Quantile-Aware Routing and Queueing for TTFT SLOs in LLM Serving*,
  ACL Findings, 2026; *Tool-Augmented LLM Serving Under Firm Deadlines: A Queueing-Control
  Approach*, 2026, doi:10.2139/ssrn.6438169 — these are largely algorithmic/heuristic or
  RL-flavored, not exact queueing-theoretic analysis; cited here only to confirm the topic's
  practical relevance, not as a source of the math below.

## A wrong shortcut, caught before shipping (documented so it is not retried)

The first hypothesis tried: since M/M/1 service is memoryless, the workload seen by an arrival
with `n` customers already present should be exactly `Erlang(n, mu)`, turning the admission-control
system into a simple state-dependent birth-death chain on `n` with birth rate
`lambda * P(D > Erlang(n,mu))`. For `D ~ Exp(theta)`, this gives the tidy closed form
`P(admit | n) = (mu/(mu+theta))^n`. This formula itself is correct in isolation (verified against
direct Monte Carlo sampling of independent `D` and `Erlang(n,mu)`), but the birth-death chain built
from it is **not** a valid model of the real system: conditioning on "admission occurred" at the
transition into level `n` size-biases the *specific* realized workload toward smaller values (via
the non-constant weight `tail_D(w)`), and this bias does **not** wash out on subsequent instants
the way ordinary memorylessness would — because the conditioning event is on the *old* value
`w` through `tail_D(w)`, not simply on "still in service", which is what plain memorylessness
resets. Concretely: `n` is *not* a sufficient statistic for this system. This was caught by
comparing the birth-death shortcut's predicted `P(admit|n)` against the *empirical* conditional
admission probability from a careful discrete-event simulation of the real system — a small but
consistent and seed-stable discrepancy (worse at larger `n`), not Monte Carlo noise. This matches
why the 2013 paper works with the full continuous workload process (a functional equation), not a
discrete-`n` shortcut.

## The correct derivation (level-crossing, validated against DES)

Let `U(t)` be the virtual waiting time (workload) process: decreases at rate 1 between events,
jumps up by a fresh `Exp(mu)` service draw at each *admitted* arrival. Poisson(`lambda`) arrivals
observe `U(t^-)` (PASTA) and draw `D ~ Exp(theta)` independently; admitted iff `D > U(t^-)`.
Standard M/G/1-style level-crossing balance for the stationary density `f(x)` of `U`
(`F` = CDF, `pi0 = P(U=0)` the idle-probability atom):

```
f(x) = lambda * pi0 * tail_D(0) * (1-B(x))  +  lambda * integral_0^x tail_D(u) f(u) (1-B(x-u)) du
```

(`B` = service CDF, `1-B(x) = e^{-mu x}`, `tail_D(u) = e^{-theta u}` for `Exp(theta)` deadlines).
Taking the Laplace-Stieltjes transform `phi(s) = E[e^{-sU}]` (including the atom `pi0` at 0) and
using `tail_D(u) = e^{-theta u}` to collapse the weighted transform to a *shift*, `phi(s+theta)`,
gives a clean recursive functional equation:

```
phi(s) = pi0 + [lambda / (s + mu)] * phi(s + theta)
```

Iterating this (a shift by `theta` each time) telescopes into a rapidly convergent series —
convergent for *any* `lambda, mu, theta > 0` (the denominators grow like `(n*theta)!`-ish, so the
system is always stable regardless of `lambda`, since large-workload arrivals get auto-rejected):

```
phi(s) = pi0 * sum_{n=0}^inf  lambda^n / prod_{j=0}^{n-1} (s + j*theta + mu)
pi0 = 1 / [sum_{n=0}^inf lambda^n / prod_{j=0}^{n-1} (j*theta + mu)]
```

`loss_prob = 1 - phi(theta)` (since `phi(theta) = E[e^{-theta U}] = E[tail_D(U)] = P(D>U)` by PASTA,
the overall acceptance probability). Raw moments of `U` (unconditional) via repeated differentiation
of `phi` at `s=0`; moments of `U` *given admission* via differentiation of `phi(s+theta)` at `s=0`
divided by `phi(theta)` (the same size-biasing-by-`tail_D` correction that broke the naive
birth-death shortcut, now applied correctly as a proper conditional-moment computation rather than
folded into a wrong Markov-state assumption). Sojourn time of an admitted customer:
`V = U + S`, `S ~ Exp(mu)` drawn fresh *after* admission (independent of `U` and of the admission
event) — so `V`'s moments given admission are the exact convolution of `U`'s admitted-conditional
moments with `Exp(mu)`'s moments (`conv_moments`, already in the library).

**Validated against an independent, from-scratch continuous-workload DES** (not the earlier,
wrong `n`-based one): `pi0`, `E[U]`, `loss_prob`, and `E[V|admitted]` all matched to within normal
Monte Carlo tolerance across three seeds (`lambda=1.2, mu=1.0, theta=0.5`: theory `pi0=0.2394` vs
DES `0.239-0.240`; theory `E[U]=1.258` vs DES `1.255-1.262`; theory `loss=0.3662` vs DES
`0.3662-0.3667`; theory `E[V|admitted]=1.654` vs DES `1.655-1.658`).

## Scope decision

1. **Exact (rapidly-convergent series) result for `Exp(theta)` deadlines** — the core deliverable:
   `pi0`, `loss_prob`, raw moments of `U` (unconditional and admitted-conditional), sojourn moments
   for admitted customers. A genuinely new exact result, not reducible to anything already in the
   repository.
2. **Reserve, not in this epic:** general (non-exponential) deadline distributions `G` — the
   `phi(s+theta)` collapse specifically exploits `Exp(theta)`'s constant-ratio tail; for general `G`
   the weighted transform `Psi(s) = integral tail_G(u) e^{-su} dF(u)` does not reduce to a shift of
   `phi`, and would need a genuinely different (harder) numerical technique (e.g. discretizing the
   Volterra integral equation directly). Multi-server (`c>1`) admission control — same complication
   as EPIC-028's boundary structure, compounded with the workload functional equation; not
   attempted here.

Full derivation with the moment formulas: `docs/roadmaps/llm_serving_deadline_admission_control_roadmap.md`.
