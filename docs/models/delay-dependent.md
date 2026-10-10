# Delay-dependent service rates

[🇷🇺 Русская версия](delay-dependent.ru.md) · [← Model catalog](../models.md)

**In plain words:** classical queueing assumes how long a customer waited has nothing to do with
how long their service then takes. Measurements say otherwise. In hospitals a patient admitted
late stays longer ("slowdown"); in retail a customer who queued longer buys more; a server under
congestion may deliberately rush or stall. This model makes the service rate depend on the
customer's *own* experienced delay — not on the queue length, not on the workload, and not on
elapsed service time, which are the dependencies the rest of this library already covers.

The practical consequence is sharp: if you plan capacity with an ordinary M/M/c while a slowdown
effect is present, you will underestimate the delay, and the error grows exactly where it hurts
most. The original paper reports `E[W]` losing even its *convexity* in `λ` — something no
classical M/M/c can do — so extrapolating from a fitted classical model is unsafe here.

### M/M/c with a delay threshold — exact

**Description:** Poisson(`λ`) arrivals, `c` identical servers, FCFS, infinite waiting room. A
customer whose queueing delay turns out to be at most `k` is served at rate `μ₁`; one that waited
longer is served at rate `μ₂`. `μ₂ < μ₁` is the slowdown case, `μ₂ > μ₁` the speedup case, and
`μ₁ = μ₂` is an ordinary M/M/c.

Stability is governed by `μ₂` **alone**: `λ < c·μ₂`. However fast `μ₁` is, a long enough queue
puts every customer past the threshold, so `μ₁` cannot rescue an overloaded system.

```python
from most_queue.theory.delay_dependent import MMcDelayDependentServiceCalc

calc = MMcDelayDependentServiceCalc(c=3, k=5.0)   # threshold on the experienced delay
calc.set_sources(l=2.0)
calc.set_servers(mu1=0.8, mu2=0.7)                # slowdown: waiting makes service slower

calc.get_w()[0]            # exact E[W]
calc.get_p0_wait()         # P(W = 0) -- found a free server
calc.get_cdf(3.0)          # exact P(W <= 3)
calc.get_tail(10.0)        # exact P(W > 10) -- deadline-violation probability
calc.get_pdf(3.0)          # exact density
calc.get_class1_prob()     # P(W <= k): the fraction served at mu1
calc.get_v()[0]            # E[V] = E[W] + E[S]
calc.get_server_state_probs()   # pi(i, j): zero-delay states by server composition
```

**Method.** Implementation of D'Auria B., Adan I.J.B.F., Bekker R., Kulkarni V., *An M/M/c queue
with queueing-time dependent service rates*, European Journal of Operational Research
299(2):566–579, 2022, [doi:10.1016/j.ejor.2021.12.023](https://doi.org/10.1016/j.ejor.2021.12.023)
(preprint [arXiv:2107.04557](https://arxiv.org/abs/2107.04557)). **The model and the solution are
theirs**; this library contributes the implementation, written from the paper. The authors also
published their own code, which was not consulted.

The state is the *virtual queueing time* `W(t)` — a customer arriving at `t` starts service at
`t + W(t)` — together with the **server state** `S(t) = (S₁, S₂)`, the number of servers that will
be busy with class-1 and class-2 customers at the moment the new service starts. The second
component is what makes the problem Markov: `W` alone is not, because how fast it drains depends
on which classes are in service. Whenever `W(t) > 0` every server is committed, so `S₁ + S₂ = c-1`
and exactly `c` phases survive.

The balance equations give integro-differential equations, which convert into two systems of
second-order linear ODEs with constant coefficients — one below the threshold, one above — coupled
by continuity at `k`. Their solution is a **mixture of matrix exponentials**, and all the unknown
constants collapse through a recursion over the boundary states onto a single scalar fixed by
normalisation.

**Two numerical points worth knowing** (both found while implementing, neither spelled out in the
paper):

- One of the matrices whose integral the mean requires is *always* singular — the spectrum
  contains `min(0, λ - c·μ₁)` in one group and `max(0, λ - c·μ₁)` in the other, so a zero
  eigenvalue sits in one of them whenever `λ ≠ c·μ₁`. The integral itself is finite; only its
  factored form is not. It is computed here by an augmented-matrix exponential rather than by
  inverting anything.
- On the surface `μ₂ = μ₁ + λ/c` the particular solution above the threshold resonates with a
  homogeneous mode and the whole representation degenerates. The paper's Remark 2 lists several
  degenerate cases but not this one, although it is visible in its own `c = 1` formula, where
  `μ₂ - μ₁ - λ` sits in a denominator. The calculator detects it and says so; perturb a rate
  slightly — the *model* is perfectly well behaved there, only this representation of it is not.

**Validation.** Three analytical references, none of which the implementation could be fitted to:
the Erlang-C reduction at `μ₁ = μ₂` (agrees to `1e-16` for `c = 1…5`), the paper's own closed form
for `c = 1` (density to `3e-16`), and the paper's published `c = 2` example, reproduced to every
printed digit of `π(0,0)`, `π(0,1)`, `π(1,0)`. Plus independent discrete-event simulation across
speedup, slowdown and single-server cases
([`MMcDelayDependentServiceSim`](../../most_queue/sim/delay_dependent.py)).

| Simulation check (12 seeds) | `E[W]` deviation |
|---|---|
| `c=2`, the paper's own example (speedup) | +0.10σ |
| `c=3`, strong speedup `μ₁=0.3, μ₂=0.8` | −0.44σ |
| `c=1`, slowdown | +0.11σ |

See [EPIC-076](../epics/EPIC-076-mmc-delay-dependent-service.md) for the full figures, including
the control experiment that explains why the heaviest case (`ρ = 0.95`) is checked on the
distribution rather than the mean.

### How much does ignoring the dependence cost?

Fitting a classical M/M/c to the *same realised mean service time* (the most favourable thing a
practitioner could do) and comparing with the exact answer, for `c = 3`, `k = 5`, `μ₁ = 0.8`,
`μ₂ = 0.7`:

| λ | exact `E[W]` | classical fit | ratio | exact `P(W>20)` | classical `P(W>20)` |
|---|---|---|---|---|---|
| 1.0 | 0.111 | 0.111 | 1.00 | 0.00000 | 0.00000 |
| 1.8 | 1.134 | 0.986 | 1.15 | 0.00050 | 0.00000 |
| 2.0 | 4.796 | 2.421 | **1.98** | 0.05835 | 0.00147 |
| 2.05 | 12.355 | 4.209 | **2.94** | 0.22037 | 0.01566 (**14× low**) |

The error is negligible at light load and explodes exactly where capacity decisions are made. The
deadline-violation probability is the worse of the two, which matters because that is usually the
quantity an SLO is written against.

`E[W]` also stops being convex in `λ` when `μ₁` is small — second differences of `E[W]` over
`λ = 0.8…2.0` are negative at `μ₁ = 0.3` and positive at `μ₁ = 0.9`. Any sizing procedure that
assumes convexity (most do) is unsound in that regime.

### Related models in this library

- [SLA / deadline-violation probability](sla.md) — `P(W > D)` from moments, for any model. Here
  the tail is exact, so no fitting is needed.
- [Deadline-aware admission control](admission-control.md) — a *different* mechanism: rejects at
  arrival instead of changing the service rate.
- [Occupancy-dependent continuous batching](continuous-batching.md) — service rate depends on the
  current *occupancy*, not on the customer's own delay.
- [Systems with impatient jobs](impatience.md) — waiting makes customers *leave* rather than
  change their service time.
