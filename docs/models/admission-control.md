# Deadline-aware admission control

[🇷🇺 Русская версия](admission-control.ru.md) · [← Model catalog](../models.md)

**In plain words:** a request carries an SLO — "I'm only useful if I get a response within
roughly this long." Instead of always queueing everyone FCFS and hoping ([SLA layer](sla.md)) or
reordering service by urgency ([EDF](edf.md)), this model makes a *proactive accept/reject*
decision the instant a request arrives: if its own (random) deadline is already infeasible given
the current backlog, reject it immediately — never queue it at all — and keep FCFS order among
everyone actually admitted. This is a third, genuinely distinct SLO-management mechanism from the
other two models on this catalog.

### M/M/1 with Exp(θ) deadline — exact via a convergent series

**Description:** Poisson(`λ`) arrivals, Exp(`μ`) service, FCFS order preserved. Each arriving job
draws its own relative deadline `D ~ Exp(θ)` and observes the current workload (virtual waiting
time) `U` it would face; admitted iff `D > U`, otherwise rejected outright. Solved exactly via a
classical level-crossing (Takács-style) functional equation for the workload's Laplace-Stieltjes
transform `φ(s) = E[e^{-sU}]`:

```
φ(s) = π₀ + [λ/(s+μ)] · φ(s+θ)
```

Because the deadline is exponential, this collapses into a rapidly convergent series (the system
is *always* stable, regardless of `λ`, since large-workload arrivals auto-reject):

```
φ(s) = π₀ · Σ_{n=0}^∞ λⁿ / Π_{j=0}^{n-1}(s + jθ + μ)
```

evaluated via exact truncated power-series (Taylor) arithmetic around `s=0` — not finite
differences — to get raw moments without numerical-differentiation error. `loss_prob = 1 - φ(θ)`;
moments of the sojourn time of an *admitted* job use the same technique applied to `φ(s+θ)`
(divided by `φ(θ)`, the acceptance probability) to correctly account for the fact that admission
*size-biases* the observed workload toward smaller values.

**A wrong shortcut, caught before shipping (documented so nobody re-derives it):** the first,
much simpler-looking hypothesis — since M/M/1 service is memoryless, treat the number in system
`n` as a sufficient statistic and build a state-dependent birth-death chain with
`P(admit|n) = (μ/(μ+θ))ⁿ` — is **wrong**. The formula itself is correct in isolation, but `n` is
*not* a sufficient statistic here: conditioning on "admission occurred" size-biases the workload
via the deadline's tail function, and — unlike ordinary elapsed-time conditioning — that bias is
*not* erased by service memorylessness. Caught via a careful discrete-event simulation showing a
small, seed-stable, `n`-growing discrepancy against the shortcut's prediction. See
[`docs/research/llm-serving-deadline-admission-control-2026.md`](../research/llm-serving-deadline-admission-control-2026.md)
for the full numerical evidence.

**Calculator class:** `MM1DeadlineAdmissionControlCalc` (`most_queue.theory.admission_control`)
**Simulation:** `MM1DeadlineAdmissionControlSim` (`most_queue.sim.admission_control`)

```python
from most_queue.theory.admission_control import MM1DeadlineAdmissionControlCalc

calc = MM1DeadlineAdmissionControlCalc()
calc.set_sources(l=1.2)
calc.set_servers(mu=1.0)
calc.set_deadline(theta=0.5)   # D ~ Exp(theta)
res = calc.run()
# res.v, res.w -- sojourn/wait moments of ADMITTED jobs (rejected jobs never join)
# res.loss_prob -- P(an arriving job is rejected outright)
```

### Accuracy and scope

Exact (convergent series, not an approximation) for `Exp(θ)` deadlines. **Not covered (reserve):**
general (non-exponential) deadline distributions — the series-collapsing trick specifically
exploits the exponential's constant-ratio tail; a general `G` would need a different numerical
technique (e.g. direct discretization of the underlying Volterra integral equation). Multi-server
(`c>1`) admission control is also not covered.

**See also:** [SLA / deadline-violation probability](sla.md) (passive, post-hoc, any discipline)
and [EDF](edf.md) (reorders service instead of rejecting) — the two other SLO-management
mechanisms in this catalog, each with a different accuracy/scope trade-off.
