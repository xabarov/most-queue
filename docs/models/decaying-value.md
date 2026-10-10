# Service rate control for jobs with decaying value

[🇷🇺 Русская версия](decaying-value.ru.md) · [← Model catalog](../models.md)

**In plain words:** a server works through a fixed batch of jobs, and the job currently being
worked on is worth less with every slot it fails to complete. If its value runs out the job is
thrown away for nothing. The controller cannot reorder anything and cannot turn anyone away — its
only lever is **how hard to run the server right now**, which costs money. The question is not
how long jobs wait but how fast to go, state by state.

The decay here runs **only during service**, not while a job waits. That is the distinction from
the [impatience](impatience.md) models, where the clock runs on everyone in the queue, and it is
what makes this a control problem over a shrinking batch rather than a stationary queue. The
paper's motivating cases are wireless streaming (a packet is worth less the longer the
transmitter spends on it), healthcare (delay in treatment erodes the benefit), and perishable
inventory (goods that decay in handling rather than in storage).

### The model

Discrete time. A batch of `B` identical jobs; the head-of-line job arrives there with value `V`.
In each slot the controller picks a completion probability `s` from a finite set `S`. With
probability `s` the job completes and earns `r(v)`, which depends on the value it still had;
otherwise the value drops by one, and a job reaching zero is **ejected** for nothing. Each slot
also costs `h(b)` to hold the `b` jobs still present and `c(s)` to run the server that fast. All
three functions are non-decreasing. The batch always clears, so this is a stochastic shortest
path problem and the objective is total expected cost.

### Exact optimal control

**Class:** `DecayingValueRateControl` (`most_queue.theory.decaying_value`)

```python
import numpy as np
from most_queue.theory.decaying_value import DecayingValueRateControl

control = DecayingValueRateControl(
    num_jobs=20,            # B
    initial_value=10,       # V
    rates=[0.1, 0.5, 0.9],  # S
    service_cost=lambda s: 5 * np.log(1 / (1 - s)),
    holding_cost=lambda b: b,
    reward=lambda v: v,
)

result = control.solve()
result.rate(backlog=15, value=4)   # how fast to run in that state
result.total_cost                  # expected cost of clearing the whole batch
control.monotonicity_conditions()  # how the policy behaves, without reading it
```

**Method.** Implementation of Master N. & Bambos N., *Service rate control for jobs with decaying
value*, Proc. 2015 American Control Conference (ACC), Chicago, pp. 3255–3260,
[doi:10.1109/ACC.2015.7171834](https://doi.org/10.1109/ACC.2015.7171834), preprint
[arXiv:1609.05355](https://arxiv.org/abs/1609.05355). **The model, the reformulation and the
monotonicity theorems are theirs.**

Written out, the Bellman equation is

```
J(b,v) = min_s { c(s) + h(b) + s[-r(v) + J(b-1,V)] + (1-s)[ J(b,v-1) if v > 1 else J(b-1,V) ] }
```

with `J(0,V) = 0`. The state never moves up — `b` only falls, and within a fixed `b` every failed
slot lowers `v` — so the problem is acyclic and needs no iteration to a fixed point. The paper's
Proposition 1 goes further: the whole thing telescopes. With

```
delta(b,v) = h(b) + min_s { c(s) - s[ r(v) + sigma(b,v-1) ] },   sigma(b,v) = sigma(b,v-1) + delta(b,v)
```

one gets `J(b,v) = J(b-1,V) + sigma(b,v)` and, the useful part,

```
mu(b,v) = min argmin_s { c(s) - s[ r(v) + sigma(b,v-1) ] }
```

The optimal rate depends on the state **only through the single scalar `r(v) + sigma(b,v-1)`**.
That turns a policy surface into a one-dimensional quantity and the solve into an `O(B·V·|S|)`
forward recursion. Both routes are implemented — `solve("direct")` uses the reformulation,
`solve("bellman")` the equation as written — because a reformulation that saves this much is
worth testing rather than trusting.

### The structural results

| | statement | needs |
|---|---|---|
| **Theorem 1** | `b -> mu(b,v)` is non-decreasing: more jobs waiting, run faster | nothing beyond `h` non-decreasing — holds in every admissible instance |
| **Theorem 2** | if `delta(b,v) >= -[r(v+1) - r(v)]` for all `v`, then `v -> mu(b,v)` is non-decreasing; reverse the inequality throughout and it is non-increasing | checkable **without computing the policy** |
| **Theorem 3** | with a constant reward, the test collapses to the sign of `h(b) + min_s{c(s) - s·r}` | constant `r` |

`monotonicity_conditions()` applies Theorems 2 and 3 and reports which way the policy must move;
`policy_monotonicity()` reads the same off a computed policy, so the two can be compared. The
conditions are sufficient, not necessary — they can stay silent on a policy that is monotone
anyway, but they are never wrong when they do fire.

The direction of `v -> mu` is genuinely ambiguous, and that is the paper's main qualitative point.
Reproducing its figure 1:

| panel | reward `r(v)` | rate set `S` | `b -> mu` | `v -> mu` |
|---|---|---|---|---|
| 1a | `v` | 0.1, 0.5, 0.9 | non-decreasing | non-decreasing |
| 1b | `v/10 + 25` | 0.6, 0.7, 0.8 | non-decreasing | **non-increasing** |
| 1c | `v/10 + 20` | 0.6, 0.7, 0.9 | non-decreasing | varies with `b` |
| 1d | `5 ln(1+v)` | 0.700, 0.705, 0.710 | non-decreasing | not monotone either way |

In 1a the server "gives up" on a job as its value decays; in 1b it does the opposite and "tries
harder" on a dying job. In 1c which of the two applies depends on how many other jobs are waiting,
and in 1d neither holds — `mu(5, ·)` is neither non-decreasing nor non-increasing. What never
varies is Theorem 1: in all four, more backlog means more speed.

### Simulation, and what the control is worth

**Class:** `DecayingValueSim` (`most_queue.sim.decaying_value`)

```python
from most_queue.sim.decaying_value import DecayingValueSim, compare_with_optimal, constant_rate_policy

DecayingValueSim(control, seed=0).run(replications=20_000).mean_cost   # realised cost
compare_with_optimal(control)                                          # myopic rule vs the optimum
compare_with_optimal(control, policy=constant_rate_policy(control, 0.5))
```

Any rule can also be priced exactly, without simulating, via `control.evaluate(policy)` — the same
acyclic sweep as the solver, minus the minimisation. The simulator is there to confirm that exact
number rather than to replace it.

| panel | optimal | myopic | best constant rate | worst constant rate | simulated optimum |
|---|---|---|---|---|---|
| 1a | **279.76** | 461.86 | 291.40 | 1353.64 | 280.04 ± 0.21 (1.3σ) |
| 1b | **−62.31** | −55.82 | −55.82 | −15.95 | −62.01 ± 0.25 (1.2σ) |
| 1c | **42.81** | 52.85 | 52.85 | 84.04 | 42.89 ± 0.24 (0.3σ) |
| 1d | **234.18** | 236.50 | 234.41 | 236.50 | 234.54 ± 0.31 (1.2σ) |

(Panel 1b is negative because its rewards outweigh its costs: the batch turns a profit.)

**A myopic rule can be worse than not adapting at all.** On panel 1a the obvious short-sighted
rule — pick the rate minimising this slot's cost `c(s) - s·r(v)` — costs 461.9 against an optimum
of 279.8, and is beaten by simply running at a *fixed* rate of 0.9 (291.4). Its adaptivity points
the wrong way.

The recursion says exactly why. The optimal rule ranks rates by `r(v) + sigma(b,v-1)`; the myopic
one drops `sigma`. At a backlog of 15 that discarded term is worth 15 to 26 while `r(v)` is only 1
to 10, so the myopic rule is reading a signal several times smaller than the real one. Concretely
it slows to 0.1 on low-value jobs where the optimum runs at 0.9 — abandoning a nearly-dead job
while the holding cost of everything queued behind it keeps accruing.

The condition for when this matters is worth remembering: **`sigma` grows with the backlog**, so
myopia is worst exactly when the system is most loaded. Conversely, when the reward barely varies
over its range — panel 1b has `r` between 25.1 and 26.0, a spread of 4% — dropping `sigma` hardly
changes which rate wins, and the myopic rule lands within 10% of optimal. Across 200 random
instances it was beaten by the best fixed rate in about a third of them.

**Validation:** three independent routes to the cost-to-go (the `delta`/`sigma` recursion, the
Bellman equation written out, and a value iteration that sweeps the state space in deliberately
reverse order until it converges — the first two exploit acyclicity, the third does not); the
realised cost of a slot-by-slot simulation against the predicted one, for the optimal policy and
for fixed-rate baselines; hand-computed closed forms at `B = V = 1` and `B = 1, V = 2`; the
identity `evaluate(optimal policy) == cost_to_go`; no policy beating the optimum in any state;
and, over 300 random instances, Theorem 1 never violated and the Theorem 2/3 conditions never
predicting a monotonicity the computed policy lacks. See
[EPIC-082](../epics/EPIC-082-decaying-value-rate-control.md).

**Scope.** A finite batch with **no arrivals** — the paper's proofs rely on the backlog only ever
decreasing, and it names arrivals as the significant open extension. Identical jobs, one server,
a common initial value `V`, and a finite rate set. The rate grid must exclude any point where
`c(s)` is infinite (the usual `c(s) = k·ln(1/(1-s))` blows up at `s = 1`).

### Related models in this library

- [Value-based scheduling under overload](value-scheduling.md) — the other half of the same idea:
  there value decides *which* job to run at a fixed speed, here it decides *how fast* to run at a
  fixed order.
- [Impatience / abandonment](impatience.md) — jobs with a decaying willingness to wait, where the
  clock runs during the wait rather than during service.
- [Imprecise computation](imprecise.md) — a third lever under pressure: shorten the work itself.
- [Deadline-aware admission control](admission-control.md) — the lever this model deliberately
  lacks, namely refusing work.
