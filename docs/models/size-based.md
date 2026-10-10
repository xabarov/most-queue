# Size-based scheduling disciplines

[🇷🇺 Русская версия](size-based.ru.md) · [← Model catalog](../models.md) ·
[← FIFO systems](fifo.md)

**In plain words:** these disciplines decide whom to serve based on the *size* of the job (known
or predicted), not on the order of arrival. Here is how the same jobs pass through a single
server under different disciplines:

![FCFS/SJF/SRPT discipline comparison](../figures/disciplines_timeline.png)

### M/G/1 SRPT

**Description:** Single-server M/G/1 under the **Shortest Remaining Processing Time** discipline (preemption by remaining work). Numerically: the Schrage–Miller formula (1966).

**In plain words:** the server is always busy with the job that has the least work *remaining*;
if a shorter one arrives, the current job is preempted and waits (see the diagram above, where
job A yields the server and is finished at the end). SRPT provably minimizes the mean sojourn
time among all disciplines.

**Calculator class:** `MG1SrptCalc`  
**Simulation:** `SizeBasedQsSim(discipline="SRPT")` — the job size is sampled on arrival.

**Example:**

```python
from most_queue.theory.srpt import MG1SrptCalc
from most_queue.random.distributions import H2Distribution

calc = MG1SrptCalc()
calc.set_sources(1.0)
h2 = H2Distribution.get_params_by_mean_and_cv(0.7, 1.2)
calc.set_servers(h2, "H")
results = calc.run()
```

### M/G/1 SJF (SPT)

**Description:** Non-preemptive service by the **smallest true size** (Shortest Job First / Shortest Processing Time).

**In plain words:** no preemption — when the server becomes free, the shortest job in the queue
is taken, but a service once started always runs to completion.

**Calculator class:** `MG1SjfCalc`  
**Simulation:** `SizeBasedQsSim(discipline="SJF")`

### M/G/1 PSJF

**Description:** Preemptive service by the **original** job size (different from SRPT).

**In plain words:** like SRPT, but the comparison uses the *full original* size, not the
remainder: an almost-finished long job will still yield to a newly arrived short one.

**Calculator class:** `MG1PsjfCalc`  
**Simulation:** `SizeBasedQsSim(discipline="PSJF")`

### M/G/1 SPJF (with predictions)

**Description:** Non-preemptive service by the **predicted** size \(Y\) (Mitzenmacher, 2020). The joint distribution \((X,Y)\) is specified by a predictor object (`PerfectPredictor`, `ExpNoisePredictor`, …).

**In plain words:** the true job size is unknown, but a *prediction* of it is available (e.g.
from an ML model) — we serve the jobs that are short "according to the forecast". The model
answers the question of how much of the SJF gain survives with imprecise predictions. With a
perfect predictor it reduces to SJF.

**Calculator class:** `MG1SpjfCalc`  
**Simulation:** `SizeBasedQsSim(discipline="SPJF")` + `set_predictor(...)`.

**Example:**

```python
from most_queue.theory.srpt import MG1SpjfCalc
from most_queue.theory.srpt.utils.predictor import ExpNoisePredictor

calc = MG1SpjfCalc()
calc.set_sources(0.5)
calc.set_servers(1.0, "M")
calc.set_predictor(ExpNoisePredictor())
results = calc.run()
```

#### Graceful degradation of predictions (learning-augmented scheduling)

**Description:** How does SPJF's mean response time change as predictions get noisier? The helper
`prediction_degradation_curve` sweeps the log-normal prediction noise σ and returns the SPJF mean
response bracketed by SRPT (size-aware optimum), SJF (perfect predictions) and blind FB/LAS. It
reports the **break-even noise** at which SPJF starts losing to the *blind* policy — reproducing the
central open problem of the SIGMETRICS 2025 survey "Queueing, Predictions, and LLMs" (there is no
free graceful-degradation guarantee).

```python
from most_queue.theory.srpt import prediction_degradation_curve

curve = prediction_degradation_curve(0.7, service_h2_params, "H")
# curve.spjf[i] at curve.sigmas[i]; curve.srpt / curve.sjf / curve.blind_fb references;
# curve.breakeven_sigma — noise where SPJF becomes worse than blind
```

The next three disciplines (FB, PS, LCFS-PR) complete the size-based family. How each of them
treats jobs of different sizes is computed by the library's own calculators:

![Slowdown by job size for FCFS/PS/FB/SRPT](../figures/slowdown.png)

### M/G/1 FB (Foreground-Background / LAS)

**Description:** A preemptive **blind** discipline: the server always serves the job with the least *attained* service; ties share the server equally. Job sizes need not be known.

**In plain words:** "give the newcomers a chance": a fresh job gets the server immediately and
keeps it until it catches up with the others in attained service. If short jobs are common
(decreasing hazard rate, CV > 1), FB approaches SRPT without knowing the sizes; if the service
time is nearly constant, FB loses even to FCFS. Exponential service is the boundary case: FB
coincides with PS.

**Calculator class:** `MG1FbCalc` (`most_queue.theory.srpt`)
**Simulation:** `FBSim` (`most_queue.sim.single_server_disciplines`)

**Example:**

```python
from most_queue.theory.srpt import MG1FbCalc
from most_queue.random.distributions import GammaDistribution

calc = MG1FbCalc()
calc.set_sources(1.0)
calc.set_servers(GammaDistribution.get_params_by_mean_and_cv(0.7, 1.2), "Gamma")
results = calc.run()
```

### M/G/1 PS (Processor Sharing)

![Processor Sharing diagram](../figures/ps.png)

**Description:** The server is shared equally among all jobs present (each of k jobs is served at rate 1/k). The state probabilities are geometric and insensitive to the shape of the service distribution; the conditional mean sojourn time of a job of size x is exactly x/(1−ρ).

**In plain words:** a model of a CPU, a web server, a shared channel: nobody waits "in a queue",
but everyone is slowed down by the same factor 1/(1−ρ). A perfectly fair discipline — the
baseline for comparison with SRPT/SJF (which are faster on average, but at the expense of long
jobs).

**Calculator class:** `MG1PSCalc` (`most_queue.theory.fifo.mg1_ps`)
**Simulation:** `ProcessorSharingSim` (`most_queue.sim.single_server_disciplines`)

**Example:**

```python
from most_queue.theory.fifo.mg1_ps import MG1PSCalc

calc = MG1PSCalc()
calc.set_sources(l=1.0)
calc.set_servers([0.7, 1.2])  # service time moments
results = calc.run()
slowdown = calc.get_mean_slowdown()          # 1/(1-rho)
t_x = calc.get_conditional_sojourn_mean(2.0)  # x/(1-rho)
```

#### Higher sojourn moments — where the insensitivity stops

The mean is insensitive; **nothing above it is**. Two workloads with the same mean service time
give the same `E[V(x)]` and the same queue length, but different variance — so a capacity or SLO
decision taken on the mean alone has no information about risk at all. Those higher moments are
available:

```python
calc = MG1PSCalc()
calc.set_sources(l=0.6)
calc.set_servers_from_moments([1.0, 5.0, 60.0])   # fits a shape (cv<=1 Erlang, cv>1 H2)

calc.get_conditional_sojourn_moments(x=2.0, num=4)  # exact raw moments of V(2.0)
calc.get_conditional_sojourn_var(2.0)               # exact variance
calc.get_conditional_sojourn_cv(2.0)                # its coefficient of variation
calc.get_v(3)                                       # unconditional raw moments
calc.get_conditional_sojourn_moments(2.0, 2, permanent_jobs=2)   # 2 permanent jobs sharing the CPU
```

Concretely, at `ρ = 0.6` and a job of size 2 (so `E[V] = 5` in all three cases):

| service time (mean 1) | `Var[V(2)]` | `CV[V(2)]` |
|---|---|---|
| Erlang(3), cv = 0.58 | 10.84 | 0.659 |
| exponential, cv = 1 | 11.69 | 0.684 |
| H2, cv = 2 | 12.38 | 0.704 |

Feed the moments to the [SLA layer](sla.md) to turn them into a deadline-violation probability.

**Method.** Implementation of Yashkov S.F., *Explicit formulas for the moments of the sojourn
time in the M/G/1 processor sharing queue with permanent jobs*,
[arXiv:math/0512281](https://arxiv.org/abs/math/0512281) (2005), building on his 1983 and 1987
papers. **The result is his**; this library contributes the implementation. The reciprocal of the
conditional sojourn-time transform has a clean power series in the M/G/1-FCFS waiting-time
distribution, and matching it against the transform itself gives a recursion for the moments.

Yashkov's own conclusion is that the exact expressions "involve an integration term, making an
exact computation difficult from a practical point of view", and the literature answered with
bounds and approximations. That difficulty is an artefact of the general-`B` formulation: once
the service time is phase-type — which is how this library represents a general service time
fitted from moments — every object in the chain stays phase-type, and the moments come out of a
few small matrix exponentials with no quadrature at all. The one numerical step left is the outer
integration over the service-time distribution in `get_v`/`get_v_moments`, and its accuracy is
self-checking: the first moment has to come back as the exact `b1/(1−ρ)`.

**Caveat worth stating plainly:** the variance depends on the *shape* of the service time, so
when you supply only moments and let `set_servers_from_moments` fit one, the answer is exact for
the **fitted** law, not for every law with those moments. The mean and the queue length are
insensitive and so are unaffected by the fit.

**Validation:** the insensitive `E[V(x)] = x/(1−ρ)` reproduced to machine precision through the
full recursion (which is never told it); the M/M/1-PS variance against a closed form; Yashkov's
small-job asymptotic `Var ~ x²ρ/(1−ρ)²`; his equation (3.10) by independent quadrature; and
paired comparison against simulation. See
[EPIC-077](../epics/EPIC-077-mg1-ps-sojourn-moments.md).

### M/G/n PS (Processor Sharing, n servers)

**Description:** Generalizes M/G/1 PS to `n` identical servers — while at most `n` jobs are
present each gets a dedicated server; once the count exceeds `n`, the combined capacity is
shared equally among everyone present. Same BCMP/Kelly insensitivity to the service-time
distribution shape (only the mean enters); the queue-length distribution is identical to the
classical M/M/n (Erlang-C) one — PS and FCFS differ in how individual jobs are served, not in
the aggregate occupancy process. `n=1` reduces exactly to `MG1PSCalc` above.

**Calculator class:** `MGnPSCalc` (`most_queue.theory.fifo.mgn_ps`)

```python
from most_queue.theory.fifo.mgn_ps import MGnPSCalc

calc = MGnPSCalc(n=4)
calc.set_sources(l=2.5)
calc.set_servers([1.0])  # service time moments
results = calc.run()      # results.v[0] -- mean sojourn, via Little's law
```

**Accuracy and scope:** exact queue-length distribution and mean sojourn/waiting for any `n`.
Higher conditional sojourn moments are **not** available for `n > 1`, and not for want of effort:
Yashkov's recursion (implemented for `n = 1` above) is built on the M/G/1-FCFS waiting-time
distribution and is specific to the single-server case. The insensitivity that makes the queue
length here identical to M/M/n does not extend to the conditional sojourn time, so the `n = 1`
result cannot simply be rescaled.

### M/G/1 LCFS-PR

![LCFS-PR diagram](../figures/lcfs_pr.png)

**Description:** A preemptive stack: a new job preempts the one in service, and preempted jobs later resume from the point of interruption. The sojourn time is distributed as an M/G/1 busy period — all moments follow from the Takács recursions; the state probabilities are the same geometric ones (BCMP).

**In plain words:** "last come, first served": a fresh job gets the server immediately, but risks
being preempted itself. The mean sojourn time is the same as under PS (b₁/(1−ρ), insensitive
to the distribution shape), but the variability is much larger — the tails are those of a busy
period. FCFS has a different mean: it also depends on b₂ (Pollaczek–Khinchine).

**Calculator class:** `MG1LcfsPrCalc` (`most_queue.theory.fifo.mg1_lcfs_pr`)
**Simulation:** `LcfsPRSim` (`most_queue.sim.single_server_disciplines`)

**See also:** [SLA / deadline-violation probability](sla.md) — turn these moments into a deadline-violation probability or SLO quantile.
