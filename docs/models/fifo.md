# FIFO systems (First In First Out discipline)

[🇷🇺 Русская версия](fifo.ru.md) · [← Model catalog](../models.md)

![M/M/c diagram](../figures/fifo_mmn.png)

**In plain words:** jobs (customers, tasks, packets) arrive at random moments, join a common
queue, and are served in order of arrival by the first server to become free. The only difference
between the models in this section is how "random" the arrivals and service are:
from fully memoryless M/M/c to general distributions approximated by a hyperexponential
(the Takahashi–Takami method).

### M/M/c

**Description:** Multi-server system with Poisson arrivals and exponential service.

**In plain words:** the "ideal call center" — both the gaps between calls and the call durations
are random and independent of the past. The simplest multi-server model; all characteristics
are computed exactly, and it is the right starting point for any analysis.

**Calculator class:** `MMnrCalc`

**Example:**

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

calc = MMnrCalc(n=3)  # 3 servers
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/M/c/r

**Description:** M/M/c with a finite queue (at most r waiting positions).

**In plain words:** same as M/M/c, but the "waiting room" has only r seats: a job that arrives
to a full system is rejected and lost. A model for systems with a finite buffer (telephony,
network equipment).

**Calculator class:** `MMnrCalc`

**Example:**

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

calc = MMnrCalc(n=3, r=20)  # 3 servers, queue capacity 20
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/M/n/0 — Erlang B (loss system)

![Erlang B loss system](../figures/loss.png)

**Description:** The classical loss system: there is no queue, and a job that finds all n servers busy is lost. The blocking probability is given by the Erlang B formula (a numerically stable recursion).

**In plain words:** how many phone lines (hospital beds, parking spots) are needed to lose no
more than a given fraction of customers. By Sevastyanov's theorem the blocking probability does
not depend on the shape of the service distribution — only on its mean — so the result also
holds for M/G/n/0.

**Calculator class:** `ErlangBCalc` (`most_queue.theory.fifo.erlang`)

**Example:**

```python
from most_queue.theory.fifo.erlang import ErlangBCalc

calc = ErlangBCalc(n=3)
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
blocking = calc.get_blocking_probability()
```

### M/M/n — Erlang C (waiting system)

**Description:** Multi-server system with an infinite queue. The waiting probability is given by the Erlang C formula; the waiting time moments are available in closed form.

**In plain words:** the basic staffing model: what is the probability that a customer has to wait,
and for how long. The wait is either zero (a server is free) or exponential — which is why all
the moments follow from a single formula.

**Calculator class:** `ErlangCCalc` (`most_queue.theory.fifo.erlang`)

**Example:**

```python
from most_queue.theory.fifo.erlang import ErlangCCalc

calc = ErlangCCalc(n=3)
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
p_wait = calc.get_waiting_probability()
```

### M/G/∞ (infinitely many servers)

![M/G/∞ diagram](../figures/m_g_inf.png)

**Description:** Every job instantly gets its own server: there is no waiting, and the number of busy servers has a Poisson distribution with mean λ·b₁, regardless of the shape of the service distribution (insensitivity).

**In plain words:** a model of an "abundant" resource — active sessions, calls in a large network,
cars on a highway. It answers the question "how much of the resource is actually in use at once"
and serves as a building block for staffing approximations.

**Calculator class:** `MGInfCalc` (`most_queue.theory.fifo.m_g_inf`)

**Example:**

```python
from most_queue.theory.fifo.m_g_inf import MGInfCalc
from most_queue.random.distributions import GammaDistribution

calc = MGInfCalc()
calc.set_sources(l=1.0)

gamma_params = GammaDistribution.get_params_by_mean_and_cv(2.0, 1.2)
b = GammaDistribution.calc_theory_moments(gamma_params, 4)
calc.set_servers(b=b)

results = calc.run()
busy_mean = calc.get_offered_load()  # mean number of busy servers
```

### M/G/1

**Description:** Single-server system with Poisson arrivals and a general service time distribution.

**In plain words:** one server, arbitrary service time (specified via raw moments). The classical
Pollaczek–Khinchine setting: the queue grows not only with the load but also with the *variability*
of the service time — for the same mean, a system with rare "heavy" jobs waits far longer than
one with identical jobs.

**Calculator class:** `MG1Calc`

**Example:**

```python
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution

calc = MG1Calc()
calc.set_sources(l=0.5)

h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=0.8)
b = H2Distribution.calc_theory_moments(h2_params, 5)
calc.set_servers(b)

results = calc.run()
```

The next four models are **size-based disciplines**: the server decides whom to serve based on
the *size* of the job (known or predicted), not on the order of arrival. Here is how the same
jobs pass through a single server under different disciplines:

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
jobs). Only the means are computed for now (higher moments — Yashkov/Ott methods — are deferred).

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

### M/G/1 LCFS-PR

![LCFS-PR diagram](../figures/lcfs_pr.png)

**Description:** A preemptive stack: a new job preempts the one in service, and preempted jobs later resume from the point of interruption. The sojourn time is distributed as an M/G/1 busy period — all moments follow from the Takács recursions; the state probabilities are the same geometric ones (BCMP).

**In plain words:** "last come, first served": a fresh job gets the server immediately, but risks
being preempted itself. The mean sojourn time is the same as under PS (b₁/(1−ρ), insensitive
to the distribution shape), but the variability is much larger — the tails are those of a busy
period. FCFS has a different mean: it also depends on b₂ (Pollaczek–Khinchine).

**Calculator class:** `MG1LcfsPrCalc` (`most_queue.theory.fifo.mg1_lcfs_pr`)
**Simulation:** `LcfsPRSim` (`most_queue.sim.single_server_disciplines`)

### GI/M/1

**Description:** Single-server system with general arrivals and exponential service.

**In plain words:** the mirror image of M/G/1 — now the "general" side is not the service but
the arrival process: the interarrival times may have any distribution (specified via moments),
while service is exponential.

**Calculator class:** `GIM1Calc`

**Example:**

```python
from most_queue.theory.fifo.gi_m_1 import GIM1Calc
from most_queue.random.distributions import GammaDistribution

calc = GIM1Calc()

gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
a = GammaDistribution.calc_theory_moments(gamma_params)
calc.set_sources(a)

calc.set_servers(mu=0.6)
results = calc.run()
```

### GI/M/c

**Description:** Multi-server system with general arrivals and exponential service.

**Calculator class:** `GiMn`

**Example:**

```python
from most_queue.theory.fifo.gi_m_n import GiMn
from most_queue.random.distributions import GammaDistribution

calc = GiMn(n=3)  # 3 servers

gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
a = GammaDistribution.calc_theory_moments(gamma_params)
calc.set_sources(a)

calc.set_servers(mu=0.6)
results = calc.run()
```

### GI/G/1 and GI/G/m (two-moment approximations)

**Description:** Approximate computation of the mean waiting time from the first two moments of the arrival and service processes: Kingman (upper bound), Krämer–Langenbach-Belz for GI/G/1 (exact for M/G/1), Allen–Cunneen for GI/G/m (exact for M/M/m).

**In plain words:** "back-of-the-napkin formulas" for capacity planning: for when only the means
and variabilities are known and no exact solution exists. Only the first moment is returned
(this is an approximation, not an exact solution); the typical KLB error is a few percent. The
Kimura formula (interpolation over D/M/s, M/D/s, M/M/s) is deferred — it requires exact D/M/s
solutions.

**Calculator classes:** `GIG1ApproxCalc`, `GIGmApproxCalc` (`most_queue.theory.fifo.gi_g_approx`)

**Example:**

```python
from most_queue.theory.fifo.gi_g_approx import GIG1ApproxCalc
from most_queue.random.distributions import GammaDistribution

a_params = GammaDistribution.get_params_by_mean_and_cv(1.0, 0.56)
b_params = GammaDistribution.get_params_by_mean_and_cv(0.7, 1.2)

calc = GIG1ApproxCalc()  # or GIG1ApproxCalc(approximation="kingman")
calc.set_sources(GammaDistribution.calc_theory_moments(a_params, 4))
calc.set_servers(GammaDistribution.calc_theory_moments(b_params, 4))
results = calc.run()  # results.w — [w1], first moment only
```

### H₂/M/c (Takahashi–Takami method)

**Description:** Multi-server system with hyperexponential arrivals (H₂) and exponential service (M). Uses the simplified algorithm of §7.6.1 (formulas for z_j, x_j, t_{j,i}, level 0).

**In plain words:** H₂ (a "mixture of two exponentials") is a universal building block: by fitting
its parameters to the mean and coefficient of variation, one can approximate almost any real
distribution (for CV < 1 — with complex-valued parameters). The Takahashi–Takami method is an
iterative numerical algorithm that solves such multi-server phase-type models exactly.

**Calculator class:** `H2MnCalc`

**Example:**

```python
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.random.distributions import H2Distribution

calc = H2MnCalc(n=3)

h2_params = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2, is_clx=True)  # mean, cv
#
# For CV<1 use the complex fit: is_clx=True.
# Important: the `QsSim` simulator cannot generate H2 with complex parameters,
# so comparison with simulation is only possible when the parameters are real-valued.
calc.set_sources(h2_params)

calc.set_servers(b=2.0)  # mean service time

results = calc.run()
```

### H₂/H₂/c (Takahashi–Takami method)

**Description:** Multi-server system with hyperexponential arrivals and hyperexponential service. Uses the algorithm of §7.6.2 (CH7).

**Calculator class:** `HkHkNCalc`

**Example:**

```python
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc
from most_queue.random.distributions import H2Distribution

calc = HkHkNCalc(n=3, k=2)

h2_arr = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2)
# For CV<1 use the complex fit: is_clx=True (the parameters may then be complex).
calc.set_sources(u=[h2_arr.p1, 1 - h2_arr.p1], lam=[h2_arr.mu1, h2_arr.mu2])

h2_srv = H2Distribution.get_params_by_mean_and_cv(2.0, 1.2)
calc.set_servers(y=[h2_srv.p1, 1 - h2_srv.p1], mu=[h2_srv.mu1, h2_srv.mu2])

results = calc.run()
```

**Note on CV<1:** for \(CV<1\) the H₂ approximation uses a *complex fit* (complex-valued parameters).
The `QsSim` simulator does not generate H₂ with complex parameters, so for validation it is
convenient to compare the calculation (H₂ complex fit) with a simulation of an equivalent
`Gamma` model matched by mean/CV (see the tests in `tests/test_tt_vs_sim_gamma_cvl1.py`).

### M/D/c

**Description:** Multi-server system with Poisson arrivals and deterministic service time.

**In plain words:** service takes exactly the same time for every job (an assembly line, a
machine cycle). Zero service variability is the best case for a queue: at the same load the
wait is half as long as in M/M/c.

**Calculator class:** `MDn`

**Example:**

```python
from most_queue.theory.fifo.m_d_n import MDn

calc = MDn(n=3)
calc.set_sources(l=2.0)
calc.set_servers(b=1.0)  # constant service time
results = calc.run()
```

### Eₖ/D/c

**Description:** Multi-server system with Erlang-distributed interarrival times and deterministic service.

**In plain words:** an Erlang arrival stream is more "rhythmic" than a Poisson one (jobs arrive
more regularly), and the service time is constant. A model of nearly deterministic production
lines.

**Calculator class:** `EkDn`

**Example:**

```python
from most_queue.theory.fifo.ek_d_n import EkDn

calc = EkDn(n=3, k=2)  # 3 servers, Erlang of order 2
calc.set_sources(l=2.0)
calc.set_servers(b=1.0)
results = calc.run()
```

### M/H₂/c (Takahashi–Takami method)

**Description:** Multi-server system with Poisson arrivals and hyperexponential service. Uses the Takahashi–Takami numerical method with complex parameters.

**Calculator class:** `MGnCalc`

**Example:**

```python
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.random.distributions import H2Distribution

calc = MGnCalc(n=5)

calc.set_sources(l=2.0)

h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=1.2, is_clx=True)
calc.set_servers(h2_params)

results = calc.run()
```
