# Systems with vacations

[🇷🇺 Русская версия](vacations.ru.md) · [← Model catalog](../models.md)

![Server life cycle in vacation models](../figures/vacations.png)

**In plain words:** the server is not always ready to work instantly. After idling it needs a
**warm-up** (machine warm-up, cold start of a server), after emptying the queue it may go into
**cooling/vacation** (power saving, scheduled maintenance), sometimes with a **delay**
(waiting to see whether another job arrives before shutting down). Jobs that arrive "at the
wrong time" have to wait longer — the models in this section compute how much longer.

### M/G/1 with multiple vacations

**Description:** The classical vacation model: having emptied the queue, the server goes on vacation; if it returns to an empty system, it immediately takes the next one. Exact solution via the Fuhrmann–Cooper decomposition: waiting time = M/G/1 waiting time + residual vacation time.

**In plain words:** a server that "keeps napping" while there is no work (power saving,
background tasks). The price jobs pay for vacations is on average half the vacation "length"
adjusted for its variability, independently of the load.

**Calculator class:** `MG1MultipleVacationsCalc` (`most_queue.theory.vacations.mg1_vacations`)
**Simulation:** `VacationQueueingSystemSimulator(1, is_multiple_vacations=True)` + `set_cold(...)`

**Example:**

```python
from most_queue.theory.vacations.mg1_vacations import MG1MultipleVacationsCalc
from most_queue.random.distributions import GammaDistribution

b = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(0.7, 1.2), 5)
vac = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(1.5, 1.2), 4)

calc = MG1MultipleVacationsCalc()
calc.set_sources(l=1.0)
calc.set_servers(b)
calc.set_vacations(vac)
results = calc.run()  # k moments of W require k+1 vacation moments
```

### M/G/1 under N-policy

![N-policy diagram](../figures/n_policy.png)

**Description:** The server switches off when the system empties and switches back on only once N jobs have accumulated; it then serves until the system is empty again. Exact solution: the extra term added to the M/G/1 waiting time is an Erlang mixture, on average (N−1)/(2λ).

**In plain words:** saving on "start-ups": the larger N, the less often the server starts, but
the longer the first accumulated jobs wait. A model for choosing the threshold N (batch start
of equipment, infrequent shuttle runs). N=1 is the ordinary M/G/1.

**Calculator class:** `MG1NPolicyCalc` (`most_queue.theory.vacations.mg1_vacations`)
**Simulation:** `NPolicyQueueSim(1, big_n=N)`

### M/G/1 with an unreliable server (breakdowns & repairs)

![Unreliable server diagram](../figures/unreliable.png)

**Description:** The server fails at Poisson rate ξ while serving; the repair time has a general distribution; the interrupted job resumes from where it stopped. Exact reduction to an M/G/1 with a "completion time" (service plus its own repairs) — Avi-Itzhak–Naor (1963).

**In plain words:** a machine that breaks under load: a job occupies the server for its service
time plus all the repairs that happen during it. The cumulants of the completion time are
computed in closed form, after which the ordinary Pollaczek–Khinchine formula applies.

**Calculator class:** `MG1UnreliableCalc` (`most_queue.theory.vacations.mg1_unreliable`)
**Simulation:** `UnreliableQueueSim` (`most_queue.sim.unreliable`)

**Example:**

```python
from most_queue.theory.vacations.mg1_unreliable import MG1UnreliableCalc
from most_queue.random.distributions import GammaDistribution

b = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(0.5, 1.2), 5)
r = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(0.4, 1.2), 5)

calc = MG1UnreliableCalc()
calc.set_sources(l=1.0)
calc.set_servers(b)
calc.set_breakdowns(xi=0.3, repair=r)
results = calc.run()
```

### M/H₂/c with warm-up

**Description:** Multi-server system with hyperexponential service and server warm-up.

**Calculator class:** `MH2nH2Warm`

**Example:**

```python
from most_queue.theory.vacations.m_h2_h2warm import MH2nH2Warm

calc = MH2nH2Warm(n=3)
# Configure the warm-up and service parameters
# (see the test test_m_h2_h2warm.py)
```

### M/M/n with H₂ cooling and H₂ warm-up

**Description:** Multi-server system with exponential service and hyperexponential cooling and warm-up.

**Calculator class:** `MMnHyperExpWarmAndCold` (`most_queue.theory.vacations.mmn_with_h2_cold_and_h2_warmup`)

**Example:** See the test `test_mmn_h2cold_h2warm.py`

### M/G/1 with warm-up

**Description:** Single-server system with warm-up.

**Calculator class:** `MG1WarmCalc`

### M/Ph/c with warm-up, delay, and vacations

**Description:** A complex system with H₂ service, H₂ warm-up, H₂ delay, and H₂ vacations.

**Calculator class:** `MGnH2ServingColdWarmDelay`

**Example:** See the test `test_mgn_with_h2_delay_cold_warm.py`
