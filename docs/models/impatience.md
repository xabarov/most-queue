# Systems with impatient jobs

[🇷🇺 Русская версия](impatience.ru.md) · [← Model catalog](../models.md)

![Impatient jobs diagram](../figures/impatience.png)

**In plain words:** each job has its own random "patience budget"; if its turn does not come in
time, it leaves unserved (an abandoned call in a call center, a cancelled order, a timed-out
request). The key questions for the model: what fraction of customers is lost, and how this
depends on the number of servers.

### M/M/1/D (with exponential impatience)

**Description:** A system where jobs may leave the queue if the wait is too long.

**Calculator class:** `MM1Impatience`

**Example:** See the test `test_impatience.py`

### Erlang-A (M/M/n+M)

**Description:** The multi-server abandonment model: n servers, Poisson arrivals, exponential service, and an exponential patience clock for every waiting job. Always stable; the workhorse of call-center staffing (Palm; Garnett–Mandelbaum–Reiman).

**In plain words:** the realistic call center: some callers hang up before an agent answers.
The model answers the two staffing questions at once — what fraction of customers is lost and
how many servers are needed to keep that fraction below a target
(`find_min_servers`).

**Calculator class:** `MMnImpatienceCalc` (`most_queue.theory.impatience.mmn`)
**Simulation:** `ImpatientQueueSim`

**Example:**

```python
from most_queue.theory.impatience.mmn import MMnImpatienceCalc

calc = MMnImpatienceCalc(n=3, theta=0.3)  # theta — impatience rate
calc.set_sources(1.0)
calc.set_servers(0.5)
results = calc.run()
p_abandon = calc.get_abandonment_probability()
n_needed = calc.find_min_servers(target_abandonment=0.05)
```
