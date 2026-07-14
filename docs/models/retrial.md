# Retrial queues

[🇷🇺 Русская версия](retrial.ru.md) · [← Model catalog](../models.md)

![Retrial queue diagram](../figures/retrial.png)

**In plain words:** there is no waiting room at all: a job that finds the server busy goes to
an invisible **orbit** and retries after a random time (a caller who got a busy tone and dials
again). Compared with an ordinary queue, the server sometimes idles while jobs sit in orbit —
so waits are longer, and the slower the retries (small γ), the worse it gets.

### M/M/1 retrial

**Description:** Classical (linear) retrial policy: each of j orbiting jobs retries at rate γ. Solved exactly by adaptive truncation of the level-dependent chain.

**Calculator class:** `MM1RetrialCalc` (`most_queue.theory.retrial`)
**Simulation:** `RetrialQueueSim` (`most_queue.sim.retrial`)

**Example:**

```python
from most_queue.theory.retrial import MM1RetrialCalc

calc = MM1RetrialCalc(gamma=0.7)
calc.set_sources(1.0)
calc.set_servers(1.43)
results = calc.run()
orbit_mean = calc.get_orbit_mean()
```

### M/G/1 retrial

**Description:** General service times; mean orbit size and waiting in closed form (Falin–Templeton): E[N_o] = λ²b₂/(2(1−ρ)) + λρ/(γ(1−ρ)). As γ→∞ the ordinary M/G/1 is recovered.

**In plain words:** the retrial penalty is an additive term on top of the ordinary M/G/1
queue length — it grows as retries become lazier. The formula was verified in this library
against the exact M/M/1-retrial solution and against simulation.

**Calculator class:** `MG1RetrialCalc` (`most_queue.theory.retrial`)
**Simulation:** `RetrialQueueSim`
