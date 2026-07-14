# Fork-Join systems

[🇷🇺 Русская версия](fork-join.ru.md) · [← Model catalog](../models.md)

![Fork-Join diagram](../figures/fork_join.png)

**In plain words:** on arrival, a job splits (fork) into several parts, the parts are served
in parallel on different servers, and the result is ready only when all parts are collected
(join). This is how parallel computing, RAID arrays, and distributed queries (map-reduce) work.
The response time is determined by the *slowest* part — which is why the mean sojourn time
grows with the number of parts even for the same total amount of work.

### M/M/c Fork-Join

**Description:** A system where a job splits into several parts served in parallel and then rejoined.

**Calculator class:** `ForkJoinMarkovianCalc`

**Example:**

```python
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc

calc = ForkJoinMarkovianCalc(n=5, k=2)  # 5 servers, 2 required
calc.set_sources(l=1.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/G/c Split-Join

**Description:** A Split-Join system with a general service time distribution.

**In plain words:** the strict variant of Fork-Join — the next job does not begin service until
the previous one has been fully reassembled (a synchronous batch pipeline).

**Calculator class:** `SplitJoinCalc`

**Example:** See the test `test_fj_sim.py`
