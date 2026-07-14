# Matrix-analytic models (MAP/PH)

[🇷🇺 Русская версия](map-ph.ru.md) · [← Model catalog](../models.md)

![MAP correlated arrivals](../figures/map_arrivals.png)

**In plain words:** real traffic is bursty — a short interarrival gap tends to be followed by
another short one. The Markovian Arrival Process (MAP) captures this correlation with a pair of
matrices (D₀, D₁); phase-type (PH) distributions play the same role for service times. The
MAP/PH/1 queue is solved **exactly** by the matrix-geometric (QBD) method — and the answer can
differ from a renewal model with identical mean/CV by several times
(see [`tutorials/map_ph_correlation.ipynb`](../../tutorials/map_ph_correlation.ipynb)).

New to PH and MAP? They are introduced step by step — with diagrams showing how Exp, Erlang,
H₂ and Coxian are all special cases of PH, and how the MMPP works — in the
[distributions reference](../distributions.md#phase-type-distributions-ph).

### MAP/PH/1

**Description:** Correlated arrivals, phase-type service, one server. Stationary distribution via the QBD logarithmic-reduction method (Latouche–Ramaswami); waiting-time moments by differentiating the arriving-job LST.

**Calculator class:** `MapPh1Calc` (`most_queue.theory.matrix.map_ph1`)
**Simulation:** `QsSim` with `set_sources(map_params, "MAP")` and `set_servers(ph_params, "PH")`

**Example:**

```python
import numpy as np
from most_queue.random.distributions import H2Distribution
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.theory.matrix.map_ph1 import MapPh1Calc

mmpp = MAP.mmpp([2.0, 0.4], np.array([[-0.2, 0.2], [0.3, -0.3]]))  # bursty arrivals
# any PH: from_exp / from_erlang / from_h2 / from_cox, or a custom (alpha, T)
service = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(0.5, 1.2))

calc = MapPh1Calc()
calc.set_sources(mmpp)
calc.set_servers(service)
results = calc.run()
```

### M/PH/1 and PH/PH/1

**Description:** Special cases via the same QBD engine: `MPh1Calc` (Poisson arrivals) reproduces Pollaczek–Khinchine exactly; `PhPh1Calc` (renewal PH arrivals) covers GI/PH-type single-server systems.

**Calculator classes:** `MPh1Calc`, `PhPh1Calc` (`most_queue.theory.matrix.map_ph1`)

### MAP/M/c

**Description:** Correlated arrivals with **c** exponential servers, solved as a level-dependent-boundary QBD (levels 0..c-1 encode the number of busy servers, homogeneous from level c on). The realistic model of a call center or data center fed by bursty traffic.

**In plain words:** the multi-server companion of MAP/PH/1 — Erlang C, but with the arrival
burstiness that Erlang C ignores. A one-phase (Poisson) MAP reproduces Erlang C exactly; a
bursty MAP with the same rate produces a much longer wait.

**Calculator class:** `MapMMcCalc` (`most_queue.theory.matrix.map_mmc`)
**Simulation:** `QsSim(c)` with `set_sources(map_params, "MAP")` and `set_servers(mu, "M")`

**Example:**

```python
import numpy as np
from most_queue.random.map_ph import MAP
from most_queue.theory.matrix.map_mmc import MapMMcCalc

mmpp = MAP.mmpp([2.5, 0.5], np.array([[-0.15, 0.15], [0.25, -0.25]]))  # bursty arrivals

calc = MapMMcCalc(n=3)  # 3 servers
calc.set_sources(mmpp)
calc.set_servers(mu=1.0)
results = calc.run()  # state probabilities + Little-law means
```

### MAP/PH/c

**Description:** The most general single-station model here: correlated MAP arrivals, phase-type service and c servers. Solved as a QBD whose phase is the MAP phase times the multiset of the busy servers' service phases.

**In plain words:** combines everything — bursty arrivals *and* variable (phase-type) service
*and* multiple servers. Reduces exactly to MAP/M/c (exponential service), MAP/PH/1 (one server)
and the Takahashi–Takami M/H₂/c (Poisson arrivals). The service-phase space grows
combinatorially, so keep the PH order and c modest (e.g. 2-phase service, c ≤ 6).

**Calculator class:** `MapPhCCalc` (`most_queue.theory.matrix.map_phc`)
**Simulation:** `QsSim(c)` with `set_sources(map_params, "MAP")` and `set_servers(ph_params, "PH")`

### BMAP/M/1

**Description:** **Batch** Markovian arrivals (jobs arrive in correlated batches) served by a single exponential server. Because a batch raises the level by more than one, this is an M/G/1-type chain; it is solved here by robust level truncation.

**In plain words:** traffic that arrives in bursts of several jobs at once (packet trains, bulk
orders), with the batch process itself Markov-modulated. Reduces exactly to M^[X]/M/1
(Poisson batches) and to MAP/M/1 (batches of size one).

**Calculator class:** `BmapM1Calc` (`most_queue.theory.matrix.bmap_m1`)

**Example:**

```python
from most_queue.random.map_ph import bmap_poisson_batch
from most_queue.theory.matrix.bmap_m1 import BmapM1Calc

# batches arrive Poisson(rate=0.5); size 1..5 with these probabilities
bmap = bmap_poisson_batch(0.5, [0.2, 0.3, 0.1, 0.2, 0.2])

calc = BmapM1Calc()
calc.set_sources(bmap)
calc.set_servers(mu=2.5)
results = calc.run()
```

### BMAP/PH/1

**Description:** Batch Markovian arrivals with **phase-type** service — the general-service member of the BMAP family. Solved by level truncation over (level, BMAP phase, service phase). A general (non-PH) service time is handled by first fitting a PH distribution to its moments.

**In plain words:** bursty batch traffic meeting variable (not just exponential) service.
Reduces exactly to BMAP/M/1 (exponential service) and to MAP/PH/1 (batches of size one).

**Calculator class:** `BmapPh1Calc` (`most_queue.theory.matrix.bmap_ph1`)
**Simulation:** `BmapPh1Sim` (`most_queue.sim.bmap`)

**Example:**

```python
from most_queue.random.map_ph import bmap_poisson_batch, PHDistribution
from most_queue.random.distributions import H2Distribution
from most_queue.theory.matrix.bmap_ph1 import BmapPh1Calc

bmap = bmap_poisson_batch(0.4, [0.2, 0.3, 0.1, 0.2, 0.2])
service = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(0.5, 1.3))

calc = BmapPh1Calc()
calc.set_sources(bmap)
calc.set_servers(service)
results = calc.run()
```
