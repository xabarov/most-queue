# Queueing Systems: Simulation & Numerical Methods 🔄

![Queue](assets/3.gif)

A Python package for simulating and analyzing queueing systems (QS) and networks.

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/xabarov/most-queue)

---
### Most-Queue 2.0 Release Notes

New unified API for both simulation and calculation classes:

```python
from most_queue.distr_utils.distribution_fitting import gamma_moments_by_mean_and_cv
from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mgn_takahasi import MGnCalc

# initialize constatants
NUM_OF_CHANNELS = 3

ARRIVAL_RATE = 1.0
SERVICE_TIME_CV = 1.2

NUM_OF_JOBS = 300_000 # number of jobs to simulate

UTILIZATION_FACTOR = 0.7

# calc service time moments using gamma distribution fitting
b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE  # average service time
b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)
gamma_params = GammaDistribution.get_params([b[0], b[1]]) # gamma distribution parameters

# run Takahasi-Takami calc method
tt = MGnCalc(n=NUM_OF_CHANNELS)

tt.set_sources(l=ARRIVAL_RATE)
tt.set_servers(b=b)

calc_results = tt.run()

# run simulation
qs = QsSim(NUM_OF_CHANNELS)

qs.set_sources(ARRIVAL_RATE, "M")
qs.set_servers(gamma_params, "Gamma")

sim_results = qs.run(NUM_OF_JOBS)

print(f"Simulation duration: {sim_results.duration:.5f} sec")
print(f"Calculation duration: {calc_results.duration:.5f} sec")

probs_print(sim_results.p, calc_results.p, 10)
times_print(sim_results.v, calc_results.v, False)
times_print(sim_results.w, calc_results.w, True)

```

See more examples in [tests](https://github.com/xabarov/most-queue/blob/main/tests/) and [tutorials](https://github.com/xabarov/most-queue/tree/main/tutorials/) folders.



---

### 🔍 Key Features

- **Simulation**: Model various types of queueing systems and networks.
- **Numerical Methods**: Solve steady-state problems in queueing theory.
- **Performance Metrics**: Analyze waiting times, sojourn times, load factors, and more.

---

### 📌 Use Cases

- **Cloud Computing**: Model infrastructure scalability and performance.
- **Call Centers**: Optimize staffing and customer wait times.
- **Transportation**: Improve traffic flow and logistics.
- **Network Traffic**: Analyze and predict data flow patterns.

---

### 📦 Installation
```bash
  pip install most-queue
```
Or install from the repository:

```bash
  pip install -e .
```

---

## 📚 Project Overview

Most_queue consists of two main parts:
 - **most_queue.theory** contains programs that implement methods for calculating queueing theory models.
 - **most_queue.sim** contains simulation programs.

## 🧪 Example Use Cases

### FIFO Queueing Systems
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | Ek/D/c           |  Numerical calculation of a multi-channel system Ek/D/n   | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_ek_d_n.py) | |
| 2.  | GI/M/1          |  Solving for QS GI/M/1     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_gi_m_1_calc.py) | |
| 3.  | GI/M/c          |  Solving for QS GI/M/c      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_gi_m_n_calc.py) | |
| 4.  | M/D/c           |  Solving for QS M/D/c        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_d_n_calc.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/queue_sim.ipynb)  |
| 5.  | M/G/1           |  Solving for QS M/G/1        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mg1_calc.py) | |
| 6.  | M/H<SUB>2</SUB>/c         |  Numerical calculation of QS M/H<SUB>2</SUB>/c by the Takahashi-Takami method with complex parameters when approximating the serving time by the H<SUB>2</SUB>-distribution    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_tt.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/mgn_takahasi_takami.ipynb) |
| 7.  | M/M/c/r         |  Solving for QS M/M/c/r        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_test.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/queue_sim.ipynb) |


### Queueing Systems with Priorities

| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/Ph/c/PR     |  Numerical calculation of QS M/Ph/c with 2 classes and PR - priority. Based on the approximation of busy periods            | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_ph_n_prty.py) | |
| 2.  | M/M/c/PR           |  Numerical calculation of QS M/M/c with 2 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of the busy period by the Cox distribution      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn_prty_busy_approx.py) | |
| 3.  | M/G/1/PR           |  Calculating QS with preemtive priorities (single-channel).     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |
| 4.  | M/G/1/NP           |  Calculating QS with non-preemtive priorities (single-channel).     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |
| 5.  | M/G/c/Priority           | Calculating QS with NP and PR (multi-channel) by method of relation      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |

### Queueing Systems with Vacations
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/H<SUB>2</SUB>/c          |  Numerical calculation of the M/H<SUB>2</SUB>/c system with H<SUB>2</SUB>-warming using the Takahashi-Takami method.      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_h2_h2warm.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/m_h2_h2warm.ipynb)| |
| 2.  | M/G/1           | Solving for QS M/G/1 with warm-up       |  | |
| 3.  | M/Ph/c         |  Multichannel queuing system with H<SUB>2</SUB>-serving time, H<SUB>2</SUB>-warm-up, H<SUB>2</SUB>-cold delay and H<SUB>2</SUB>-cold (vacations). The system uses complex parameters, which allows you to calculate systems with arbitrary serving, warm-up, cold-delay and cold variation coefficients | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_with_h2_delay_cold_warm.py) | |
| 4.  | M/M/c          |  Multichannel queuing system with exp serving time, H<SUB>2</SUB>-warm-up and H<SUB>2</SUB>-cold (vacations). The system uses complex parameters, which allows to calculate systems with arbitrary warm-up and cold variation coefficients    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn_h2cold_h2warm.py) | |

### Queueing Systems with Negative arrivals
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/G/1 RCS         |  Exact calculation of sojourn time for M/G/1 with RCS (remove customer from service) negative arrivals. Service time approximates by H<SUB>2</SUB> or Gamma distribution     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mg1_rcs.py) | |
| 2.  | M/G/c RCS         |  Numerical calculation of M/G/c with RCS negative arrivals. Service time approximates by H<SUB>2</SUB> distribution     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_rcs.py) | |
| 3.  | M/G/c disaster         |  Numerical calculation of M/G/c with disaster (remove all customer from service and queue by negative arrival). Service time approximates by H<SUB>2</SUB> distribution     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_disaster.py) | |


### Fork-Join Queueing Systems

| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | M/M/c/Fork-Join       |  Solving for Fork-Join queueing system      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_sim.py) |  |
| 2. | M/G/c/Split-Join       |  Solving for Split-Join queueing system      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_sim.py) |  |


### Others
| #   | Kendall Notations |  Description      | Example | Tutorial |
|-----|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M<sup>x</sup>/M/1          |  Solving for the of M<sup>x</sup>/M/1 QS with batch arrival    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_batch.py) | |
| 2.  | M/M/1/D         |  Solving for M/M/1 with exponential impatience     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_impatience.py) | |
| 3. | M/M/1/N          |  Solving for the Engset model for M/M/1 with a finite number of sources.     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_engset.py) | |
| 4.  | Queuing Network |  Numerical calculation of queuing network     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_network_no_prty.py) |  |
| 5.  | Queuing Network with Priorities  |  Numerical calculation of queuing network with priorities in nodes      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_network_sim_prty.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/network_with_priorities.ipynb) |
| 6.  | Queuing Network Optimization  | Optimization of queuing network transition matrix     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_network_opt.py) |  |


---

### 🔍 Search & Indexing Keywords
- Queueing theory
- Simulation
- Numerical methods
- Queueing networks
- Performance analysis
- Cloud computing
- Call center optimization
- Transportation systems
- Network traffic
- Python package

---

### 📁 Examples & Tutorials
- Look [tests](https://github.com/xabarov/most-queue/tree/main/tests) for examples with comparison of theoretical and simulation results.
- Look [tutorials](https://github.com/xabarov/most-queue/tree/main/tutorials) for jupyter tutorials

---

### 👥 Contributing

Contributions are welcome!

- Open an [issue](https://github.com/xabarov/most-queue/issues) for bugs or suggestions.
- Submit a pull request for feature enhancements.
- Contact me at xabarov1985@gmail.com for questions.
