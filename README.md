# Queueing Systems: Simulation & Numerical Methods

![Queue](assets/queue_long.png)

A Python package for simulating and analyzing queueing systems (QS) and networks. 

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/xabarov/most-queue)


## About
This repository focuses on solving steady-state problems in queueing theory.

Key Features:
- Simulate various types of queueing systems and networks.
- Numerical methods for solving queueing theory problems.
- Analyze system performance metrics such as waiting times, soujourn times, load factor and etc.


## Use Cases
- Modeling cloud computing infrastructure.
- Designing efficient call centers.
- Optimizing transportation systems.
- Network traffic analysis.

## Contributing
Contributions are welcome! If you find any issues or have suggestions, please open an [issue](https://github.com/xabarov/most-queue/issues). Your pull requests are also appreciated. You can write me at [xabarov1985@gmail.com](mailto:xabarov1985@gmail.com) 

## Roadmap
- Expand support for more queueing models.
- Implement advanced numerical solution methods.

---

## Installation
Install most-queue with pip
```bash
  pip install most-queue
```

### Description of the Project

Most_queue consists of two main parts:
 - **most_queue.theory** contains programs that implement methods for calculating queuing theory models. 
 - **most_queue.sim** contains simulation programs. 
### Package most_queue.theory

Package consist of following submodules:

#### Batch
| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M<sup>x</sup>/M/1          | batch.batch_mm1                         | Solving for the of M<sup>x</sup>/M/1 QS with batch arrival    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_batch.py) | |

#### Closed
| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | M/M/1/N          | closed.engset_model                      | Solving for the Engset model for M/M/1 with a finite number of sources.     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_engset.py) | |


#### Fork-Join

| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | M/M/c/Fork-Join       | fork_join.fj_calc                           | Solving for Fork-Join queueing system      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_im.py) |  |


#### Priority Queueing Systems

| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/Ph/c/PR     | priority.m_ph_n_prty                       | Numerical calculation of QS M/Ph/c with 2 classes and PR - priority. Based on the approximation of busy periods            | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_ph_n_prty.py) | |
| 2.  | M/M/c/PR           | priority.mmn_prty_pnz_approx               | Numerical calculation of QS M/M/c with 2 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of the busy period by the Cox distribution      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn_prty_pnz_approx.py) | |
| 3.  | M/M/c/PR           | priority.mmn3_pnz_cox_approx               | Numerical calculation of QS M/M/c with 3 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of busy period by the Cox distribution     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn3_pnz_cox_approx.py) | |
| 4.  | M/G/1           | priority.priority_calc                     | A set of functions for calculating QS with priorities (single-channel, multi-channel). The multichannel calculation is carried out by the method of relation      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) |[link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |

#### FIFO Queueing Systems
| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1. | Ek/D/c           | fifo.ek_d_n_calc                       | Numerical calculation of a multi-channel system Ek/D/n   | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_ek_d_n.py) | |
| 2.  | GI/M/1          | fifo.gi_m_1_calc                       | Solving for QS GI/M/1     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_gi_m_1.py) | |
| 3.  | GI/M/c          | fifo.gi_m_n_calc                       | Solving for QS GI/M/c      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_gi_m_n.py) | |
| 4.  | M/D/c           | fifo.m_d_n_calc                        | Solving for QS M/D/c        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_d_n_calc.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/queue_sim.ipynb)  |
| 5.  | M/G/1           | fifo.mg1_calc                          | Solving for QS M/G/1        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mg1_calc.py) | |
| 6.  | M/H2/c         | fifo.mgn_tt                            | Numerical calculation of QS M/H2/c by the Takahashi-Takami method with complex parameters when approximating the serving time by the H2 distribution    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_tt.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/mgn_takahasi_takami.ipynb) |
| 7.  | M/M/c/r         | fifo.mmnr_calc                         | Solving for QS M/M/c/r        | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_test.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/queue_sim.ipynb) |


#### Networks

| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | General Network | network_calc                      | Numerical calculation of queuing network with priorities in nodes      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_network_im_prty.py) | |
| 2.  | -               | networks.network_viewer                 | Utility to view network structure        | | |

#### Vacations
| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/H2/c          | vacations.m_h2_h2warm                      | Numerical calculation of the M/H2/c system with H2-warming using the Takahasi-Takagi method.      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_m_h2_h2warm.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/m_h2_h2warm.ipynb)| |
| 2.  | M/G/1           | vacations.mg1_warm_calc                     | Solving for QS M/G/1 with "warm-up       |  | |
| 3.  | M/Ph/c         | vacations.mgn_with_h2_delay_cold_warm    | Multichannel queuing system with H2 serving time, H2 warm-up, H2 cold delay and H2 cold (vacations). The system uses complex parameters, which allows you to calculate systems with arbitrary serving, warm-up, cold-delay and cold variation coefficients | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mgn_with_h2_delay_cold_warm.py) | |
| 4.  | M/M/c          | vacations.mmn_with_h2_cold_h2_warmup  | Multichannel queuing system with exp serving time, H2 warm-up and H2 cold (vacations). The system uses complex parameters, which allows to calculate systems with arbitrary warm-up and cold variation coefficients    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_mmn_h2cold_h2warm.py) | |

#### Impatience
| #   | Kendall Notations | Package Name                      | Description      | Example | Tutorial |
|-----|-------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M/M/1/D         | impatience.impatience_mm1                   | Solving for M/M/1 with exponential impatience     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_impatience.py) | |

### Package most_queue.sim

| #   | Kendall Notations | Package Name               | Description    | Example | Tutorial   |
|-----|-------------------|----------------------------|-------------------------------------------------------------------------------------------------|-------------------|-------------------|
| 1.  | M<sup>x</sup>/M/c          | batch_sim                   | Simulation of QS in a batch system    | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_batch.py) |  |
| 2.  | M/G/c/Fork-Join       | fj_delta_sim               | Simulation of QS fork-join with a delay in the start of processing between channels             | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_delta.py) |  |
| 3.  | M/M/c           | fj_sim                     | Simulation of QS with fork-join process     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_fj_im.py) |  |
| 4.  | -           | flow_sum_sim               | Simulation of flow summation      |  |  |
| 5.  | GI/G/c      | impatient_queue_sim           | Simulation of QS GI/G/m/n with impatience      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_impatience.py) |  |
| 6.  | QN     | priority_network           | Simulation of queuing network with priorities in nodes     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_network_im_prty.py) |  |
| 7.  | GI/G/c/n/Pri   | priority_queue_sim         | Simulation of QS GI/G/c/n with priorities   | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_prty.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/priority_queue.ipynb)  |
| 8.  | GI/G/c/n       | qs_sim                     | Simulation of QS GI/G/c/n     | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_qs_sim_test.py) | [link](https://github.com/xabarov/most-queue/blob/main/tutorials/mgn_takahasi_takami.ipynb) |
| 9.  | GI/G/c/n/N     | queue_finite_source_sim | Simulation of QS GI/G/c/n with finite sources      | [link](https://github.com/xabarov/most-queue/blob/main/tests/test_engset.py) |  |


### Usage
- Look [here](https://github.com/xabarov/most-queue/tree/main/tests) for examples
- Look [here](https://github.com/xabarov/most-queue/tree/main/tutorials) for jupyter tutorials

### Simple example: Calculate M/G/1 queueing system calculation with `most_queue` library. Run simulation and compare results with theory.

```python
import numpy as np

from most_queue.general_utils.tables import probs_print, times_print
from most_queue.rand_distribution import H2Distribution
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.mg1_calc import MG1Calculation

l = 1  # input flow intensity
b1 = 0.7  # average service time
coev = 1.2  # coefficient of variation of service time
num_of_jobs = 1000000  # number of jobs for simulation

# selecting parameters of the approximating H2-distribution for service time [y1, mu1, mu2]:
params = H2Distribution.get_params_by_mean_and_coev(b1, coev)
b = H2Distribution.calc_theory_moments(params, 4)

# calculation using numerical methods
mg1_num = MG1Calculation(l, b)
w_ch = mg1_num.get_w()
p_ch = mg1_num.get_p()
v_ch = mg1_num.get_v()

# running simulation for verification of results
qs = QueueingSystemSimulator(1)
qs.set_servers(params, "H")
qs.set_sources(l, "M")
qs.run(num_of_jobs)
w_sim = qs.w
p_sim = qs.get_p()
v_sim = qs.v

# outputting the results
print("M/H2/1")

times_print(w_sim, w_ch, True)
times_print(v_sim, v_ch, False)
probs_print(p_sim, p_ch, 10)
```

### Priority queue example.   For verification, comparing results with those calculated using the method of [invariant relations](https://cyberleninka.ru/article/n/raschet-mnogokanalnyh-sistem-obsluzhivaniya-s-absolyutnym-i-otnositelnym-prioritetami-na-osnove-invariantov-otnosheniya)

```python
n = 5  # number of servers
k = 3  # number of classes of requests
l = [0.2, 0.3, 0.4]  # service intensities by request classes
lsum = sum(l)
num_of_jobs = 300000  # number of jobs for the simulation

# Set up the parameters for service times at initial moments.
# Set average service times by class
b1 = [0.45 * n, 0.9 * n, 1.35 * n]

# Coefficient of variation of service time let's be the same for all classes
coev = 0.577

# second initial moments
b2 = [0] * k
for i in range(k):
    b2[i] = (b1[i] ** 2) * (1 + coev ** 2)

b_sr = sum(b1) / k

# get the coefficient of load
ro = lsum * b_sr / n

# now, given the two initial moments, select parameters for the approximating Gamma distribution
# and add them to the list of parameters params
params = []
for i in range(k):
    params.append(GammaDistribution.get_params([b1[i], b2[i]]))

b = []
for j in range(k):
    b.append(GammaDistribution.calc_theory_moments(params[j], 4))

print("\nComparison of data from the simulation and results calculated using the method of invariant relations (R) \n"
      "time spent in a multi-channel queue with priorities")
print("Number of servers: " + str(n) + "\nNumber of classes: " + str(k) + "\nCoefficient of load: {0:<1.2f}".format(ro) +
      "\nCoefficient of variation of service time: " + str(coev) + "\n")
print("PR (Preamptive) priority")

# when creating the simulation, pass the number of servers, number of classes and type of priority.
# PR - absolute with re-service of requests
qs = PriorityQueueSimulator(n, k, "PR")

# to set up sources of requests and service servers, we need to specify a set of dictionaries with fields
# type - distribution type,
# params - its parameters.
# The number of such dictionaries in the lists sources and servers_params corresponds to the number of classes

sources = []
servers_params = []
for j in range(k):
    sources.append({'type': 'M', 'params': l[j]})
    servers_params.append({'type': 'Gamma', 'params': params[j]})

qs.set_sources(sources)
qs.set_servers(servers_params)

# start the simulation
qs.run(num_of_jobs)

# get the initial moments of time spent

v_sim = qs.v

# calculate them as well using the method of invariant relations (for comparison)
v_teor = priority_calc.get_v_prty_invar(l, b, n, 'PR')

assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3

times_print_with_classes(v_sim, v_teor, False)

print("NP (Non-preamptive) priority")

# The same for relative priority (NP)
qs = PriorityQueueSimulator(n, k, "NP")
sources = []
servers_params = []
for j in range(k):
    sources.append({'type': 'M', 'params': l[j]})
    servers_params.append({'type': 'Gamma', 'params': params[j]})

qs.set_sources(sources)
qs.set_servers(servers_params)

qs.run(num_of_jobs)

v_sim = qs.v

v_teor = priority_calc.get_v_prty_invar(l, b, n, 'NP')

times_print_with_classes(v_sim, v_teor, False)
```

### References
-  [(Ryzhikov Y.I. *Computational Methods*. BHV-Petersburg, 2007.)](https://www.litres.ru/book/uriy-ryzhikov/vychislitelnye-metody-uchebnoe-posobie-644835/)


