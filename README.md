# Queueing systems. Simulation and numerical methods of Queuing theory

![Queue](assets/queue6.jpg)

Python package for calculation and simulation of queueing systems (QS) and networks. 

MOST (MOCT,  Массовое обслуживание стационарные задачи) stands for “Mass Service Steady-State Problems” and refers to tasks within queueing theory that involve large-scale service systems operating under steady-state conditions. 
- [Рыжиков Ю. И. Вычислительные методы. – БХВ-Петербург, 2007.](https://www.litres.ru/book/uriy-ryzhikov/vychislitelnye-metody-uchebnoe-posobie-644835/)

Queueing theory. Numerical methods. 


## Authors
- [xabarov](https://github.com/xabarov)

## Installation
Install most-queue with pip
```bash
  pip install most-queue
```

## DESCRIPTION
Most_queue consists of two main parts:
 - **.theory** contains programs that implement methods for calculating queuing theory models. 
 - **.sim** contains simulation programs. 
### Package .theory
| #   | Package name                      | Description                                                                                                                                                                                                                                                |
|-----|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.  | batch_mm1                         |        Calculation of M/M/1 QS with batch arrival                                                                                                                                                                                              |
| 2. | ek_d_n_calc  | Numerical calculation of a multi-channel system Ek/D/n |
| 3. | engset_model |   Calculation of the Engset model for M/M/1 with a finite number of sources.    |
| 4. | fj_calc    | Numerical calculation of the initial moments of the random variables maximum distribution    | 
| 5.  | gi_m_1_calc                       | Numerical calculation of QS GI/M/1    |
| 6.  | gi_m_n_calc                       | Numerical calculation of QS GI/M/n    |
| 7. | impatience_calc                   | Calculation of M/M/1 with exponential impatience    |
| 8.  | m_d_n_calc                        | Numerical calculation of QS M/D/n    |
| 9.  | m_h2_h2_warm                        | Calculation of the M/H2/n system with H2-warming using the Takahasi-Takagi method.   |
| 10.  | m_ph_n_prty   | Numerical calculation of QS M/Ph/n with 2 classes and PR - priority. Based on the approximation of busy periods    |
| 11.  | mg1_calc                          | Numerical calculation of QS M/G/1    |
| 12.  | mg1_warm_calc                     | Numerical calculation of QS M/G/1 with "warm-up"    |
| 13. | mgn_tt                            | Numerical calculation of QS M/H2/n by the Takahashi-Takami method with complex parameters when approximating the serving time by the H2 distribution       |
| 14. | mgn_with_h2_delay_cold_warm.py    | Multichannel queuing system with H2 serving time, H2 warm-up, H2 cold delay and H2 cold (vacations). The system uses complex parameters, which allows you to calculate systems with arbitrary serving, warm-up, cold-delay and cold variation coefficients |
| 15. | mmn_prty_pnz_approx               | Calculation of QS M/M/2 with 2 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of the busy period by the Cox distribution  |
| 16. | mmn_with_h2_cold_h2_warmup.py     | Multichannel queuing system with exp serving time, H2 warm-up and H2 cold (vacations). The system uses complex parameters, which allows to calculate systems with arbitrary warm-up and cold variation coefficients    |
| 17. | mmn3_pnz_cox_approx               | Calculation of QS M/M/2 with 3 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of busy period by the Cox distribution      |
| 18. | mmnr_calc                         | Calculation of QS M/M/n/r     |
| 19. | network_calc                      | Calculation of queuing network with priorities in nodes     |
| 20. | network_viewer.py                 | Utility to view network structure        |
| 21. | priority_calc                     | A set of functions for calculating QS with priorities (single-channel, multi-channel). The multichannel calculation is carried out by the method of relation     |


### Package .sim
| #  | Package name               | Description |
| ------------- |----------------------------|------------- |
| 1. | batch_sim                   | Simulation of QS in a batch system |
| 2.  | fj_delta_sim               | Simulation of QS fork-join with a delay in the start of processing between channels | 
| 3.  | fj_sim                     | Simulation of QS with fork-join process |
| 4.  | flow_sum_sim               | Simulation of flow summation | 
| 5.  | impatient_sim.py           | Simulation of QS GI/G/m/n with impatience | 
| 6.  | priority_network           | Simulation of queuing network with priorities in nodes | 
| 7.  | priority_queue_sim         | Simulation of QS GI/G/m/n  with priorities  | 
| 8.  | qs_sim                     | Simulation of QS GI/G/m/n  | 
| 9.  | queue_finite_source_sim.py | Simulation of QS GI/G/m/n with finite sources | 

## Usage
- Look [here](https://github.com/xabarov/most-queue/tree/main/tests) for examples
- Look [here](https://github.com/xabarov/most-queue/tree/main/tutorials) for jupyter tutorials

### Simple example: Calculate M/G/1 queueing system calculation with `most_queue` library. Run simulation and compare results with theory.

```python
import numpy as np

from most_queue.general_utils.tables import probs_print, times_print
from most_queue.rand_distribution import H2_dist
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.mg1_calc import get_p, get_v, get_w

l = 1  # input flow intensity
b1 = 0.7  # average service time
coev = 1.2  # coefficient of variation of service time
num_of_jobs = 1000000  # number of jobs for simulation

# selecting parameters of the approximating H2-distribution for service time [y1, mu1, mu2]:
params = H2_dist.get_params_by_mean_and_coev(b1, coev)
b = H2_dist.calc_theory_moments(*params, 4)

# calculation using numerical methods
w_ch = get_w(l, b)
p_ch = get_p(l, b, 100)
v_ch = get_v(l, b)

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
    params.append(Gamma.get_mu_alpha([b1[i], b2[i]]))

b = []
for j in range(k):
    b.append(Gamma.calc_theory_moments(params[j][0], params[j][1], 4))

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






