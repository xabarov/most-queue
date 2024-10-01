# MostQueue

![Queue](assets/queue6.jpg)

Python package for calculation and simulation of queueing systems (QS) and networks. 
Queueing theory. Numerical methods. 

Пакет python для расчета и моделирования систем массового обслуживания (СМО) и сетей (СеМО).
Теория массового обслуживания. Численные методы.

## Authors
- [xabarov](https://github.com/xabarov)

## Installation
Install most-queue with pip
```bash
  pip install most-queue
  pip install -r requirements.txt
```

## DESCRIPTION
Most_queue consists of two main parts:
 - **.theory** contains programs that implement methods for calculating queuing theory models. 
 - **.sim** contains simulation programs. 
 - **.tests** contains tests. 
 - **.rand_distribution** - A set of functions and classes designed to generate a random values and select parameters for distributions like H2, C2, Ek, Pa, Gamma 

### Package .theory
| #   | Package name                      | Description                                                                                                                                                                                                                                                |
|-----|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.  | diff5dots                         | Numerical calculation of the derivative of a function                                                                                                                                                                                                      |
| 2.  | fj_calc                           | Numerical calculation of the initial moments of the random variables maximum distribution                                                                                                                                                                  | 
| 3.  | m_ph_n_prty                       | Numerical calculation of QS M/Ph/n with 2 classes and PR - priority. Based on the approximation of busy periods                                                                                                                                         |
| 4.  | mg1_calc                          | Numerical calculation of QS M/G/1                                                                                                                                                                                                                       |
| 5.  | gi_m_1_calc                       | Numerical calculation of QS GI/M/1                                                                                                                                                                                                                      |
| 6.  | gi_m_n_calc                       | Numerical calculation of QS GI/M/n                                                                                                                                                                                                                      |
| 7.  | m_d_n_calc                        | Numerical calculation of QS M/D/n                                                                                                                                                                                                                       |
| 8.  | ek_d_n_calc                       | Numerical calculation of QS Ek/D/n                                                                                                                                                                                                                      |
| 9.  | mg1_warm_calc                     | Numerical calculation of QS M/G/1 with "warm-up"                                                                                                                                                                                                        |
| 10. | mgn_tt                            | Numerical calculation of QS M/H2/n by the Takahashi-Takami method with complex parameters when approximating the serving time by the H2 distribution                                                                                                    |
| 11. | mmn3_pnz_cox_approx               | Calculation of QS M/M/2 with 3 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of busy period by the Cox distribution                                                                                        |
| 12. | mmn_prty_pnz_approx               | Calculation of QS M/M/2 with 2 classes, PR - priority by the Takahashi-Takami numerical method based on the approximation of the busy period by the Cox distribution                                                                                    |
| 13. | mmnr_calc                         | Calculation of QS M/M/n/r                                                                                                                                                                                                                               |
| 14. | network_calc                      | Calculation of queuing network with priorities in nodes                                                                                                                                                                                                    |
| 15. | passage_time                      | Calculation of the initial transition times between arbitrary tiers of the Markov chain                                                                                                                                                                    |
| 16. | priority_calc                     | A set of functions for calculating QS with priorities (single-channel, multi-channel). The multichannel calculation is carried out by the method of relation invariants                                                                                 |
| 17. | student_stat                      | Calculation of confidence probability, confidence intervals for random variable with unknown RMS based on Student's t-distribution.                                                                                                                        |
| 18. | convolution_sum_calc              | Calculate the initial moments of sums of random variables                                                                                                                                                                                                  |
| 19. | weibull                           | Selection of Weibull distribution parameters, calculation of CDF, PDF, Tail values                                                                                                                                                                         |
| 20. | flow_sum                          | Flow summation, numerical calculation                                                                                                                                                                                                                      |
| 21. | impatience_calc                   | Calculation of M/M/1 with exponential impatience                                                                                                                                                                                                           |
| 22. | batch_mm1                         | Calculation of QS M/M/1 with batch arrival                                                                                                                                                                                                              |
| 23. | engset_model.py                   | Calculation of QS M/M/1 with finite sources                                                                                                                                                                                                             |
| 24. | generate_pareto_noise.py.py       | Create noise by Pareto dist                                                                                                                                                                                                                                |
| 25. | network_viewer.py                 | Utility to view network structure                                                                                                                                                                                                                          |
| 26. | mmn_with_h2_cold_h2_warmup.py     | Multichannel queuing system with exp serving time, H2 warm-up and H2 cold (vacations). The system uses complex parameters, which allows to calculate systems with arbitrary warm-up and cold variation coefficients                                        |
| 27. | mgn_with_h2_delay_cold_warm.py    | Multichannel queuing system with H2 serving time, H2 warm-up, H2 cold delay and H2 cold (vacations). The system uses complex parameters, which allows you to calculate systems with arbitrary serving, warm-up, cold-delay and cold variation coefficients |
### Package .sim
| #  | Package name               | Description |
| ------------- |----------------------------|------------- |
| 1.  | fj_delta_sim               | Simulation of QS fork-join with a delay in the start of processing between channels | 
| 2.  | fj_sim                     | Simulation of QS with fork-join process | 
| 3.  | priority_network           | Simulation of queuing network with priorities in nodes | 
| 4.  | qs_sim                     | Simulation of QS GI/G/m/n  | 
| 5.  | priority_queue_sim         | Simulation of QS GI/G/m/n  with priorities  | 
| 6.  | flow_sum_sim               | Simulation of flow summation | 
| 7.  | impatient_sim.py           | Simulation of QS GI/G/m/n with impatience | 
| 8.  | batch_sim.py               | Simulation of QS GI/G/m/n with batch arrival | 
| 9.  | queue_finite_source_sim.py | Simulation of QS GI/G/m/n with finite sources | 

## Usage
- Look [here](https://github.com/xabarov/most-queue/tree/main/most_queue/tests) for examples
- Look [here](https://github.com/xabarov/most-queue/tree/main/most_queue/tutorials) for jupyter tutorials

- Simple example: numeric calc for M/G/1 QS. Run simulation for verification

```python

import numpy as np

from most_queue.general_utils.tables import probs_print, times_print
from most_queue.rand_distribution import H2_dist, Pareto_dist, Uniform_dist
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.mg1_calc import get_p, get_v, get_w


"""
Test QS M/G/1
"""
arrival_lambda = 1  # arrival intensity
b1 = 0.7  # mean serving time
coev = 1.2  # standard deviation of serving time
num_of_jobs = 1000000  # jobs num for simulation

# get of parameters of the approximating H2-distribution for service time [y1, mu1, mu2]:
params = H2_dist.get_params_by_mean_and_coev(b1, coev)
b = H2_dist.calc_theory_moments(*params, 4)

# get wait moments, sojourn time, probs by numeric calc
w_ch = get_w(arrival_lambda, b)
v_ch = get_v(arrival_lambda, b)
p_ch = get_p(arrival_lambda, b, 100)

# run simulation for verification
qs = QueueingSystemSimulator(1)  # 1 - num of channels

qs.set_servers(params, "H") 
qs.set_sources(arrival_lambda, "M")

qs.run(num_of_jobs)

# get sim results
w_sim = qs.w
p_sim = qs.get_p()
v_sim = qs.v

print("M/H2/1")

times_print(w_sim, w_ch, True)
times_print(v_sim, v_ch, False)
probs_print(p_sim, p_ch, 10)

```

- As an example, we will provide the code for сomparison of calculation results by the Takahashi-Takami and Simulation.

```python

    import math
    from most_queue.sim.qs_sim import QueueingSystemSimulator
    from most_queue.rand_distribution import Gamma
    import time
    from most_queue.theory.mgn_tt import MGnCalc
    from most_queue.general_utils.tables import probs_print, times_print
  
    n = 3  # number of channels
    l = 1.0  # input flow intensity
    ro = 0.8  # utilization
    b1 = n * ro / l  # mean service time
    num_of_jobs = 300000  # num of simulation jobs
    b_coev = 1.5  # service time coefficient of variation


    #  Calculation of initial moments of service time based on a given average and coefficient of variation
    b = [0.0] * 3
    alpha = 1 / (b_coev ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    tt_start = time.process_time()

    #  Run of the Takahashi-Takami method
    tt = MGnCalc(n, l, b)
    tt.run()

    # obtaining results of numerical calculations

    p_tt = tt.get_p()
    v_tt = tt.get_v()
    tt_time = time.process_time() - tt_start

    # You can also find out how many iterations were required
    num_of_iter = tt.num_of_iter_

    # Run simulation to verify results
    im_start = time.process_time()

    qs = QueueingSystemSimulator(n)

    # we set the input flow of jobs. M is exponential with intensity l
    qs.set_sources(l, 'M')

    # We set the parameters of the service channels using Gamma distribution.
    # Select the distribution parameters
    gamma_params = Gamma.get_mu_alpha([b[0], b[1]])
    qs.set_servers(gamma_params, 'Gamma')

    # Run simulation
    qs.run(num_of_jobs)

    # Get sim results
    p = qs.get_p()
    v_sim = qs.v
    im_time = time.process_time() - im_start

    print("\nComparison of calculation results by the Takahashi-Takami and simulation.\n"
          "Sim- M/Gamma/{0:^2d}\nTT- M/H2/{0:^2d}"
          "Utilization: {1:^1.2f}\nService time coef of variation: {2:^1.2f}\n".format(n, ro, b_coev))
    print("T-T iterations num: {0:^4d}".format(num_of_iter))
    print("TT run time: {0:^5.3f} c".format(tt_time))
    print("Sim run time: {0:^5.3f} c".format(im_time))
    probs_print(p, p_tt, 10)

    times_print(v_sim, v_tt, False)
      
```







