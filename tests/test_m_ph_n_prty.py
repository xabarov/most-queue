"""
Testing the calculation of M/PH, M/n queue with 2 classes of jobs 
and absolute priority using the numerical method by Takacs-Takaki
based on the approximation of busy periods by Cox's second-order distribution.
For verification, we use simulation
"""
import math
import time

from most_queue.rand_distribution import (
    CoxDistribution,
    ExpDistribution,
    GammaDistribution,
)
from most_queue.sim.queueing_systems.priority import PriorityQueueSimulator
from most_queue.theory.queueing_systems.priority.mgn_invar_approx import MGnInvarApproximation
from most_queue.theory.queueing_systems.priority.preemptive.m_ph_n_busy_approx import MPhNPrty


def test_m_ph_n_prty():
    """
    Testing the calculation of M/PH, M/n queue with 2 classes of jobs 
    and absolute priority using the numerical method by Takacs-Takaki
    based on the approximation of busy periods by Cox's second-order distribution.
    For verification, we use simulation
    """
    num_of_jobs = 300000  # number of served jobs in IM

    is_cox = False  # use Cox's second-order distribution or H2-distribution for approximating busy periods
    max_iter = 100  # maximum number of iterations for the numerical method
    # Investigation of the influence of the average time spent by requests of class 2 on the load factor
    n = 4  # number of servers
    K = 2  # number of classes
    ros = 0.75  # load factor of the queue
    bH_to_bL = 2  # service time for class H is less than L by this factor
    lH_to_lL = 1.5  # arrival rate of requests of class H is lower than L by this factor
    l_H = 1.0  # arrival rate of the input stream of type 1 requests
    l_L = lH_to_lL * l_H  # arrival rate of the input stream of type 2 requests
    bHcoev = 1.2  # investigated coefficients of variation for service times of class 1

    print(f"coev =  {bHcoev:5.3f}")

    lsum = l_L + l_H
    bsr = n * ros / lsum
    bH1 = lsum * bsr / (l_L * bH_to_bL + l_H)
    bL1 = bH_to_bL * bH1
    bH = [0.0] * 3
    alpha = 1 / (bHcoev ** 2)
    bH[0] = bH1
    bH[1] = math.pow(bH[0], 2) * (math.pow(bHcoev, 2) + 1)
    bH[2] = bH[1] * bH[0] * (1.0 + 2 / alpha)

    gamma_params = GammaDistribution.get_params([bH[0], bH[1]])

    mu_L = 1.0 / bL1

    cox_params = CoxDistribution.get_params(bH)

    # calculation using the numerical method:
    tt_start = time.process_time()
    tt = MPhNPrty(mu_L, cox_params, l_L, l_H, n=n, is_cox=is_cox,
                  max_iter=max_iter, verbose=False)
    tt.run()
    tt_time = time.process_time() - tt_start

    iter_num = tt.run_iterations_num_
    v2_tt = tt.get_low_class_v1()

    mu_L = 1.0 / bL1

    bL = ExpDistribution.calc_theory_moments(mu_L, 3)

    b = []
    b.append(bH)
    b.append(bL)

    invar_start = time.process_time()
    invar_calc = MGnInvarApproximation([l_H, l_L], b, n=n)
    v = invar_calc.get_v(priority='PR', num=2)
    v2_invar = v[1][0]
    invar_time = time.process_time() - invar_start

    im_start = time.process_time()

    qs = PriorityQueueSimulator(n, K, "PR")
    sources = []
    servers_params = []

    sources.append({'type': 'M', 'params': l_H})
    sources.append({'type': 'M', 'params': l_L})
    servers_params.append({'type': 'Gamma', 'params': gamma_params})
    servers_params.append({'type': 'M', 'params': mu_L})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # running the simulation:
    qs.run(num_of_jobs)

    # getting the results of the simulation:
    v_sim = qs.v
    v1_sim = v_sim[0][0]
    v2_sim = v_sim[1][0]

    im_time = time.process_time() - im_start

    print("\nComparison of the results calculated using the numerical method with approximation of busy periods by Cox's second-order distribution and simulation.")
    print(f"ro: {ros:1.2f}")
    print(f"n : {n}")
    print(f"Number of served jobs for simulation: {num_of_jobs}")
    print(f'Calc iterations: {iter_num}')

    print("\n")
    print("Average times spent in the queue by requests of class 2")
    print("-" * 45)
    rows = ["Ours", "Sim", "Invar"]
    values = [
        v2_tt,
        v2_sim,
        v2_invar,
    ]

    times = [tt_time,
             im_time,
             invar_time]

    headers = ["Calc type", "v1 low", 'calc time, s']

    print("{0:^15s}|{1:^15s}|{2:^15s}".format(*headers))
    print("-" * 45)
    for row, value, t in zip(rows, values, times):
        print("{0:^15s}|{1:^14.3f} | {2:^14.3f}".format(row, value, t))
    print("-" * 45)
    print("\n")


if __name__ == "__main__":
    test_m_ph_n_prty()
