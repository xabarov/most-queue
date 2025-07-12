"""
Testing the calculation of M/PH, M/n queue with 2 classes of jobs 
and absolute priority using the numerical method by Takacs-Takaki
based on the approximation of busy periods by Cox's second-order distribution.
For verification, we use simulation
"""
import math
import os
import time

import yaml

from most_queue.rand_distribution import ExpDistribution, GammaDistribution
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation
from most_queue.theory.priority.preemptive.m_ph_n_busy_approx import MPhNPrty

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params['num_of_channels'])

SERVICE_TIME_CV = float(params['service']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

NUM_OF_CLASSES = 2
ARRIVAL_RATE_HIGH = float(params['arrival']['rate'])
ARRIVAL_RATE_LOW = 1.5*ARRIVAL_RATE_HIGH

MAX_ITER = 100  # maximum number of iterations for the numerical method
IS_COX = False  # use C2 distribution or H2-distribution for approximating busy periods
SERVICE_PROPORTION = 2  # service time for class H is less than L by this factor


def test_m_ph_n_prty():
    """
    Testing the calculation of M/PH, M/n queue with 2 classes of jobs 
    and absolute priority using the numerical method by Takacs-Takaki
    based on the approximation of busy periods by Cox's second-order distribution.
    For verification, we use simulation
    """

    print(f"coev =  {SERVICE_TIME_CV:5.3f}")

    lsum = ARRIVAL_RATE_LOW + ARRIVAL_RATE_HIGH
    bsr = NUM_OF_CHANNELS * UTILIZATION_FACTOR / lsum
    b1_high = lsum * bsr / (ARRIVAL_RATE_LOW *
                            SERVICE_PROPORTION + ARRIVAL_RATE_HIGH)
    b1_low = SERVICE_PROPORTION * b1_high
    b_high = [0.0] * 3
    alpha = 1 / (SERVICE_TIME_CV ** 2)
    b_high[0] = b1_high
    b_high[1] = math.pow(b_high[0], 2) * (math.pow(SERVICE_TIME_CV, 2) + 1)
    b_high[2] = b_high[1] * b_high[0] * (1.0 + 2 / alpha)

    gamma_params = GammaDistribution.get_params([b_high[0], b_high[1]])

    mu_low = 1.0 / b1_low

    # calculation using the numerical method:
    tt_start = time.process_time()
    tt = MPhNPrty(mu_low, b_high, ARRIVAL_RATE_LOW, ARRIVAL_RATE_HIGH,
                  n=NUM_OF_CHANNELS, is_cox=IS_COX,
                  max_iter=MAX_ITER, verbose=False)
    tt.run()
    tt_time = time.process_time() - tt_start

    iter_num = tt.run_iterations_num_
    v_low_tt = tt.get_low_class_v1()

    mu_low = 1.0 / b1_low

    b_low = ExpDistribution.calc_theory_moments(mu_low, 3)

    b = []
    b.append(b_high)
    b.append(b_low)

    invar_start = time.process_time()
    invar_calc = MGnInvarApproximation(
        [ARRIVAL_RATE_HIGH, ARRIVAL_RATE_LOW], b, n=NUM_OF_CHANNELS)
    v = invar_calc.get_v(priority='PR', num=2)
    v_low_invar = v[1][0]
    invar_time = time.process_time() - invar_start

    im_start = time.process_time()

    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, NUM_OF_CLASSES, "PR")
    sources = []
    servers_params = []

    sources.append({'type': 'M', 'params': ARRIVAL_RATE_HIGH})
    sources.append({'type': 'M', 'params': ARRIVAL_RATE_LOW})
    servers_params.append({'type': 'Gamma', 'params': gamma_params})
    servers_params.append({'type': 'M', 'params': mu_low})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # running the simulation:
    qs.run(NUM_OF_JOBS)

    # getting the results of the simulation:
    v_sim = qs.v
    v_low_sim = v_sim[1][0]

    sim_time = time.process_time() - im_start

    print("\nComparison of the results calculated using the numerical method with approximation")
    print(" of busy periods by Cox's second-order distribution and simulation.")
    print(f"ro: {UTILIZATION_FACTOR:1.2f}")
    print(f"n : {NUM_OF_CHANNELS}")
    print(f"Number of served jobs for simulation: {NUM_OF_JOBS}")
    print(f'Calc iterations: {iter_num}')

    print("\n")
    print("Average times spent in the queue by requests of class 2")
    print("-" * 45)
    headers = ["Calc type", "v1 low", 'calc time, s']

    print("{0:^15s}|{1:^15s}|{2:^15s}".format(*headers))
    print("-" * 45)
    row = 'Ours'
    print(f"{row:^15}|{v_low_tt:^14.3f} | {tt_time:^14.3f}")
    row = 'Invar'
    print(f"{row:^15}|{v_low_invar:^14.3f} | {invar_time:^14.3f}")
    print("-" * 45)
    row = 'Sim'
    print(f"{row:^15}|{v_low_sim:^14.3f} | {sim_time:^14.3f}")
    print("-" * 45)
    print("\n")

    assert abs(v_low_tt-v_low_sim) < 0.1, ERROR_MSG


if __name__ == "__main__":
    test_m_ph_n_prty()
