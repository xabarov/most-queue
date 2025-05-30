"""
Testing the calculation of M/PH, M/n queue with 2 classes of jobs 
and absolute priority using the numerical method by Takacs-Takaki
based on the approximation of busy periods by Cox's second-order distribution.
For verification, we use simulation
"""
import math
import time

from most_queue.rand_distribution import (
    ExpDistribution,
    GammaDistribution,
)
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation
from most_queue.theory.priority.preemptive.m_ph_n_busy_approx import MPhNPrty

NUM_OF_SERVERS = 4
NUM_OF_CLASSES = 2
ARRIVAL_RATE_HIGH = 1.0
ARRIVAL_RATE_LOW = 1.5
UTILIZATION = 0.75
NUM_OF_JOBS = 300000
SERVICE_TIME_CV = 1.2

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
    bsr = NUM_OF_SERVERS * UTILIZATION / lsum
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
                  n=NUM_OF_SERVERS, is_cox=IS_COX,
                  max_iter=MAX_ITER, verbose=False)
    tt.run()
    tt_time = time.process_time() - tt_start

    iter_num = tt.run_iterations_num_
    v2_tt = tt.get_low_class_v1()

    mu_low = 1.0 / b1_low

    b_low = ExpDistribution.calc_theory_moments(mu_low, 3)

    b = []
    b.append(b_high)
    b.append(b_low)

    invar_start = time.process_time()
    invar_calc = MGnInvarApproximation(
        [ARRIVAL_RATE_HIGH, ARRIVAL_RATE_LOW], b, n=NUM_OF_SERVERS)
    v = invar_calc.get_v(priority='PR', num=2)
    v2_invar = v[1][0]
    invar_time = time.process_time() - invar_start

    im_start = time.process_time()

    qs = PriorityQueueSimulator(NUM_OF_SERVERS, NUM_OF_CLASSES, "PR")
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
    v2_sim = v_sim[1][0]

    im_time = time.process_time() - im_start

    print("\nComparison of the results calculated using the numerical method with approximation")
    print(" of busy periods by Cox's second-order distribution and simulation.")
    print(f"ro: {UTILIZATION:1.2f}")
    print(f"n : {NUM_OF_SERVERS}")
    print(f"Number of served jobs for simulation: {NUM_OF_JOBS}")
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
        print(f"{row:^15}|{value:^14.3f} | {t:^14.3f}")
    print("-" * 45)
    print("\n")


if __name__ == "__main__":
    test_m_ph_n_prty()
