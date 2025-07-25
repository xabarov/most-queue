"""
Testing the calculation of M/PH, M/n queue with 2 classes of jobs
and absolute priority using the numerical method by Takacs-Takaki
based on the approximation of busy periods by Cox's second-order distribution.
For verification, we use simulation
"""

import os

import yaml

from most_queue.random.distributions import ExpDistribution, GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation
from most_queue.theory.priority.preemptive.m_ph_n_busy_approx import MPhNPrty, TakahashiTakamiParams

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params["num_of_channels"])

SERVICE_TIME_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

NUM_OF_CLASSES = 2
ARRIVAL_RATE_HIGH = float(params["arrival"]["rate"])
ARRIVAL_RATE_LOW = 1.5 * ARRIVAL_RATE_HIGH

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

    print(f"cv =  {SERVICE_TIME_CV:5.3f}")

    lsum = ARRIVAL_RATE_LOW + ARRIVAL_RATE_HIGH
    bsr = NUM_OF_CHANNELS * UTILIZATION_FACTOR / lsum
    b1_high = lsum * bsr / (ARRIVAL_RATE_LOW * SERVICE_PROPORTION + ARRIVAL_RATE_HIGH)
    b1_low = SERVICE_PROPORTION * b1_high
    b_high = gamma_moments_by_mean_and_cv(b1_high, SERVICE_TIME_CV)

    gamma_params = GammaDistribution.get_params([b_high[0], b_high[1]])

    mu_low = 1.0 / b1_low

    # calculation using the numerical method:
    calc_params = TakahashiTakamiParams()
    calc_params.is_cox = IS_COX
    calc_params.max_iter = MAX_ITER

    tt = MPhNPrty(
        n=NUM_OF_CHANNELS,
        calc_params=calc_params,
    )
    tt.set_sources(l_low=ARRIVAL_RATE_LOW, l_high=ARRIVAL_RATE_HIGH)
    tt.set_servers(b_high=b_high, mu_low=mu_low)
    calc_results = tt.run()

    iter_num = tt.num_of_iter_

    mu_low = 1.0 / b1_low

    b_low = ExpDistribution.calc_theory_moments(mu_low, len(b_high))

    b = []
    b.append(b_high)
    b.append(b_low)

    invar_calc = MGnInvarApproximation(n=NUM_OF_CHANNELS, priority="PR")
    invar_calc.set_sources([ARRIVAL_RATE_HIGH, ARRIVAL_RATE_LOW])
    invar_calc.set_servers(b)
    invar_results = invar_calc.run()

    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, NUM_OF_CLASSES, "PR")
    sources = []
    servers_params = []

    sources.append({"type": "M", "params": ARRIVAL_RATE_HIGH})
    sources.append({"type": "M", "params": ARRIVAL_RATE_LOW})
    servers_params.append({"type": "Gamma", "params": gamma_params})
    servers_params.append({"type": "M", "params": mu_low})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # running the simulation:
    sim_results = qs.run(NUM_OF_JOBS)

    print("\nComparison of the results calculated using the numerical method with approximation")
    print(" of busy periods by Cox's second-order distribution and simulation.")
    print(f"ro: {UTILIZATION_FACTOR:1.2f}")
    print(f"n : {NUM_OF_CHANNELS}")
    print(f"Number of served jobs for simulation: {NUM_OF_JOBS}")
    print(f"Calc iterations: {iter_num}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print("\n")
    print("Mean sojourn times")
    print("-" * 60)
    headers = ["Calc type", "v1 high", "v1 low", "calc time, s"]

    print("{0:^15s}|{1:^14s}|{2:^15s}|{3:^15s}".format(*headers))
    print("-" * 60)
    row = "Ours"
    print(f"{row:^15}|{calc_results.v[0][0]:^14.3f}|{calc_results.v[1][0]:^14.3f} | {calc_results.duration:^14.3f}")
    row = "Invar"
    print(f"{row:^15}|{invar_results.v[0][0]:^14.3f}|{invar_results.v[1][0]:^14.3f} | {invar_results.duration:^14.3f}")
    print("-" * 60)
    row = "Sim"
    print(f"{row:^15}|{sim_results.v[0][0]:^13.3f} |{sim_results.v[1][0]:^14.3f} | {sim_results.duration:^14.3f}")
    print("-" * 60)
    print("\n")

    assert abs(calc_results.v[1][0] - sim_results.v[1][0]) < 0.1, ERROR_MSG


if __name__ == "__main__":
    test_m_ph_n_prty()
