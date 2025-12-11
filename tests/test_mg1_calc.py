"""
Testing the M/G/1 queueing system calculation
For verification, we use simulation modeling
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import H2Distribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


SERVICE_TIME_CV = float(params["service"]["cv"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

NUM_OF_CHANNELS = 1


def test_mg1():
    """
    Testing the M/G/1 queueing system calculation
    For verification, we use simulation modeling
    """
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE

    # selecting parameters of the approximating H2-distribution
    # for service time H2Params [p1, mu1, mu2]:
    h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
    print(h2_params)
    b = H2Distribution.calc_theory_moments(h2_params, 5)

    # calculation using numerical methods
    mg1_num = MG1Calc()
    mg1_num.set_sources(ARRIVAL_RATE)
    mg1_num.set_servers(b)

    calc_results = mg1_num.run()

    assert abs(UTILIZATION_FACTOR - calc_results.utilization) < PROBS_ATOL

    # running IM for verification of results
    qs = QsSim(1)
    qs.set_servers(h2_params, "H")
    qs.set_sources(ARRIVAL_RATE, "M")
    sim_results = qs.run(NUM_OF_JOBS)

    # outputting the results
    print("M/H2/1")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)
    probs_print(sim_results.p, calc_results.p, 10)

    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)

    assert np.allclose(
        np.array(sim_results.p[:10]), np.array(calc_results.p[:10]), atol=PROBS_ATOL, rtol=PROBS_RTOL
    ), ERROR_MSG


if __name__ == "__main__":
    test_mg1()
