"""
Test of GI/M/1 queueing system calculation.
For verification, we use simulation
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import probs_print, times_print
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gi_m_1 import GiM1

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
ARRIVAL_CV = float(params["arrival"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def test_gi_m_1():
    """
    Test of GI/M/1 queueing system calculation.
    For verification, we use simulation
    """

    a1 = 1 / ARRIVAL_RATE  # average interval between requests
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR  # service intensity

    # calculation of parameters approximating Gamma-distribution for arrival
    # times
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(a1, ARRIVAL_CV)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculation of initial moments of time spent and waiting in the queueing
    # system
    gm1_calc = GiM1()
    gm1_calc.set_sources(a)
    gm1_calc.set_servers(mu)

    calc_results = gm1_calc.run()

    # for verification, we use sim.
    # create an instance of the sim class and pass the number of service
    # channels
    qs = QsSim(1)

    # set the input stream. The method needs to be passed parameters
    # of distribution as a list and type of distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels. Parameters (in our case, the service intensity)
    # and type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start simulation
    sim_results = qs.run(NUM_OF_JOBS)

    # Output results
    print("\nGamma/M/1 simulation and calculation results\n")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    times_print(sim_results.w, calc_results.w, True)
    times_print(sim_results.v, calc_results.v, False)
    probs_print(sim_results.p, calc_results.p, size=10)

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], rtol=PROBS_RTOL, atol=PROBS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_gi_m_1()
