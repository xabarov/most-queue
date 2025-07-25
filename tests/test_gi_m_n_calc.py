"""
Testing the GI/M/n queueing system calculation.
For verification, we use imitational modeling.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gi_m_n import GiMn

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params["num_of_channels"])

ARRIVAL_RATE = float(params["arrival"]["rate"])
ARRIVAL_CV = float(params["arrival"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def test_gi_m_n():
    """
    Testing the GI/M/n queueing system calculation.
    For verification, we use imitational modeling.
    """

    a1 = 1.0 / ARRIVAL_RATE  # average interval between arrivals
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE  # average service time given ro
    mu = 1 / b1  # service intensity

    # calculate parameters of the approximating Gamma distribution for arrival
    # times
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(a1, ARRIVAL_CV)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculate raw moments of sojourn and waiting times in the queueing
    # system

    gi_m_n_calc = GiMn(n=NUM_OF_CHANNELS)

    gi_m_n_calc.set_sources(a=a)
    gi_m_n_calc.set_servers(mu=mu)

    calc_results = gi_m_n_calc.run()

    print(f"utilization: {calc_results.utilization:0.4f}")

    # for verification, we use simulation.
    # create an instance of the Simulation class and pass the number of servers
    qs = QsSim(NUM_OF_CHANNELS)

    # set the ariival distribution paprams.
    # The method needs to be passed parameters as a list and the type of
    # distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels.
    # The method should receive parameters (in our case, the service intensity)
    # and the type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start the simulation:
    sim_results = qs.run(NUM_OF_JOBS)

    # Output results
    print("\nGamma/M/n simulation and calculation results:")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)
    probs_print(sim_results.p, calc_results.p)

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], rtol=PROBS_RTOL, atol=PROBS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_gi_m_n()
