"""
Testing the GI/M/n queueing system calculation.
For verification, we use imitational modeling.
"""

import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
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
    gamma_params = GammaDistribution.get_params_by_mean_and_coev(a1, ARRIVAL_CV)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculate initial moments of sojourn and waiting times in the queueing
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
    qs.run(NUM_OF_JOBS)

    # get the list of initial moments of sojourn and waiting times in the
    # queueing system
    v_sim = qs.v
    w_sim = qs.w

    # get the distribution of probabilities of states in the queueing system
    p_sim = qs.get_p()

    # Output results
    print("\nGamma")

    times_print(w_sim, calc_results.w)
    times_print(v_sim, calc_results.v, is_w=False)
    probs_print(p_sim, calc_results.p)

    assert np.allclose(v_sim, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(p_sim[:10], calc_results.p[:10], rtol=PROBS_RTOL, atol=PROBS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_gi_m_n()
