"""
Test for Engset model (M/M/1 with a finite number of sources)
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.sim.finite_source import QueueingFiniteSourceSim
from most_queue.theory.closed.engset import Engset

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params["num_of_channels"])

SOURCE_NUM = 7
ARRIVAL_RATE = float(params["arrival"]["rate"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def test_engset():
    """
    Test for Engset model (M/M/1 with a finite number of sources)
    """
    # Calculation of the Engset model
    service_rate = ARRIVAL_RATE / UTILIZATION_FACTOR
    engset = Engset()
    engset.set_sources(l=ARRIVAL_RATE, number_of_sources=SOURCE_NUM)
    engset.set_servers(mu=service_rate)

    calc_results = engset.run()

    print(f"v1 = {calc_results.v[0]:3.3f}, w1 = {calc_results.w[0]:3.3f}")

    # Simulation of the system with a finite number of sources

    finite_source_sim = QueueingFiniteSourceSim(1, SOURCE_NUM)

    finite_source_sim.set_sources(ARRIVAL_RATE, "M")
    finite_source_sim.set_servers(service_rate, "M")

    sim_results = finite_source_sim.run(NUM_OF_JOBS)

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    # Comparison of the results from the simulation and the analytical model

    probs_print(sim_results.p, calc_results.p)

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)

    assert np.allclose(sim_results.w[:3], calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.v[:3], calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":

    test_engset()
