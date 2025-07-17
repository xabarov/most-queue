"""
Test for Engset model (M/M/1 with a finite number of sources)
"""

import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
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

    # Get probabilities of states of the system
    p_num = engset.get_p()

    engset_results = engset.run()

    print(f"v1 = {engset_results.v[0]:3.3f}, w1 = {engset_results.w[0]:3.3f}")

    # Simulation of the system with a finite number of sources

    finite_source_sim = QueueingFiniteSourceSim(1, SOURCE_NUM)

    finite_source_sim.set_sources(ARRIVAL_RATE, "M")
    finite_source_sim.set_servers(service_rate, "M")

    finite_source_sim.run(NUM_OF_JOBS)

    p_sim = finite_source_sim.get_p()
    v_sim = finite_source_sim.v
    w_sim = finite_source_sim.w

    # Comparison of the results from the simulation and the analytical model

    probs_print(p_sim, p_num)

    assert np.allclose(p_sim[:10], p_num[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG

    times_print(w_sim, engset_results.w, is_w=True)
    times_print(v_sim, engset_results.v, is_w=False)

    assert np.allclose(w_sim, engset_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(v_sim, engset_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":

    test_engset()
