"""
Testing the M/D/n queueing system calculation.
For verification, we use simulation modeling
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import probs_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_d_n import MDn

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


def test_mdn():
    """
    Testing the M/D/n queueing system calculation.
    For verification, we use simulation modeling
    """

    b = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE  # service time from given ro

    # calculation of the probabilities of queueing system states
    mdn = MDn(n=NUM_OF_CHANNELS)
    mdn.set_sources(l=ARRIVAL_RATE)
    mdn.set_servers(b=b)

    calc_results = mdn.run()

    print(f"GI/M/n queueing system utilization: {calc_results.utilization: 0.4f}")

    assert abs(UTILIZATION_FACTOR - calc_results.utilization) < PROBS_ATOL

    # for verification, we use simulation modeling
    # create an instance of the simulation class and pass the number of
    # service channels
    qs = QsSim(NUM_OF_CHANNELS)

    # set arrivals. The method needs to be passed distribution parameters and
    # type of distribution.
    qs.set_sources(ARRIVAL_RATE, "M")

    # set the service channels. To the method we pass parameters (in our case, service time)
    # and type of distribution - D (deterministic).
    qs.set_servers(b, "D")

    # start the simulation. The method takes the number of jobs to simulate.
    sim_results = qs.run(NUM_OF_JOBS)

    # Output results

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    probs_print(calc_results.p, sim_results.p)

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mdn()
