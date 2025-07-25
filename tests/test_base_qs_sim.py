"""
Test the simulation model for an M/M/n/r system
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_waiting_moments, probs_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_d_n import MDn
from most_queue.theory.fifo.mmnr import MMnrCalc

# Open config.yaml

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
BUFFER = int(params["buffer"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def test_sim_mmnr():
    """
    Test the simulation model for an M/M/n/r system
    """
    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)  # Service intensity

    # Create simulation instance
    qs = QsSim(NUM_OF_CHANNELS, buffer=BUFFER)

    # Set arrival process parameters and distribution as exponential
    qs.set_sources(ARRIVAL_RATE, "M")
    # Set service time parameters and distribution as exponential
    qs.set_servers(mu, "M")

    # Run the simulation
    sim_results = qs.run(NUM_OF_JOBS)

    mmnr = MMnrCalc(n=NUM_OF_CHANNELS, r=BUFFER)
    mmnr.set_sources(l=ARRIVAL_RATE)
    mmnr.set_servers(mu=mu)
    calc_results = mmnr.run()

    print_waiting_moments(sim_results.w, calc_results.w)

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    # Verify simulation results against theoretical calculations
    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_sim_mdn():
    """
    Test the simulation model for a M/D/n
    """
    qs = QsSim(NUM_OF_CHANNELS)

    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)  # Service intensity

    qs.set_sources(ARRIVAL_RATE, "M")
    # Using same load coefficient as before
    qs.set_servers(1.0 / mu, "D")

    qs.run(NUM_OF_JOBS)

    mdn = MDn(n=NUM_OF_CHANNELS)
    mdn.set_sources(l=ARRIVAL_RATE)
    mdn.set_servers(b=1.0 / mu)
    mdn_results = mdn.run()
    p_sim = qs.get_p()

    probs_print(p_sim=p_sim, p_num=mdn_results.p, size=10)

    assert np.allclose(p_sim[:10], mdn_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_sim_mmnr()
    test_sim_mdn()
