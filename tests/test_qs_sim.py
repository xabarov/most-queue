"""
Test the simulation of a queueing system
For verification, compare with results for M/M/3
"""

import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params["num_of_channels"])

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_TIME_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

QUEUE_LENGTH = 30


def test_sim():
    """
    Test the simulation of a queueing system
    For verification, compare with results for M/M/3 and M/D/3 systems
    """
    # Calculate service rate based on utilization
    service_rate = ARRIVAL_RATE / (NUM_OF_CHANNELS * UTILIZATION_FACTOR)

    # Initialize simulation model
    qs = QsSim(NUM_OF_CHANNELS, buffer=QUEUE_LENGTH)

    # Set arrival process parameters and distribution (M for Markovian)
    qs.set_sources(ARRIVAL_RATE, "M")

    # Set service time parameters and distribution (M for Markovian)
    qs.set_servers(service_rate, "M")

    # Run simulation with 300,000 arrivals
    qs.run(NUM_OF_JOBS)

    # Get simulated waiting times
    w_sim = qs.w
    p_sim = qs.get_p()

    # Calculate theoretical waiting times using MMnr model
    mmnr = MMnrCalc(ARRIVAL_RATE, service_rate, NUM_OF_CHANNELS, QUEUE_LENGTH)
    w_num = mmnr.get_w()
    p_num = mmnr.get_p()

    # Print comparison of simulation and theoretical results
    times_print(w_sim, w_num, True)
    probs_print(p_num=p_num, p_sim=p_sim, size=10)

    assert np.allclose(w_sim, w_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(p_sim[:10], p_num[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_sim()
