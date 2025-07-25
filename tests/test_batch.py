"""
Test module for batch arrival queueing systems.
Includes tests for the following models:
- Mx/M/1/infinite
The main function is test_batch_mm1, which tests the Mx/M/1/infinite model.
It compares the results of the simulation and the analytical solution.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments
from most_queue.sim.batch import QueueingSystemBatchSim
from most_queue.theory.batch.mm1 import BatchMM1

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = 1
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

BATCH_SIZE = 5
BATCH_PROBABILITIES = [0.2, 0.3, 0.1, 0.2, 0.2]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def calc_mean_batch_size(batch_probs):
    """
    Calc mean batch size
    batch_probs - probs of batch size 1, 2, .. len(batch_probs)
    """
    mean = 0
    for i, prob in enumerate(batch_probs):
        mean += (i + 1) * prob
    return mean


def test_batch_mm1():
    """
    Test QS Mx/M/1/infinite with batch arrivals
    """

    # probs of batch size 1, 2, .. 5
    mean_batch_size = calc_mean_batch_size(BATCH_PROBABILITIES)

    mu = ARRIVAL_RATE * mean_batch_size / UTILIZATION_FACTOR  # serving intensity

    batch_calc = BatchMM1()
    batch_calc.set_sources(l=ARRIVAL_RATE, batch_probs=BATCH_PROBABILITIES)
    batch_calc.set_servers(mu=mu)

    calc_results = batch_calc.run()
    print(f"Utilization: {calc_results.utilization: 0.4f}")

    qs = QueueingSystemBatchSim(NUM_OF_CHANNELS, BATCH_PROBABILITIES)

    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(mu, "M")

    sim_results = qs.run(NUM_OF_JOBS)

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print_sojourn_moments(sim_results.v[0], calc_results.v[0])

    assert np.allclose(sim_results.v[0], calc_results.v[0], atol=MOMENTS_ATOL, rtol=MOMENTS_RTOL), ERROR_MSG


if __name__ == "__main__":

    test_batch_mm1()
