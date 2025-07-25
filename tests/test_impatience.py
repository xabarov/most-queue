"""
Test for M/M/1 queue with exponential impatience.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, probs_print
from most_queue.sim.impatient import ImpatientQueueSim
from most_queue.theory.impatience.mm1 import MM1Impatience

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


ARRIVAL_RATE = float(params["arrival"]["rate"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

NUM_OF_JOBS = int(params["num_of_jobs"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

NUM_OF_CHANNELS = 1
IMPATIENCE_RATE = 0.2


def test_impatience():
    """
    Test for M/M/1 queue with exponential impatience.
    """
    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)  # service rate

    # Calculate theoretical results
    imp_calc = MM1Impatience(gamma=IMPATIENCE_RATE)
    imp_calc.set_sources(ARRIVAL_RATE)
    imp_calc.set_servers(mu)
    calc_results = imp_calc.run()

    # Simulate the queue
    qs = ImpatientQueueSim(NUM_OF_CHANNELS)

    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(mu, "M")
    qs.set_impatience(IMPATIENCE_RATE, "M")

    sim_results = qs.run(NUM_OF_JOBS)

    # Print results

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print_sojourn_moments(sim_results.v[0], calc_results.v[0])
    probs_print(sim_results.p, calc_results.p)

    assert abs(calc_results.v[0] - sim_results.v[0]) < 0.02

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], rtol=PROBS_RTOL, atol=PROBS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_impatience()
