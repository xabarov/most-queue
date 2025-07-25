"""
Test the MMnPR2ClsBusyApprox class for priority queueing systems.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_raw_moments, probs_print
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import MMnPR2ClsBusyApprox

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params["num_of_channels"])

ARRIVAL_RATE_HIGH = float(params["arrival"]["rate"])
ARRIVAL_RATE_LOW = 1.4

SERVICE_TIME_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

UTILIZATION_HIGH = 0.4


def test_mmn_prty():
    """
    Test function for evaluating the performance of a priority queueing system approximation.
    This function compares simulation results with theoretical calculations using Cox approximation.
    """

    num_priority = 2  # Number of priority classes

    # Setting up arrival streams and server parameters
    sources = []
    servers_params = []

    # Defining arrival rates and service rates for each class
    arrival_rates = [ARRIVAL_RATE_HIGH, ARRIVAL_RATE_LOW]

    mu_high = ARRIVAL_RATE_HIGH / (NUM_OF_CHANNELS * UTILIZATION_HIGH)
    mu_low = ARRIVAL_RATE_LOW / (NUM_OF_CHANNELS * (UTILIZATION_FACTOR - UTILIZATION_HIGH))
    service_rates = [mu_high, mu_low]

    for j in range(num_priority):
        sources.append({"type": "M", "params": arrival_rates[j]})
        servers_params.append({"type": "M", "params": service_rates[j]})

    # Setting up the priority queueing system simulator
    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, num_priority, "PR")
    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # Running the simulation
    sim_results = qs.run(NUM_OF_JOBS)

    # Setting up and running the theoretical approximation model
    tt = MMnPR2ClsBusyApprox(n=NUM_OF_CHANNELS)
    tt.set_sources(l_low=ARRIVAL_RATE_LOW, l_high=ARRIVAL_RATE_HIGH)
    tt.set_servers(mu_low=mu_low, mu_high=mu_high)
    calc_results = tt.run()

    print(f"utilization: {calc_results.utilization:0.4f}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    # Printing comparison results
    print("\nComparison of theoretical calculation with Cox approximation and simulation results.")
    print(f"Utilization factor for low-priority class: {UTILIZATION_FACTOR:^6.2f}")
    print(f"Utilization factor for high-priority class: {UTILIZATION_HIGH:^6.2f}")

    print(f"Number of simulated jobs: {NUM_OF_JOBS:d}\n")

    print("Probs for low-priority class:")
    probs_print(p_sim=sim_results.p[1], p_num=calc_results.p, size=10)

    # Printing time moments comparison
    print_raw_moments(sim_moments=sim_results.v[0], calc_moments=calc_results.v[0], header="soujorn time for 1 class")
    print_raw_moments(sim_moments=sim_results.w[0], calc_moments=calc_results.w[0], header="waiting time for 1 class")
    print_raw_moments(
        sim_moments=[sim_results.v[1][0]],
        calc_moments=[calc_results.v[1][0]],
        header="mean soujorn time for 2 class",
    )

    # Asserting the accuracy of the results
    assert np.allclose(sim_results.v[1][0], calc_results.v[1][0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results.p[1][:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mmn_prty()
