"""
Test the MMn_PRTY_PNZ_Cox_approx class for priority queueing systems.
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import \
    MMn_PRTY_PNZ_Cox_approx

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params['num_of_channels'])

ARRIVAL_RATE_HIGH = float(params['arrival']['rate'])
ARRIVAL_RATE_LOW = 1.4

SERVICE_TIME_CV = float(params['service']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])

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

    mu_high = ARRIVAL_RATE_HIGH / (NUM_OF_CHANNELS*UTILIZATION_HIGH)
    mu_low = ARRIVAL_RATE_LOW / \
        (NUM_OF_CHANNELS*(UTILIZATION_FACTOR - UTILIZATION_HIGH))
    service_rates = [mu_high, mu_low]

    for j in range(num_priority):
        sources.append({'type': 'M', 'params': arrival_rates[j]})
        servers_params.append({'type': 'M', 'params': service_rates[j]})

    # Setting up the priority queueing system simulator
    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, num_priority, "PR")
    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # Running the simulation
    qs.run(NUM_OF_JOBS)

    # Getting simulation results
    p_sim = qs.get_p()
    v_sim = qs.v

    # Setting up and running the theoretical approximation model
    tt = MMn_PRTY_PNZ_Cox_approx(
        NUM_OF_CHANNELS, mu_low, mu_high, ARRIVAL_RATE_LOW, ARRIVAL_RATE_HIGH)
    tt.run()
    p_num = tt.get_p()
    v_num = tt.get_second_class_v1()

    # Printing comparison results
    print("\nComparison of theoretical calculation with Cox approximation and simulation results.")
    print(
        f"Utilization factor for low-priority class: {UTILIZATION_FACTOR:^6.2f}")
    print(
        f"Utilization factor for high-priority class: {UTILIZATION_HIGH:^6.2f}")

    print(f"Number of simulated jobs: {NUM_OF_JOBS:d}\n")

    print("Probabilities of system states for low-priority class")

    # Printing probability comparison table
    probs_print(p_sim=p_sim[1], p_num=p_num, size=10)

    # Printing time moments comparison
    times_print(sim_moments=[v_sim[1][0]], calc_moments=[v_num], is_w=False)

    # Asserting the accuracy of the results
    assert np.allclose(
        v_sim[1][0], v_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(p_sim[1][:10], p_num[:10],
                       atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mmn_prty()
