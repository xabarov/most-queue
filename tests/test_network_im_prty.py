"""
Test the priority queueing network simulation with priority queues at nodes.
Compare results with numerical calculations using decomposition method.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_multiclass
from most_queue.random.distributions import H2Distribution
from most_queue.sim.networks.priority_network import PriorityNetwork
from most_queue.theory.networks.open_network_prty import OpenNetworkCalcPriorities

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_JOBS = int(params["num_of_jobs"])
ERROR_MSG = params["error_msg"]

NUM_OF_CLASSES = 3
NUM_OF_NODES = 5
# Distribution of channels (servers) at each node
NUM_OF_CHANNELS = [3, 2, 3, 4, 3]
ARRIVAL_RATES = [0.1, 0.3, 0.4]  # Arrival rates for each job class
TRANSITION_MATRIX_FIRST_CLS = np.matrix(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0.4, 0.6, 0, 0, 0],
        [0, 0, 0.2, 0.4, 0.4, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
TRANSITION_MATRIX_SECOND_CLS = np.matrix(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0.5, 0.3, 0, 0, 0.2],
        [0, 0, 0, 0.7, 0.3, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)
TRANSITION_MATRIX_THIRD_CLS = np.matrix(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0.6, 0.4, 0, 0, 0],
        [0, 0, 0, 0.8, 0.2, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)

# Coefficient of variation for service times for each job class
SERVICE_TIME_CVS = [1.5, 1.3, 1.4]


def test_network():
    """
    Test the priority queueing network simulation  with priority queues at nodes.
    Compare results with numerical calculations using decomposition method.
    """
    # List of transition probability matrices for each class
    big_r = [
        TRANSITION_MATRIX_FIRST_CLS,
        TRANSITION_MATRIX_SECOND_CLS,
        TRANSITION_MATRIX_THIRD_CLS,
    ]

    b = []  # List of service time moments for each class and node
    nodes_prty = []  # Priority distribution at each node: [node][class]

    serv_params = []

    h2_params = []
    for m in range(NUM_OF_NODES):
        nodes_prty.append([])
        b1 = 0.7 * NUM_OF_CHANNELS[m] / sum(ARRIVAL_RATES)
        serv_params.append([])
        for j in range(NUM_OF_CLASSES):
            if m % 2 == 0:
                nodes_prty[m].append(j)
            else:
                nodes_prty[m].append(NUM_OF_CLASSES - j - 1)

            h2_params.append(H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CVS[j]))
            serv_params[m].append({"type": "H", "params": h2_params[m]})

    for k in range(NUM_OF_CLASSES):
        b.append([])
        for m in range(NUM_OF_NODES):
            b[k].append(H2Distribution.calc_theory_moments(h2_params[m], 4))

    prty = ["NP"] * NUM_OF_NODES

    # Create simulation of priority network:
    qn = PriorityNetwork(NUM_OF_CLASSES)

    qn.set_sources(L=ARRIVAL_RATES, R=big_r)
    qn.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS, prty=prty, nodes_prty=nodes_prty)

    #  Run simulation of priority network:
    sim_results = qn.run(NUM_OF_JOBS)

    #  Get raw moments of soujorney time from calculation:
    net_calc = OpenNetworkCalcPriorities()
    net_calc.set_sources(R=big_r, L=ARRIVAL_RATES)
    net_calc.set_nodes(n=NUM_OF_CHANNELS, b=b, prty=prty, nodes_prty=nodes_prty)
    calc_results = net_calc.run()

    # Get utilization factor of each node
    loads = calc_results.loads

    #  Print results
    print("-" * 60)
    print(f"Channels at nodes: {NUM_OF_CHANNELS}")
    print(f"Node utilization coefficients: {[float(round(load, 3)) for load in loads]}")

    print("-" * 60)
    print("Relative Priority ('NP')")
    print_sojourn_multiclass(sim_results.v, calc_results.v)

    assert abs(sim_results.v[0][0] - calc_results.v[0][0] < 2.0), ERROR_MSG
    assert abs(sim_results.v[1][0] - calc_results.v[1][0] < 2.0), ERROR_MSG
    assert abs(sim_results.v[2][0] - calc_results.v[2][0] < 2.0), ERROR_MSG

    prty = ["PR"] * NUM_OF_NODES  # Absolute priority at each node
    qn = PriorityNetwork(NUM_OF_CLASSES)

    qn.set_sources(L=ARRIVAL_RATES, R=big_r)
    qn.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS, prty=prty, nodes_prty=nodes_prty)
    sim_results = qn.run(NUM_OF_JOBS)

    net_calc = OpenNetworkCalcPriorities()
    net_calc.set_sources(R=big_r, L=ARRIVAL_RATES)
    net_calc.set_nodes(n=NUM_OF_CHANNELS, b=b, prty=prty, nodes_prty=nodes_prty)
    calc_results = net_calc.run()

    print("-" * 60)
    print("Absolute Priority ('PR')")
    print_sojourn_multiclass(sim_results.v, calc_results.v)

    assert abs(sim_results.v[0][0] - calc_results.v[0][0] < 2.0), ERROR_MSG
    assert abs(sim_results.v[1][0] - calc_results.v[1][0] < 2.0), ERROR_MSG
    assert abs(sim_results.v[2][0] - calc_results.v[2][0] < 2.0), ERROR_MSG


if __name__ == "__main__":

    test_network()
