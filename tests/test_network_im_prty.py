"""
Test the priority queueing network simulation with priority queues at nodes.
Compare results with numerical calculations using decomposition method.
"""

import os

import numpy as np
import yaml

from most_queue.general.tables import times_print_with_classes
from most_queue.rand_distribution import H2Distribution
from most_queue.sim.networks.priority_network import PriorityNetwork
from most_queue.theory.networks.open_network_prty import \
    OpenNetworkCalcPriorities

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

            h2_params.append(H2Distribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CVS[j]))
            serv_params[m].append({"type": "H", "params": h2_params[m]})

    for k in range(NUM_OF_CLASSES):
        b.append([])
        for m in range(NUM_OF_NODES):
            b[k].append(H2Distribution.calc_theory_moments(h2_params[m], 4))

    prty = ["NP"] * NUM_OF_NODES

    # Create simulation of priority network:
    qn = PriorityNetwork(
        NUM_OF_CLASSES,
        ARRIVAL_RATES,
        big_r,
        NUM_OF_CHANNELS,
        prty,
        serv_params,
        nodes_prty,
    )

    #  Run simulation of priority network:
    qn.run(NUM_OF_JOBS)

    # Get initial moments of soujorney time from simulation:
    v_sim = qn.v_network

    #  Get initial moments of soujorney time from calculation:
    net_calc = OpenNetworkCalcPriorities(big_r, b, NUM_OF_CHANNELS, ARRIVAL_RATES, prty, nodes_prty)
    net_calc = net_calc.run()
    v_num = net_calc["v"]

    # Get utilization factor of each node
    loads = net_calc["loads"]

    #  Print results

    print("-" * 60)
    print(f"Channels at nodes: {NUM_OF_CHANNELS}")
    print(f"Node utilization coefficients: {[float(round(load, 3)) for load in loads]}")

    print("-" * 60)
    print("Relative Priority ('NP')")
    times_print_with_classes(v_sim, v_num, False)

    assert abs(v_sim[0][0] - v_num[0][0] < 2.0), ERROR_MSG
    assert abs(v_sim[1][0] - v_num[1][0] < 2.0), ERROR_MSG
    assert abs(v_sim[2][0] - v_num[2][0] < 2.0), ERROR_MSG

    prty = ["PR"] * NUM_OF_NODES  # Absolute priority at each node
    qn = PriorityNetwork(
        NUM_OF_CLASSES,
        ARRIVAL_RATES,
        big_r,
        NUM_OF_CHANNELS,
        prty,
        serv_params,
        nodes_prty,
    )
    qn.run(NUM_OF_JOBS)
    v_sim = qn.v_network

    net_calc = OpenNetworkCalcPriorities(big_r, b, NUM_OF_CHANNELS, ARRIVAL_RATES, prty, nodes_prty)
    net_calc = net_calc.run()
    v_num = net_calc["v"]

    print("-" * 60)
    print("Absolute Priority ('PR')")
    times_print_with_classes(v_sim, v_num, False)

    assert abs(v_sim[0][0] - v_num[0][0] < 2.0), ERROR_MSG
    assert abs(v_sim[1][0] - v_num[1][0] < 2.0), ERROR_MSG
    assert abs(v_sim[2][0] - v_num[2][0] < 2.0), ERROR_MSG


if __name__ == "__main__":

    test_network()
