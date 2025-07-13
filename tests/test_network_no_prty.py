"""
Test the queueing network simulation.
Compare results with numerical calculations using decomposition method.
"""

import os

import numpy as np
import yaml

from most_queue.general.tables import times_print
from most_queue.rand_distribution import H2Distribution
from most_queue.sim.networks.network import NetworkSimulator
from most_queue.theory.networks.open_network import OpenNetworkCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_JOBS = int(params["num_of_jobs"])
SERVICE_TIME_CV = float(params["service"]["cv"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

NUM_OF_NODES = 5
# Distribution of channels (servers) at each node
NUM_OF_CHANNELS = [3, 2, 3, 4, 3]

TRANSITION_MATRIX = np.matrix(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0.4, 0.6, 0, 0, 0],
        [0, 0, 0.2, 0.4, 0.4, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)


def test_network():
    """
    Test the queueing network simulation.
    Compare results with numerical calculations using decomposition method.
    """
    b = []  # List of service time moments for each node
    serv_params = []

    for m in range(NUM_OF_NODES):
        b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS[m] / ARRIVAL_RATE

        h2_params = H2Distribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CV)
        serv_params.append({"type": "H", "params": h2_params})

        b.append(H2Distribution.calc_theory_moments(h2_params, 4))

    net_calc = OpenNetworkCalc(TRANSITION_MATRIX, b, NUM_OF_CHANNELS, ARRIVAL_RATE)
    net_calc = net_calc.run()
    v_num = net_calc["v"]

    # Get utilization factor of each node
    loads = net_calc["loads"]

    # Create simulation
    qn = NetworkSimulator(ARRIVAL_RATE, TRANSITION_MATRIX, NUM_OF_CHANNELS, serv_params)

    #  Run simulation
    qn.run(NUM_OF_JOBS)

    # Get initial moments of sojourn time from simulation:
    v_sim = qn.v_network

    print("-" * 60)
    print(f"Channels at nodes: {NUM_OF_CHANNELS}")
    print(f"Node utilization coefficients: {[float(round(load, 3)) for load in loads]}")

    print("-" * 60)
    print("Results")
    times_print(v_sim, v_num, False)

    assert np.allclose(v_sim, v_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":

    test_network()
