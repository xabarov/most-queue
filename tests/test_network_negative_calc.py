"""
Test the queueing network with negative jobs calculation.
Compare results with numerical calculations using decomposition method.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments
from most_queue.random.distributions import H2Distribution
from most_queue.sim.negative import NegativeServiceType
from most_queue.sim.networks.negative_network import NegativeNetwork
from most_queue.theory.networks.negative_network import NegativeNetworkCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
# Network simulations converge slower; use a larger sample here.
NUM_OF_JOBS = int(params["num_of_jobs"]) * 3
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

# Negative arrival rate (global)
NEGATIVE_ARRIVAL_RATE = 0.3

# Negative types for each node (mix of DISASTER and RCS)
NEGATIVE_TYPES = [
    NegativeServiceType.DISASTER,
    NegativeServiceType.RCS,
    NegativeServiceType.DISASTER,
    NegativeServiceType.RCS,
    NegativeServiceType.DISASTER,
]


def test_negative_network():
    """
    Test the queueing network with negative jobs simulation.
    Compare results with numerical calculations using decomposition method.
    """
    b = []  # List of service time moments for each node
    serv_params = []

    for m in range(NUM_OF_NODES):
        b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS[m] / ARRIVAL_RATE

        h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
        serv_params.append({"type": "H", "params": h2_params})

        b.append(H2Distribution.calc_theory_moments(h2_params, 4))

    # Create calculation
    net_calc = NegativeNetworkCalc(negative_arrival_type="global")
    net_calc.set_sources(R=TRANSITION_MATRIX, arrival_rate=ARRIVAL_RATE, negative_arrival_rate=NEGATIVE_ARRIVAL_RATE)
    net_calc.set_nodes(b=b, n=NUM_OF_CHANNELS, negative_types=NEGATIVE_TYPES)
    num_results = net_calc.run()

    print(f"Intensities: {num_results.intensities}")

    # Get utilization factor of each node
    loads = num_results.loads

    # Create simulation
    qn = NegativeNetwork(negative_arrival_type="global")

    qn.set_sources(
        positive_arrival_rate=ARRIVAL_RATE,
        R=TRANSITION_MATRIX,
        negative_arrival_rate=NEGATIVE_ARRIVAL_RATE,
    )
    qn.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS, negative_types=NEGATIVE_TYPES)

    #  Run simulation
    sim_results = qn.run(NUM_OF_JOBS)

    print("-" * 60)
    print(f"Channels at nodes: {NUM_OF_CHANNELS}")
    print(f"Node utilization coefficients: {[float(round(load, 3)) for load in loads]}")
    print(f"Negative arrival rate: {NEGATIVE_ARRIVAL_RATE}")
    print(f"Negative types: {[nt.name for nt in NEGATIVE_TYPES]}")

    print("-" * 60)
    print("Results")
    print_sojourn_moments(sim_results.v, num_results.v)

    assert np.allclose(sim_results.v, num_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_negative_network_per_node():
    """
    Test the queueing network with negative jobs simulation (per_node mode).
    Compare results with numerical calculations using decomposition method.
    """
    b = []  # List of service time moments for each node
    serv_params = []

    for m in range(NUM_OF_NODES):
        b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS[m] / ARRIVAL_RATE

        h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
        serv_params.append({"type": "H", "params": h2_params})

        b.append(H2Distribution.calc_theory_moments(h2_params, 4))

    # Per-node negative arrival rates (different for each node)
    NEGATIVE_ARRIVAL_RATES = [0.1, 0.05, 0.15, 0.08, 0.12]

    # Create calculation
    net_calc = NegativeNetworkCalc(negative_arrival_type="per_node")
    net_calc.set_nodes(b=b, n=NUM_OF_CHANNELS, negative_types=NEGATIVE_TYPES)
    net_calc.set_sources(
        R=TRANSITION_MATRIX,
        arrival_rate=ARRIVAL_RATE,
        negative_arrival_rates=NEGATIVE_ARRIVAL_RATES,
    )
    num_results = net_calc.run()

    print(f"Intensities: {num_results.intensities}")

    # Get utilization factor of each node
    loads = num_results.loads

    # Create simulation
    qn = NegativeNetwork(negative_arrival_type="per_node")

    # For per_node type, set_nodes must be called before set_sources
    qn.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS, negative_types=NEGATIVE_TYPES)
    qn.set_sources(
        positive_arrival_rate=ARRIVAL_RATE,
        R=TRANSITION_MATRIX,
        negative_arrival_rates=NEGATIVE_ARRIVAL_RATES,
    )

    #  Run simulation
    sim_results = qn.run(NUM_OF_JOBS)

    print("-" * 60)
    print(f"Channels at nodes: {NUM_OF_CHANNELS}")
    print(f"Node utilization coefficients: {[float(round(load, 3)) for load in loads]}")
    print(f"Negative arrival rates (per node): {NEGATIVE_ARRIVAL_RATES}")
    print(f"Negative types: {[nt.name for nt in NEGATIVE_TYPES]}")

    print("-" * 60)
    print("Results (per_node mode)")
    print_sojourn_moments(sim_results.v, num_results.v)

    assert np.allclose(sim_results.v, num_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":

    test_negative_network()
    print("\n" + "=" * 60)
    print("Testing per_node mode")
    print("=" * 60 + "\n")
    test_negative_network_per_node()
