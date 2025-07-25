"""
Test the NetworkOptimizerPlus class.
"""

import numpy as np

from most_queue.random.distributions import ExpDistribution
from most_queue.theory.networks.open_network import OpenNetworkCalc
from most_queue.theory.networks.opt.transition_plus import NetworkOptimizerPlus, Strategy


def test_network_opt_plus():
    """
    Test the NetworkOptimizer class.
    """
    num_of_nodes = 6
    num_of_channels = [1, 3, 2, 1, 3, 2]
    arrival_rate = 3.8
    transition_mrx = np.matrix(
        [
            [0.2, 0.7, 0.1, 0, 0, 0, 0],
            [0, 0.3, 0, 0.3, 0.4, 0, 0],
            [0, 0, 0.5, 0.2, 0.3, 0, 0],
            [0, 0, 0, 0, 0.7, 0.3, 0],
            [0.2, 0, 0, 0, 0.4, 0.2, 0.2],
            [0, 0, 0, 0, 0, 0.4, 0.6],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )

    max_ends = [0, 0, 0, 0, 0.3, 0.6, 1.0]

    service_times = [
        ExpDistribution.calc_theory_moments(1.0) for _m in range(num_of_nodes)
    ]  # List of service time moments for each node

    top_k = 3
    for strategy in [
        Strategy.MIN_AND_MAX,
        Strategy.TOP_ONE,
        Strategy.RANDOM,
        Strategy.TOP_K,
        Strategy.ALL,
    ]:
        network = OpenNetworkCalc()
        network.set_sources(R=transition_mrx, arrival_rate=arrival_rate)
        network.set_nodes(b=service_times, n=num_of_channels)
        optimizer = NetworkOptimizerPlus(
            network=network,
            maximum_rates_to_end=max_ends,
            is_service_markovian=True,
            verbose=False,
        )

        _, best_v1 = optimizer.run(strategy=strategy, top_k=top_k)
        print("\n")
        optimizer.print_last_state(header=f"Strategy: {strategy}")

        assert best_v1 < 18.0, f"Best value for strategy {strategy} is too high."


if __name__ == "__main__":
    test_network_opt_plus()
