"""
Test the NetworkOptimizerWithApprox class.
"""
import numpy as np

from most_queue.rand_distribution import ExpDistribution
from most_queue.theory.networks.opt.transition_approx import NetworkOptimizerWithApprox


def test_network_opt_approx():
    """
    Test the NetworkOptimizerWithApprox class.
    """
    num_of_nodes = 6
    num_of_channels = [1, 3, 2, 1, 3, 2]
    arrival_rate = 3.8
    transition_mrx = np.matrix([
        [0.2, 0.7, 0.1, 0, 0, 0, 0],
        [0, 0.3, 0, 0.3, 0.4, 0, 0],
        [0, 0, 0.5, 0.2, 0.3, 0, 0],
        [0, 0, 0, 0, 0.7, 0.3, 0],
        [0.2, 0, 0, 0, 0.4, 0.2, 0.2],
        [0, 0, 0, 0, 0, 0.4, 0.6],
        [0, 0, 0, 0, 0, 0, 1]
    ])

    max_ends = [0, 0, 0, 0, 0.3, 0.6, 1.0]

    service_times = [ExpDistribution.calc_theory_moments(1.0) for _m in range(
        num_of_nodes)]  # List of service time moments for each node

    optimizer = NetworkOptimizerWithApprox(
        transition_matrix=transition_mrx, arrival_rate=arrival_rate,
        b=service_times, num_channels=num_of_channels,
        maximum_rates_to_end=max_ends,
        is_service_markovian=True, verbose=True)

    _best_transition_matrix, best_v1 = optimizer.run()

    assert best_v1 < 16.0, f'The best V1 value should be less than 16.0. Found: {best_v1}'


if __name__ == "__main__":
    test_network_opt_approx()
