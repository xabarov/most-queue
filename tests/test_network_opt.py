"""
Test the NetworkOptimizer class.
@article{рыжиков2019численные,
    title={Численные методы теории очередей},
    author={Рыжиков, ЮИ},
    journal={Учебное пособие--М.: Лань},
    year={2019}
}
"""
import numpy as np

from most_queue.general.tables import print_mrx
from most_queue.rand_distribution import ExpDistribution
from most_queue.theory.networks.opt.transition import NetworkOptimizer


def test_network_opt():
    """
    Test the NetworkOptimizer class.
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

    optimizer = NetworkOptimizer(
        transition_matrix=transition_mrx, arrival_rate=arrival_rate,
        b=service_times, num_channels=num_of_channels,
        maximum_rates_to_end=max_ends,
        is_service_markovian=True, verbose=True)
    best_r, v1 = optimizer.run()

    print_mrx(best_r)
    print(f"Best mean sojourn time: {v1:0.2f}")


if __name__ == "__main__":
    test_network_opt()
