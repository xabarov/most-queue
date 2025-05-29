"""
Compare queue implementation performance
"""
import time

from most_queue.rand_distribution import (
    GammaDistribution,  # For generating gamma distribution parameters
)
from most_queue.sim.base import QsSim  # Main queueing system simulation class


def compare_calc_times():
    """
    Compare calculation times for different buffer types in the queueing system.

    Returns:
        Prints comparison results of 'list' and 'deque' buffer types performance.
    """
    num_channels = 3              # Number of service channels
    arrival_rate = 1.0           # Arrival rate intensity
    utilization_factor = 0.7     # System utilization factor
    mean_service_time = (num_channels * utilization_factor) / arrival_rate
    num_jobs = 1_000_000         # Number of jobs for simulation
    variation_coeff = 1.2        # Coefficient of variation for service time

    buffer_types = ['list', 'deque']

    execution_times = {'list': 0, 'deque': 0}

    for buffer_type in buffer_types:
        # Calculate initial moments of service time based on mean and variation coefficient
        moments = [0.0] * 3
        alpha = 1 / (variation_coeff ** 2)

        moments[0] = mean_service_time
        moments[1] = (moments[0] ** 2) * (variation_coeff ** 2 + 1)
        moments[2] = moments[1] * moments[0] * (1.0 + 2 / alpha)

        # Start simulation timing for this buffer type
        sim_start_time = time.process_time()

        # Initialize queueing system with specified buffer type
        qs = QsSim(num_channels, buffer_type=buffer_type)

        # Configure arrival process (Exponential distribution with rate λ)
        qs.set_sources(arrival_rate, kendall_notation='M')

        # Configure service process (Gamma distribution with parameters derived from moments)
        gamma_params = GammaDistribution.get_params(moments[:2])
        qs.set_servers(gamma_params, kendall_notation='Gamma')

        # Run the simulation
        qs.run(num_jobs)

        # Record execution time for this buffer type
        execution_times[buffer_type] = time.process_time() - sim_start_time

    print("Execution times:", execution_times)


if __name__ == "__main__":
    compare_calc_times()
