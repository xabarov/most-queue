"""
Test QS M/G/n queue with disasters.
"""
import math

from most_queue.general.tables import times_print
from most_queue.rand_distribution import H2Distribution
from most_queue.sim.queueing_systems.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.queueing_systems.negative.mg1_disasters import MG1Disasters


def test_mg1():
    """
    Test QS M/G/1 queue with disasters.
    """

    l_pos = 1.0  # arrival rate of positive jobs
    l_neg = 0.3  # arrival rate of negative jobs
    n = 1
    num_of_jobs = 100_000
    ro = 0.7
    b1 = n * ro / l_pos  # average service time
    b_coev = 1.57

    b = [0.0] * 3
    alpha = 1 / (b_coev ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    print(f'Service time moments: {b}')

    # Run calc
    queue_calc = MG1Disasters(l_pos, l_neg, b)

    v_calc = queue_calc.get_v()

    # Run simulation
    queue_sim = QsSimNegatives(
        n, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    h2_params = H2Distribution.get_params(b)
    queue_sim.set_servers(h2_params, 'H')

    queue_sim.run(num_of_jobs)

    v_sim = queue_sim.get_v()

    times_print(v_sim, v_calc, is_w=False)


if __name__ == "__main__":
    test_mg1()
