"""
Test M/G/1 queue with warm-up phase.
Compare theoretical and simulated moments.
"""
import numpy as np

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.mg1_warm_calc import MG1WarmCalc


def calculate_gamma_moments(mean, cv):
    """
    Helper function to calculate Gamma distribution parameters.
    """
    alpha = 1 / (cv ** 2)
    b1 = mean
    b2 = (b1 ** 2) * (cv ** 2 + 1)
    b3 = b2 * b1 * (1 + 2 / alpha)

    return [b1, b2, b3]


ARRIVAL_RATE = 1.0
NUM_OF_JOBS = 300_000
SERVICE_TIME_CV = 1.57  # Service time CV for the warm-up phase

UTILIZATION = 0.7  # System utilization

MEAN_WARMUP_FACTOR = 1.5  # Mean time for warm-up phase is factor*mean service time
WARMUP_CV = 1.57  # Warm-up service time CV


def test_mg1_warm():
    """
    Test M/G/1 queue with warm-up phase.
    Compare theoretical and simulated moments.
    """

    b1 = UTILIZATION / ARRIVAL_RATE
    b_s = calculate_gamma_moments(b1, SERVICE_TIME_CV)
    service_params = GammaDistribution.get_params(b_s)

    # Warm phase parameters
    mean_warmup_time = b1*MEAN_WARMUP_FACTOR
    b_w = calculate_gamma_moments(mean_warmup_time, WARMUP_CV)
    warmup_params = GammaDistribution.get_params(b_w)

    # Initialize the Vacation Queueing System Simulator
    simulator = VacationQueueingSystemSimulator(1, is_service_on_warm_up=True)

    # Set warm-up phase parameters
    simulator.set_servers(service_params, 'Gamma')
    simulator.set_warm(warmup_params, 'Gamma')

    # Configure the simulator
    simulator.set_sources(ARRIVAL_RATE, 'M')

    # Run simulations
    simulator.run(NUM_OF_JOBS)
    v_sim = simulator.get_v()

    mg1_calc = MG1WarmCalc(ARRIVAL_RATE, b_s, b_w)
    v_num = mg1_calc.get_v()

    times_print(v_sim, v_num, is_w=False)

    # assert all close with relative percent 20%
    assert np.allclose(v_sim, v_num, rtol=0.2)


if __name__ == "__main__":
    test_mg1_warm()
