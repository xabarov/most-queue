"""
Test the M/H2/1 and M/Gamma/1 queueing systems with RCS discipline.
"""
import math

import numpy as np

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_rcs import MG1NegativeCalcRCS


def calc_service_moments(utilization_factor: float,
                         service_time_variation_coef: float,
                         l_pos: float):
    """
    Gamma service time moments calculation.
    """
    b1 = 1 * utilization_factor / l_pos  # average service time

    b = [0.0] * 3
    alpha = 1 / (service_time_variation_coef ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * \
        (math.pow(service_time_variation_coef, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    return b

ARRIVAL_RATE_POSITIVE = 1.0
ARRIVAL_RATE_NEGATIVE = 0.8
NUM_OF_JOBS = 300000
SERVICE_TIME_CV = 2.15
UTILIZATION = 0.7



def test_mg1_gamma_rcs():
    """
    Test the  M/Gamma/1 queueing systems with RCS discipline.
    """

    b = calc_service_moments(UTILIZATION, SERVICE_TIME_CV, ARRIVAL_RATE_POSITIVE)

    # Run simulation
    queue_sim = QsSimNegatives(
        1, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, 'M')
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(NUM_OF_JOBS)

    v_sim = queue_sim.get_v()

    m_gamma_1_calc = MG1NegativeCalcRCS(
        ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE, b, service_time_approx_dist='gamma')
    v1_gamma_calc = m_gamma_1_calc.get_v1()

    times_print(v_sim[0], v1_gamma_calc, is_w=False,
                header='sojourn time Gamma')

    # assert is all close with rtol 10%
    assert np.allclose(v_sim[0], v1_gamma_calc, rtol=0.1)


if __name__ == "__main__":
    test_mg1_gamma_rcs()
