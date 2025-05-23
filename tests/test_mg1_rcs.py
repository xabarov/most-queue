"""
Test the M/H2/1 and M/Gamma/1 queueing systems with RCS discipline.
"""
import math

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution, H2Distribution
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


def test_mg1_h2_rcs():
    """
    Test the  M/H2/1 queueing systems with RCS discipline.
    """

    num_of_jobs = 300_000
    l_neg = 0.3
    l_pos = 0.4
    ro = 0.8
    n = 1
    b_coev = 1.2

    b = calc_service_moments(ro, b_coev, l_pos)

    h2_params = H2Distribution.get_params(b)

    # Run simulation
    queue_sim = QsSimNegatives(
        n, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    queue_sim.set_servers(h2_params, 'H')

    queue_sim.run(num_of_jobs)

    v_sim = queue_sim.get_v()

    m_hyper_1_calc = MG1NegativeCalcRCS(
        l_pos, l_neg, b, service_time_approx_dist='h2')
    v1_h2_calc = m_hyper_1_calc.get_v1()

    times_print(v_sim[0], v1_h2_calc, is_w=False, header='sojourn time H2')


def test_mg1_gamma_rcs():
    """
    Test the  M/Gamma/1 queueing systems with RCS discipline.
    """

    num_of_jobs = 300_000
    l_neg = 0.9
    l_pos = 1.0
    ro = 0.7
    n = 1
    b_coev = 2.1

    b = calc_service_moments(ro, b_coev, l_pos)

    # Run simulation
    queue_sim = QsSimNegatives(
        n, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(num_of_jobs)

    v_sim = queue_sim.get_v()

    m_gamma_1_calc = MG1NegativeCalcRCS(
        l_pos, l_neg, b, service_time_approx_dist='gamma')
    v1_gamma_calc = m_gamma_1_calc.get_v1()

    times_print(v_sim[0], v1_gamma_calc, is_w=False,
                header='sojourn time Gamma')


if __name__ == "__main__":
    test_mg1_gamma_rcs()
    test_mg1_h2_rcs()
