"""
Test the M/H2/1 and M/Gamma/1 queueing systems with RCS discipline.
"""
import math
import os

import numpy as np
import yaml

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_rcs import MG1NegativeCalcRCS

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV = float(params['service']['cv'])
NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])

ARRIVAL_RATE_POSITIVE = float(params['arrival']['rate'])
ARRIVAL_RATE_NEGATIVE = 0.8*ARRIVAL_RATE_POSITIVE


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


def test_mg1_gamma_rcs():
    """
    Test the  M/Gamma/1 queueing systems with RCS discipline.
    """

    b = calc_service_moments(
        UTILIZATION_FACTOR, SERVICE_TIME_CV, ARRIVAL_RATE_POSITIVE)

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
                header='Sojourn time in M/G/1 with RCS disasters')

    # assert is all close with rtol 10%
    assert np.allclose(v_sim[0], v1_gamma_calc,
                       rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_mg1_gamma_rcs()
