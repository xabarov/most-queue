"""
Test QS M/G/1 queue with disasters.
Under development
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import times_print_with_two_numerical
from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_disasters import MG1Disasters
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc

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


def test_mg1():
    """
    Test QS M/G/1 queue with disasters.
    Compare theoretical and simulated moments.
    MG1 calculate moments using Pollaczek-Khintchine formula.
        Jain, Gautam, and Karl Sigman. "A Pollaczek–Khintchine formula 
        for M/G/1 queues with disasters."
        Journal of Applied Probability 33.4 (1996): 1191-1200.
    T-T is ours method (based on Takahasi-Takagi)
    """

    b1 = 1 * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time

    approximation = 'gamma'

    if approximation == 'h2':
        b_params = H2Distribution.get_params_by_mean_and_coev(
            b1, SERVICE_TIME_CV)
        b = H2Distribution.calc_theory_moments(b_params)
    else:
        b_params = GammaDistribution.get_params_by_mean_and_coev(
            b1, SERVICE_TIME_CV)
        b = GammaDistribution.calc_theory_moments(b_params)

    # Run calc
    mg1_queue_calc = MG1Disasters(
        ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE,
        b, approximation=approximation)

    v_calc1 = mg1_queue_calc.get_v()

    mgn_queue_calc = MGnNegativeDisasterCalc(
        1, ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE, b)

    mgn_queue_calc.run()

    v_calc_tt = mgn_queue_calc.get_v()

    # Run simulation
    queue_sim = QsSimNegatives(
        1, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, 'M')
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, 'M')

    if approximation == 'h2':
        queue_sim.set_servers(b_params, 'H')
    else:
        queue_sim.set_servers(b_params, 'Gamma')

    queue_sim.run(NUM_OF_JOBS)

    v_sim = queue_sim.get_v()

    times_print_with_two_numerical(v_sim, v_calc1, v_calc_tt, is_w=False,
                                   num1_header='MG1', num2_header='T-T')

    assert np.allclose(v_sim, v_calc_tt, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    # when MG1 will work, add assert with v_calc1


if __name__ == "__main__":
    test_mg1()
