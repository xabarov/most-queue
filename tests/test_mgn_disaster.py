"""
Test QS M/G/n queue with disasters.
"""
import math
import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params['num_of_channels'])

SERVICE_TIME_CV = float(params['service']['cv'])
NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

ARRIVAL_RATE_POSITIVE = float(params['arrival']['rate'])
ARRIVAL_RATE_NEGATIVE = 0.3*ARRIVAL_RATE_POSITIVE


def test_mgn():
    """
    Test QS M/G/n queue with disasters.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / \
        ARRIVAL_RATE_POSITIVE  # average service time

    b = [0.0] * 3
    alpha = 1 / (SERVICE_TIME_CV ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(SERVICE_TIME_CV, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    # Run simulation
    queue_sim = QsSimNegatives(
        NUM_OF_CHANNELS, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, 'M')
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(NUM_OF_JOBS)

    p_sim = queue_sim.get_p()
    v_sim = queue_sim.get_v()
    v_sim_served = queue_sim.get_v_served()
    v_sim_broken = queue_sim.get_v_broken()
    w_sim = queue_sim.get_w()

    # Run calc
    queue_calc = MGnNegativeDisasterCalc(
        NUM_OF_CHANNELS, ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE,
        b, verbose=False, accuracy=1e-8)

    queue_calc.run()

    p_calc = queue_calc.get_p()
    v_calc = queue_calc.get_v()
    v_calc_served = queue_calc.get_v_served()
    v_calc_broken = queue_calc.get_v_broken()
    w_calc = queue_calc.get_w()

    probs_print(p_sim, p_calc)
    times_print(v_sim, v_calc, is_w=False, header='sojourn total')
    times_print(v_sim_served, v_calc_served,
                is_w=False, header='sojourn served')
    times_print(v_sim_broken, v_calc_broken,
                is_w=False, header='sojourn broken')
    times_print(w_sim, w_calc)

    assert np.allclose(v_sim, v_calc, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(p_sim[:10], p_calc[:10],
                       atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mgn()
