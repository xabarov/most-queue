"""
Testing the M/G/1 queueing system calculation
For verification, we use simulation modeling 
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import H2Distribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calculation

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)


SERVICE_TIME_CV = float(params['service']['cv'])
ARRIVAL_RATE = float(params['arrival']['rate'])
NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])

NUM_OF_CHANNELS = 1


def test_mg1():
    """
    Testing the M/G/1 queueing system calculation
    For verification, we use simulation modeling
    """
    b1 = UTILIZATION_FACTOR*NUM_OF_CHANNELS/ARRIVAL_RATE

    # selecting parameters of the approximating H2-distribution
    # for service time H2Params [p1, mu1, mu2]:
    h2_params = H2Distribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CV)
    print(h2_params)
    b = H2Distribution.calc_theory_moments(h2_params, 4)

    # calculation using numerical methods
    mg1_num = MG1Calculation(ARRIVAL_RATE, b)
    w_num = mg1_num.get_w()
    p_num = mg1_num.get_p()
    v_num = mg1_num.get_v()

    # running IM for verification of results
    qs = QsSim(1)
    qs.set_servers(h2_params, "H")
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.run(NUM_OF_JOBS)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    # outputting the results
    print("M/H2/1")

    times_print(w_sim, w_num, True)
    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num, 10)

    assert np.allclose(w_sim, w_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(v_sim, v_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)

    assert np.allclose(np.array(p_sim[:10]), np.array(
        p_num[:10]), atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mg1()
