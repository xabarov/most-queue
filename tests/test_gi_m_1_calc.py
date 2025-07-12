"""
Test of GI/M/1 queueing system calculation.
For verification, we use simulation 
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gi_m_1 import GiM1

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params['arrival']['rate'])
ARRIVAL_CV = float(params['arrival']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])


def test_gi_m_1():
    """
    Test of GI/M/1 queueing system calculation.
    For verification, we use simulation 
    """

    a1 = 1 / ARRIVAL_RATE    # average interval between requests
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR  # service intensity

    # calculation of parameters approximating Gamma-distribution for arrival times
    gamma_params = GammaDistribution.get_params_by_mean_and_coev(
        a1, ARRIVAL_CV)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculation of initial moments of time spent and waiting in the queueing system
    gm1_calc = GiM1(a, mu)
    v_num = gm1_calc.get_v()
    w_num = gm1_calc.get_w()

    # calculation of probabilities of states in the queueing system
    p_num = gm1_calc.get_p()

    # for verification, we use sim.
    # create an instance of the sim class and pass the number of service channels
    qs = QsSim(1)

    # set the input stream. The method needs to be passed parameters
    # of distribution as a list and type of distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels. Parameters (in our case, the service intensity)
    # and type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start simulation
    qs.run(NUM_OF_JOBS)

    # get the list of initial moments of time spent and waiting in the queueing system
    v_sim = qs.v
    w_sim = qs.w

    # get the distribution of probabilities of states in the queueing system
    p_sim = qs.get_p()

    # Output results
    print("\nGamma\n")

    times_print(w_sim, w_num, True)
    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num)

    assert np.allclose(v_sim, v_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(p_sim[:10], p_num[:10],
                       rtol=PROBS_RTOL, atol=PROBS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_gi_m_1()
