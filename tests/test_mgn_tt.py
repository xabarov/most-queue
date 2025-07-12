"""
Testing the Takahasi-Takami method for calculating an M/H2/n queue

When the coefficient of variation of service time is less than 1, 
the parameters of the approximating H2 distribution
are complex, which does not prevent obtaining meaningful results.

For verification, simulation is used.

"""
import math
import os
import time

import numpy as np
import yaml

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mgn_takahasi import MGnCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params['num_of_channels'])

ARRIVAL_RATE = float(params['arrival']['rate'])
SERVICE_TIME_CV = float(params['service']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])


def test_mgn_tt():
    """
    Testing the Takahasi-Takami method for calculating an M/H2/n queue
    """

    # calculate initial moments of service time based
    # on the given average and coefficient of variation
    b = [0.0] * 3
    alpha = 1 / (SERVICE_TIME_CV ** 2)
    b[0] = NUM_OF_CHANNELS * UTILIZATION_FACTOR / \
        ARRIVAL_RATE  # average service time
    b[1] = math.pow(b[0], 2) * (math.pow(SERVICE_TIME_CV, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    tt_start = time.process_time()
    # run Takahasi-Takami method
    tt = MGnCalc(NUM_OF_CHANNELS, ARRIVAL_RATE, b)
    tt.run()
    # get numerical calculation results
    p_num = tt.get_p()
    v_num = tt.get_v()

    tt_time = time.process_time() - tt_start
    # also can find out how many iterations were required
    num_of_iter = tt.num_of_iter_

    # run simulation for verification of the results
    im_start = time.process_time()

    qs = QsSim(NUM_OF_CHANNELS)

    # set arrival process. M - exponential with rate l
    qs.set_sources(ARRIVAL_RATE, 'M')

    # set server parameters as Gamma distribution.
    # Distribution parameters are selected using the method from the random_distribution library
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    qs.set_servers(gamma_params, 'Gamma')

    # Run simulation
    qs.run(NUM_OF_JOBS)

    # Get results
    p_sim = qs.get_p()
    v_sim = qs.v
    im_time = time.process_time() - im_start

    # print results

    print("\nComparison of calculation results by the Takahasi-Takami method and simulation.")
    print(
        f"Simulation - M/Gamma/{NUM_OF_CHANNELS:^2d}")
    print(
        f"Takahasi-Takami - M/H2/{NUM_OF_CHANNELS:^2d} with complex parameters")
    print(f"Utilization factor: {UTILIZATION_FACTOR:^1.2f}")
    print(f"Coefficient of variation of service time: {SERVICE_TIME_CV:^1.2f}")
    print(
        f"Number of iterations of the Takahasi-Takami algorithm: {num_of_iter:^4d}")
    print(f"Takahasi-Takami algorithm execution time: {tt_time:^5.3f} s")
    print(f"Simulation execution time: {im_time:^5.3f} s")
    probs_print(p_sim, p_num, 10)

    times_print(v_sim, v_num, False)

    assert np.allclose(
        v_sim, v_num, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(p_sim[:10], p_num[:10],
                       atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mgn_tt()
