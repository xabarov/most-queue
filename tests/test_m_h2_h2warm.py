"""
Test M/H2/n system with H2-warming using the Takahasi-Takagi method.
"""
import math
import time

import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.vacations import VacationQueueingSystemSimulator
from most_queue.theory.queueing_systems.vacations.m_h2_h2warm import MH2nH2Warm


def test_m_h2_h2warm():
    """
    Test M/H2/n system with H2-warming using the Takahasi-Takagi method.
    """
    n = 5  # number of channels
    l = 1.0  # intensity of the arrivals
    ro = 0.7  # load factor
    b1 = n * 0.7  # average service time
    b1_warm = n * 0.1  # average warming time
    num_of_jobs = 100000  # number of jobs for the simulation
    b_coevs = [1.5]  # coefficient of variation of service time
    b_coev_warm = 1.2  # coefficient of variation of warming time
    buff = None  # buffer - unlimited
    verbose = False  # do not output explanations during calculations

    for b_coev in b_coevs:
        b = [0.0] * 3
        alpha = 1 / (b_coev ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        b_w = [0.0] * 3
        b_w[0] = b1_warm
        alpha = 1 / (b_coev_warm ** 2)
        b_w[1] = math.pow(b_w[0], 2) * (math.pow(b_coev_warm, 2) + 1)
        b_w[2] = b_w[1] * b_w[0] * (1.0 + 2 / alpha)

        im_start = time.process_time()
        qs = VacationQueueingSystemSimulator(n, buffer=buff)
        qs.set_sources(l, 'M')

        gamma_params = GammaDistribution.get_params(b)
        gamma_params_warm = GammaDistribution.get_params(b_w)
        qs.set_servers(gamma_params, 'Gamma')
        qs.set_warm(gamma_params_warm, 'Gamma')
        qs.run(num_of_jobs)
        p = qs.get_p()
        v_sim = qs.v
        im_time = time.process_time() - im_start

        tt_start = time.process_time()
        tt = MH2nH2Warm(l, b, b_w, n, buffer=buff, verbose=verbose)

        tt.run()
        p_tt = tt.get_p()
        v_tt = tt.get_v()
        tt_time = time.process_time() - tt_start

        num_of_iter = tt.num_of_iter_

        print("\nComparison of results calculated by the Takacs-Takaichi method and Simulation.")
        print(
            f"Simulation - M/Gamma/{n:^2d}\nTakacs-Takaichi - M/H2/{n:^2d} with complex parameters")
        print(f"Load factor: {ro:^1.2f}")
        print(f'Coefficient of variation of service time {b_coev:0.3f}')
        print(f'Coefficient of variation of warming time {b_coev_warm:0.3f}')
        print(
            f"Number of iterations of the Takacs-Takaichi algorithm: {num_of_iter:^4d}")
        print(
            f"Time taken by the Takacs-Takaichi algorithm: {tt_time:^5.3f} s")
        print(f"Simulation time: {im_time:^5.3f} s")

        probs_print(p, p_tt, 10)
        times_print(v_sim, v_tt, False)

        assert np.allclose(np.array(v_sim), np.array(v_tt), rtol=0.1)


if __name__ == "__main__":
    test_m_h2_h2warm()
