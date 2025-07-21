"""
Test M/H2/n system with H2-warming using the Takahashi-Takami method.
"""

import os
import time

import numpy as np
import yaml

from most_queue.general.distribution_fitting import gamma_moments_by_mean_and_coev
from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.m_h2_h2warm import MH2nH2Warm

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params["num_of_channels"])

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_TIME_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

WARM_UP_CV = float(params["warm-up"]["cv"])

WARM_AP_AVE_PERCENT = 0.3


def test_m_h2_h2warm():
    """
    Test M/H2/n system with H2-warming using the Takahashi-Takami method.
    """
    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE
    b = gamma_moments_by_mean_and_coev(b1, SERVICE_TIME_CV)

    b_w1 = b[0] * WARM_AP_AVE_PERCENT
    b_w = gamma_moments_by_mean_and_coev(b_w1, WARM_UP_CV)

    im_start = time.process_time()
    qs = VacationQueueingSystemSimulator(NUM_OF_CHANNELS)
    qs.set_sources(ARRIVAL_RATE, "M")

    gamma_params = GammaDistribution.get_params(b)
    gamma_params_warm = GammaDistribution.get_params(b_w)
    qs.set_servers(gamma_params, "Gamma")
    qs.set_warm(gamma_params_warm, "Gamma")
    qs.run(NUM_OF_JOBS)
    p_sim = qs.get_p()
    v_sim = qs.v
    w_sim = qs.w
    im_time = time.process_time() - im_start

    tt_start = time.process_time()
    tt = MH2nH2Warm(n=NUM_OF_CHANNELS)
    tt.set_sources(ARRIVAL_RATE)
    tt.set_servers(b=b, b_warm=b_w)

    tt_results = tt.run()
    tt_time = time.process_time() - tt_start

    print(f"utulization: {tt_results.utilization: 0.4f}")

    num_of_iter = tt.num_of_iter_

    print("\nComparison of results calculated by the Takacs-Takaichi method and Simulation.")
    print(f"Simulation - M/Gamma/{NUM_OF_CHANNELS:^2d}")
    print(f" Takasi-Takaichi - M/H2/{NUM_OF_CHANNELS:^2d} with complex parameters")
    print(f"Load factor: {UTILIZATION_FACTOR:^1.2f}")
    print(f"Coefficient of variation of service time {SERVICE_TIME_CV:0.3f}")
    print(f"Coefficient of variation of warming time {WARM_UP_CV:0.3f}")
    print(f"Number of iterations of the Takacs-Takaichi algorithm: {num_of_iter:^4d}")
    print(f"Time taken by the Takacs-Takaichi algorithm: {tt_time:^5.3f} s")
    print(f"Simulation time: {im_time:^5.3f} s")

    probs_print(p_sim, tt_results.p, 10)
    times_print(v_sim, tt_results.v, False)
    times_print(w_sim, tt_results.w, True)

    assert np.allclose(v_sim, tt_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(p_sim[:10], tt_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_m_h2_h2warm()
