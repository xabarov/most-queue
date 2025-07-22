"""
Testing the Takahasi-Takami method for calculating an M/H2/n queue

When the coefficient of variation of service time is less than 1,
the parameters of the approximating H2 distribution
are complex, which does not prevent obtaining meaningful results.

For verification, simulation is used.

"""

import os

import numpy as np
import yaml

from most_queue.io.tables import probs_print, times_print
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mgn_takahasi import MGnCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
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


def test_mgn_tt():
    """
    Testing the Takahasi-Takami method for calculating an M/H2/n queue
    """

    # calculate initial moments of service time based
    # on the given average and coefficient of variation

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE  # average service time
    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # run Takahasi-Takami method
    tt = MGnCalc(n=NUM_OF_CHANNELS)
    tt.set_sources(l=ARRIVAL_RATE)
    tt.set_servers(b=b)

    # get numerical calculation results
    calc_results = tt.run()

    # also can find out how many iterations were required
    num_of_iter = tt.num_of_iter_

    qs = QsSim(NUM_OF_CHANNELS)

    # set arrival process. M - exponential with rate l
    qs.set_sources(ARRIVAL_RATE, "M")

    # set server parameters as Gamma distribution.
    # Distribution parameters are selected using the method from the
    # random_distribution library
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    qs.set_servers(gamma_params, "Gamma")

    # Run simulation
    sim_results = qs.run(NUM_OF_JOBS)

    # print results

    print("\nComparison of calculation results by the Takahasi-Takami method and simulation.")
    print(f"Simulation - M/Gamma/{NUM_OF_CHANNELS:^2d}")
    print(f"Takahasi-Takami - M/H2/{NUM_OF_CHANNELS:^2d} with complex parameters")
    print(f"Utilization factor: {UTILIZATION_FACTOR:^1.2f}")
    print(f"Coefficient of variation of service time: {SERVICE_TIME_CV:^1.2f}")
    print(f"Number of iterations of the Takahasi-Takami algorithm: {num_of_iter:^4d}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    probs_print(sim_results.p, calc_results.p, 10)

    times_print(sim_results.v, calc_results.v, False)
    times_print(sim_results.w, calc_results.w, True)

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mgn_tt()
