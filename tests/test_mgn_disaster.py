"""
Test QS M/G/n queue with disasters.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import probs_print, times_print
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params["num_of_channels"])

SERVICE_TIME_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

ARRIVAL_RATE_POSITIVE = float(params["arrival"]["rate"])
ARRIVAL_RATE_NEGATIVE = 0.3 * ARRIVAL_RATE_POSITIVE


def test_mgn():
    """
    Test QS M/G/n queue with disasters.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time

    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # Run simulation
    queue_sim = QsSimNegatives(NUM_OF_CHANNELS, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")

    sim_results = queue_sim.run(NUM_OF_JOBS)

    # Run calc
    queue_calc = MGnNegativeDisasterCalc(n=NUM_OF_CHANNELS)
    queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    queue_calc.set_servers(b=b)

    calc_results = queue_calc.run()

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    probs_print(sim_results.p, calc_results.p)
    times_print(sim_results.v, calc_results.v, is_w=False, header="sojourn total")
    times_print(sim_results.v_served, calc_results.v_served, is_w=False, header="sojourn served")
    times_print(sim_results.v_broken, calc_results.v_broken, is_w=False, header="sojourn broken")
    times_print(sim_results.w, calc_results.w)

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mgn()
