"""
Test function to compare results of the simulation and Takahashi-Takami (TT) algorithm
for an MMn queueing system with H2 cold and warm-up phases.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_waiting_moments, probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.mmn_with_h2_cold_and_h2_warmup import MMnHyperExpWarmAndCold

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params["num_of_channels"])

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_TIME_CV = float(params["service"]["cv"])

WARM_UP_CV = float(params["warm-up"]["cv"])
COOLING_CV = float(params["cooling"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

WARMUP_TIME_PROPORTION = 0.3  # percentage of mean service time for warm-up phase
COLD_TIME_PROPORTION = 0.2  # percentage of mean service time for cold phase


def test_mmn_h2cold_h2_warm():
    """
    Test function to compare results of the Implicit Method (IM) and Takahashi-Takami (TT) algorithm
    for an MMn queueing system with H2 cold and warm-up phases.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE
    mean_warmup_time = b1 * WARMUP_TIME_PROPORTION
    mean_cold_time = b1 * COLD_TIME_PROPORTION

    # Initialize the Vacation Queueing System Simulator
    simulator = VacationQueueingSystemSimulator(NUM_OF_CHANNELS, buffer=None)

    service_rate = 1.0 / b1
    simulator.set_servers(service_rate, "M")

    # Set warm-up phase parameters
    b_w = gamma_moments_by_mean_and_cv(mean_warmup_time, WARM_UP_CV)
    warmup_params = GammaDistribution.get_params(b_w)
    simulator.set_warm(warmup_params, "Gamma")

    # Set cold phase parameters
    b_c = gamma_moments_by_mean_and_cv(mean_cold_time, COOLING_CV)
    cold_params = GammaDistribution.get_params(b_c)
    simulator.set_cold(cold_params, "Gamma")

    # Configure the simulator
    simulator.set_sources(ARRIVAL_RATE, "M")

    sim_results = simulator.run(NUM_OF_JOBS)

    tt = MMnHyperExpWarmAndCold(n=NUM_OF_CHANNELS)
    tt.set_sources(ARRIVAL_RATE)
    tt.set_servers(mu=service_rate, b_warm=b_w, b_cold=b_c)
    calc_results = tt.run()

    num_of_iter = tt.num_of_iter_

    print(f"utilization {calc_results.utilization:0.4f}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print("warms starts", simulator.warm_phase.starts_times)
    print("warms after cold starts", simulator.warm_after_cold_starts)
    print("cold starts", simulator.cold_phase.starts_times)
    print("zero wait arrivals num", simulator.zero_wait_arrivals_num)

    print(
        f"\nComparison of results calculated using the Takahashi-Takami method and simulation.\n"
        f"Sim - M/Gamma/{{{NUM_OF_CHANNELS:2d}}} with Gamma warming\n"
        f"Takahashi-Takami - M/M/{{{NUM_OF_CHANNELS:2d}}} with H2-warming and H2-cooling "
        f"with complex parameters\n"
        f"Utilization coefficient: {UTILIZATION_FACTOR:.2f}"
    )
    print(f"Variation coefficient of warming time {WARM_UP_CV:.3f}")
    print(f"Variation coefficient of cooling time {COOLING_CV:.3f}")
    print(f"Number of iterations in the Takahashi-Takami algorithm: {num_of_iter:4d}")
    print(
        f"Probability of being in the warming state\n"
        f"\tSim: {simulator.get_warmup_prob():.3f}\n"
        f"\tCalc: {calc_results.warmup_prob:.3f}"
    )
    print(
        f"Probability of being in the cooling state\n"
        f"\tSim: {simulator.get_cold_prob():.3f}\n"
        f"\tCalc: {calc_results.cold_prob:.3f}"
    )

    probs_print(p_sim=sim_results.p, p_num=calc_results.p, size=10)

    print_waiting_moments(sim_moments=sim_results.w, calc_moments=calc_results.w)

    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_mmn_h2cold_h2_warm()
