"""
Test M/G/1 queue with warm-up phase.
Compare theoretical and simulated moments.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.mg1_warm_calc import MG1WarmCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = 1

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_TIME_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

WARM_UP_CV = float(params["warm-up"]["cv"])

MEAN_WARMUP_FACTOR = 1.5  # Mean time for warm-up phase is factor*mean service time


def test_mg1_warm():
    """
    Test M/G/1 queue with warm-up  phase.
    Compare theoretical and simulated moments.
    """

    b1 = UTILIZATION_FACTOR / ARRIVAL_RATE
    b_s = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)
    service_params = GammaDistribution.get_params(b_s)

    # Warm phase parameters
    mean_warmup_time = b1 * MEAN_WARMUP_FACTOR
    b_w = gamma_moments_by_mean_and_cv(mean_warmup_time, WARM_UP_CV)
    warmup_params = GammaDistribution.get_params(b_w)

    # Initialize the Vacation Queueing System Simulator
    simulator = VacationQueueingSystemSimulator(1, is_service_on_warm_up=True)

    # Set warm-up phase parameters
    simulator.set_servers(service_params, "Gamma")
    simulator.set_warm(warmup_params, "Gamma")

    # Configure the simulator
    simulator.set_sources(ARRIVAL_RATE, "M")

    # Run simulations
    sim_results = simulator.run(NUM_OF_JOBS)

    mg1_calc = MG1WarmCalc()
    mg1_calc.set_sources(ARRIVAL_RATE)
    mg1_calc.set_servers(b=b_s, b_warm=b_w)
    calc_results = mg1_calc.run()

    print(f"utilization: {calc_results.utilization: 0.4f}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print_sojourn_moments(simulator.v, calc_results.v)

    # assert all close with relative percent 20%
    assert np.allclose(simulator.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_mg1_warm()
