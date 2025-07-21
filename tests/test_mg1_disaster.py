"""
Test QS M/G/1 queue with disasters.
Under development
"""

import os

import numpy as np
import yaml

from most_queue.general.tables import times_print_with_two_numerical
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_disasters import MG1Disasters
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

ARRIVAL_RATE_POSITIVE = float(params["arrival"]["rate"])
ARRIVAL_RATE_NEGATIVE = 0.8 * ARRIVAL_RATE_POSITIVE


def test_mg1():
    """
    Test QS M/G/1 queue with disasters.
    Compare theoretical and simulated moments.
    MG1 calculate moments using Pollaczek-Khintchine formula.
        Jain, Gautam, and Karl Sigman. "A Pollaczek–Khintchine formula
        for M/G/1 queues with disasters."
        Journal of Applied Probability 33.4 (1996): 1191-1200.
    T-T is ours method (based on Takahashi-Takami)
    """

    b1 = 1 * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time

    approximation = "gamma"

    b_params = GammaDistribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CV)
    b = GammaDistribution.calc_theory_moments(b_params)

    # Run calc
    mg1_queue_calc = MG1Disasters()
    mg1_queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    mg1_queue_calc.set_servers(b=b)
    mg1_calc_result = mg1_queue_calc.run()

    mgn_queue_calc = MGnNegativeDisasterCalc(n=1)
    mgn_queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    mgn_queue_calc.set_servers(b=b)

    tt_calc_result = mgn_queue_calc.run()

    # Run simulation
    queue_sim = QsSimNegatives(1, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")

    if approximation == "h2":
        queue_sim.set_servers(b_params, "H")
    else:
        queue_sim.set_servers(b_params, "Gamma")

    queue_sim.run(NUM_OF_JOBS)

    v_sim = queue_sim.get_v()

    times_print_with_two_numerical(
        v_sim, mg1_calc_result.v, tt_calc_result.v, is_w=False, num1_header="MG1", num2_header="T-T"
    )

    assert np.allclose(v_sim, tt_calc_result.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    # when MG1 will work, add assert with v_calc1


if __name__ == "__main__":
    test_mg1()
