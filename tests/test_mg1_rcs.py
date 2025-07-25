"""
Test the M/H2/1 and M/Gamma/1 queueing systems with RCS discipline.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_raw_moments
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_rcs import MG1NegativeCalcRCS

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


def test_mg1_gamma_rcs():
    """
    Test the M/Gamma/1 queueing systems with RCS discipline.
    """

    b1 = UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE
    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # Run simulation
    queue_sim = QsSimNegatives(1, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")

    sim_results = queue_sim.run(NUM_OF_JOBS)

    m_gamma_1_calc = MG1NegativeCalcRCS()
    m_gamma_1_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    m_gamma_1_calc.set_servers(b=b)
    calc_results = m_gamma_1_calc.run()

    print(f"Utilization calc: {calc_results.utilization: 0.4f}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    print_raw_moments(
        sim_results.v[0],
        calc_results.v[0],
        header="Sojourn time in M/G/1 with RCS disasters",
    )

    # assert is all close with rtol 10%
    assert np.allclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_mg1_gamma_rcs()
