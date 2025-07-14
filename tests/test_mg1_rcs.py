"""
Test the M/H2/1 and M/Gamma/1 queueing systems with RCS discipline.
"""

import os

import numpy as np
import yaml

from most_queue.general.distribution_fitting import \
    gamma_moments_by_mean_and_coev
from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
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
    Test the  M/Gamma/1 queueing systems with RCS discipline.
    """

    b1 = UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE
    b = gamma_moments_by_mean_and_coev(b1, SERVICE_TIME_CV)

    # Run simulation
    queue_sim = QsSimNegatives(1, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")

    queue_sim.run(NUM_OF_JOBS)

    v_sim = queue_sim.get_v()

    m_gamma_1_calc = MG1NegativeCalcRCS(
        ARRIVAL_RATE_POSITIVE,
        ARRIVAL_RATE_NEGATIVE,
        b,
        service_time_approx_dist="gamma",
    )
    v1_gamma_calc = m_gamma_1_calc.get_v1()

    times_print(
        v_sim[0],
        v1_gamma_calc,
        is_w=False,
        header="Sojourn time in M/G/1 with RCS disasters",
    )

    # assert is all close with rtol 10%
    assert np.allclose(v_sim[0], v1_gamma_calc, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_mg1_gamma_rcs()
