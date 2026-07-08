"""
Test for the infinite-server queue M/G/inf.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_g_inf import MGInfCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

SERVICE_MEAN = 2.0
# "infinite" servers for simulation: far more than the offered load a = l * b1
SIM_CHANNELS = 100


def test_m_g_inf():
    """
    M/G/inf: Poisson state probabilities (insensitivity), zero waiting,
    sojourn time = service time. Validated against QsSim with a large
    number of channels and Gamma service with CV != 1.
    """
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV)
    b = GammaDistribution.calc_theory_moments(gamma_params, 4)

    calc = MGInfCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b=b)
    calc_results = calc.run()

    offered_load = calc.get_offered_load()
    assert np.isclose(offered_load, ARRIVAL_RATE * SERVICE_MEAN), ERROR_MSG

    qs = QsSim(SIM_CHANNELS)
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(gamma_params, "Gamma")
    sim_results = qs.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG

    # no waiting: every job starts service immediately
    assert np.allclose(calc_results.w, 0.0), ERROR_MSG
    assert sim_results.w[0] < MOMENTS_ATOL / 100, ERROR_MSG

    # sojourn time equals service time
    print_sojourn_moments(sim_results.v, calc_results.v)
    assert np.allclose(sim_results.v[:3], calc_results.v[:3], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_m_g_inf()
