"""
Tests for M/G/1 with an unreliable server (Avi-Itzhak-Naor):
completion-time reduction vs simulation with breakdowns.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.unreliable import UnreliableQueueSim
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.vacations.mg1_unreliable import MG1UnreliableCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

SERVICE_MEAN = 0.5
FAILURE_RATE = 0.3
REPAIR_MEAN = 0.4
REPAIR_CV = 1.2


def _gamma_moments(mean: float, cv: float, num: int = 5) -> list[float]:
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean, cv)
    return GammaDistribution.calc_theory_moments(gamma_params, num)


def test_mg1_unreliable_vs_sim():
    """
    Unreliable M/G/1 vs simulation with Poisson breakdowns and Gamma repairs.
    """
    b = _gamma_moments(SERVICE_MEAN, SERVICE_CV)
    r = _gamma_moments(REPAIR_MEAN, REPAIR_CV)

    calc = MG1UnreliableCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc.set_breakdowns(FAILURE_RATE, r)
    calc_results = calc.run()

    sim = UnreliableQueueSim()
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(GammaDistribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV), "Gamma")
    sim.set_breakdowns(FAILURE_RATE, GammaDistribution.get_params_by_mean_and_cv(REPAIR_MEAN, REPAIR_CV), "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)

    assert np.allclose(sim_results.w[:3], calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.v[:3], calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_mg1_unreliable_no_failures_equals_mg1():
    """
    Degenerate case: zero failure rate reduces the model exactly to M/G/1.
    """
    b = _gamma_moments(SERVICE_MEAN, SERVICE_CV)
    r = _gamma_moments(REPAIR_MEAN, REPAIR_CV)

    calc = MG1UnreliableCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc.set_breakdowns(0.0, r)
    calc_results = calc.run()

    mg1 = MG1Calc()
    mg1.set_sources(l=ARRIVAL_RATE)
    mg1.set_servers(b)
    mg1_results = mg1.run()

    assert np.allclose(calc_results.w, mg1_results.w[:3], rtol=1e-12), ERROR_MSG
    assert np.allclose(calc_results.v, mg1_results.v[:3], rtol=1e-12), ERROR_MSG


def test_completion_moments_mean_identity():
    """
    E[C] = b1 * (1 + xi * r1) — the mean completion time identity.
    """
    b = _gamma_moments(SERVICE_MEAN, SERVICE_CV)
    r = _gamma_moments(REPAIR_MEAN, REPAIR_CV)

    calc = MG1UnreliableCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc.set_breakdowns(FAILURE_RATE, r)
    c = calc.get_completion_moments(3)

    assert np.isclose(c[0], SERVICE_MEAN * (1 + FAILURE_RATE * REPAIR_MEAN), rtol=1e-12), ERROR_MSG


if __name__ == "__main__":
    test_mg1_unreliable_vs_sim()
    test_mg1_unreliable_no_failures_equals_mg1()
    test_completion_moments_mean_identity()
