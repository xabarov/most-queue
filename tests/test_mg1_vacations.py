"""
Tests for classic M/G/1 vacation models: multiple vacations and N-policy
(Fuhrmann-Cooper decomposition).
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.vacations import NPolicyQueueSim, VacationQueueingSystemSimulator
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.vacations.mg1_vacations import MG1MultipleVacationsCalc, MG1NPolicyCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

VACATION_MEAN = 1.5
VACATION_CV = 1.2
N_POLICY_THRESHOLD = 5


def _gamma_moments(mean: float, cv: float, num: int = 5) -> list[float]:
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean, cv)
    return GammaDistribution.calc_theory_moments(gamma_params, num)


def test_mg1_multiple_vacations_vs_sim():
    """
    M/G/1 with multiple vacations vs simulation (Gamma service, Gamma vacations).
    """
    b = _gamma_moments(UTILIZATION_FACTOR / ARRIVAL_RATE, SERVICE_CV)
    vacation = _gamma_moments(VACATION_MEAN, VACATION_CV, 4)

    calc = MG1MultipleVacationsCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc.set_vacations(vacation)
    calc_results = calc.run()

    sim = VacationQueueingSystemSimulator(1, is_multiple_vacations=True)
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(GammaDistribution.get_params_by_mean_and_cv(UTILIZATION_FACTOR / ARRIVAL_RATE, SERVICE_CV), "Gamma")
    sim.set_cold(GammaDistribution.get_params_by_mean_and_cv(VACATION_MEAN, VACATION_CV), "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)

    assert np.allclose(sim_results.w[:3], calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.v[:3], calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_mg1_multiple_vacations_zero_limit():
    """
    Degenerate case: vanishing vacations reduce the model to the plain M/G/1.
    """
    b = _gamma_moments(UTILIZATION_FACTOR / ARRIVAL_RATE, SERVICE_CV)
    d = 1e-6  # deterministic vacation of vanishing length
    vacation = [d, d**2, d**3, d**4]

    calc = MG1MultipleVacationsCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc.set_vacations(vacation)
    calc_results = calc.run()

    mg1 = MG1Calc()
    mg1.set_sources(l=ARRIVAL_RATE)
    mg1.set_servers(b)
    mg1_results = mg1.run()

    assert np.allclose(calc_results.w, mg1_results.w[:3], rtol=1e-5), ERROR_MSG


def test_mg1_n_policy_vs_sim():
    """
    M/G/1 under N-policy vs simulation (server sleeps until N jobs accumulate).
    """
    b = _gamma_moments(UTILIZATION_FACTOR / ARRIVAL_RATE, SERVICE_CV)

    calc = MG1NPolicyCalc(big_n=N_POLICY_THRESHOLD)
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc_results = calc.run()

    sim = NPolicyQueueSim(1, big_n=N_POLICY_THRESHOLD)
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(GammaDistribution.get_params_by_mean_and_cv(UTILIZATION_FACTOR / ARRIVAL_RATE, SERVICE_CV), "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)

    assert np.allclose(sim_results.w[:3], calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.v[:3], calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_mg1_n_policy_properties():
    """
    N = 1 reduces exactly to the plain M/G/1; the additional delay moments for
    N = 3 match the hand-computed Erlang mixture values: E[D] = 1/l,
    E[D^2] = 8/(3*l^2); the mean extra wait is (N-1)/(2*l) for any N.
    """
    b = _gamma_moments(UTILIZATION_FACTOR / ARRIVAL_RATE, SERVICE_CV)

    mg1 = MG1Calc()
    mg1.set_sources(l=ARRIVAL_RATE)
    mg1.set_servers(b)
    mg1_results = mg1.run()

    calc1 = MG1NPolicyCalc(big_n=1)
    calc1.set_sources(l=ARRIVAL_RATE)
    calc1.set_servers(b)
    results1 = calc1.run()
    assert np.allclose(results1.w, mg1_results.w[:3], rtol=1e-12), ERROR_MSG

    calc3 = MG1NPolicyCalc(big_n=3)
    calc3.set_sources(l=ARRIVAL_RATE)
    calc3.set_servers(b)
    d_moments = calc3._additional_delay_moments(2)  # pylint: disable=protected-access
    assert np.isclose(d_moments[0], 1.0 / ARRIVAL_RATE), ERROR_MSG
    assert np.isclose(d_moments[1], 8.0 / (3.0 * ARRIVAL_RATE**2)), ERROR_MSG

    for big_n in (2, 5, 10):
        calc = MG1NPolicyCalc(big_n=big_n)
        calc.set_sources(l=ARRIVAL_RATE)
        calc.set_servers(b)
        results = calc.run()
        extra = results.w[0] - mg1_results.w[0]
        assert np.isclose(extra, (big_n - 1) / (2.0 * ARRIVAL_RATE), rtol=1e-10), ERROR_MSG


if __name__ == "__main__":
    test_mg1_multiple_vacations_vs_sim()
    test_mg1_multiple_vacations_zero_limit()
    test_mg1_n_policy_vs_sim()
    test_mg1_n_policy_properties()
