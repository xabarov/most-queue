"""
Tests for EPIC-004 models: Erlang-A (M/M/n+M) and retrial queues
(M/M/1 and M/G/1 with the classical linear retrial policy).
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.impatient import ImpatientQueueSim
from most_queue.sim.retrial import RetrialQueueSim
from most_queue.theory.fifo.erlang import ErlangCCalc
from most_queue.theory.impatience.mm1 import MM1Impatience
from most_queue.theory.impatience.mmn import MMnImpatienceCalc
from most_queue.theory.retrial import MG1RetrialCalc, MM1RetrialCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
NUM_OF_CHANNELS = int(params["num_of_channels"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

IMPATIENCE_RATE = 0.3
RETRIAL_RATE = 0.7


# --------------------------------------------------------------- Erlang-A
def test_erlang_a_reduces_to_mm1_impatience():
    """n=1 reduces exactly to the existing MM1Impatience calculator."""
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR

    calc = MMnImpatienceCalc(n=1, theta=IMPATIENCE_RATE)
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(mu)
    results = calc.run()

    exact = MM1Impatience(gamma=IMPATIENCE_RATE)
    exact.set_sources(ARRIVAL_RATE)
    exact.set_servers(mu)
    exact_results = exact.run()

    k = min(len(results.p), len(exact_results.p))
    assert np.allclose(results.p[:k], exact_results.p[:k], atol=1e-10), ERROR_MSG
    assert np.isclose(results.w[0], exact_results.w[0], rtol=1e-10), ERROR_MSG
    assert np.isclose(results.v[0], exact_results.v[0], rtol=1e-10), ERROR_MSG


def test_erlang_a_vs_sim():
    """Erlang-A (n=3) against the impatience simulator."""
    service_rate = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)

    calc = MMnImpatienceCalc(n=NUM_OF_CHANNELS, theta=IMPATIENCE_RATE)
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(service_rate)
    calc_results = calc.run()

    sim = ImpatientQueueSim(NUM_OF_CHANNELS)
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(service_rate, "M")
    sim.set_impatience(IMPATIENCE_RATE, "M")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_erlang_a_theta_to_zero_is_erlang_c():
    """Vanishing impatience recovers the Erlang C mean wait and states."""
    service_rate = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)

    calc = MMnImpatienceCalc(n=NUM_OF_CHANNELS, theta=1e-8)
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(service_rate)

    exact = ErlangCCalc(n=NUM_OF_CHANNELS)
    exact.set_sources(l=ARRIVAL_RATE)
    exact.set_servers(mu=service_rate)
    exact_results = exact.run()

    assert np.isclose(calc.get_w1(), exact_results.w[0], rtol=1e-4), ERROR_MSG
    assert np.allclose(calc.get_p()[:10], exact_results.p[:10], atol=1e-5), ERROR_MSG


def test_erlang_a_staffing():
    """Staffing helper returns the minimal n meeting the abandonment target."""
    mu = 1.0
    calc = MMnImpatienceCalc(n=1, theta=IMPATIENCE_RATE)
    calc.set_sources(5.0)
    calc.set_servers(mu)

    n_min = calc.find_min_servers(target_abandonment=0.05)
    check = MMnImpatienceCalc(n=n_min, theta=IMPATIENCE_RATE)
    check.set_sources(5.0)
    check.set_servers(mu)
    assert check.get_abandonment_probability() <= 0.05, ERROR_MSG
    if n_min > 1:
        below = MMnImpatienceCalc(n=n_min - 1, theta=IMPATIENCE_RATE)
        below.set_sources(5.0)
        below.set_servers(mu)
        assert below.get_abandonment_probability() > 0.05, ERROR_MSG


# ---------------------------------------------------------------- retrial
def test_mm1_retrial_vs_sim():
    """Exact truncated-chain M/M/1 retrial against the orbit simulator."""
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR

    calc = MM1RetrialCalc(gamma=RETRIAL_RATE)
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(mu)
    calc_results = calc.run()

    sim = RetrialQueueSim(gamma=RETRIAL_RATE)
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(mu, "M")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    print(f"retrial M/M/1: w1 sim={sim_results.w[0]:.4f}, calc={calc_results.w[0]:.4f}")
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.isclose(sim_results.w[0], calc_results.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_mg1_retrial_formula_matches_exact_mm1():
    """The Falin-Templeton mean-orbit formula equals the exact M/M/1 solution."""
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR
    for gamma in (0.3, 1.0, 5.0):
        exact = MM1RetrialCalc(gamma=gamma)
        exact.set_sources(ARRIVAL_RATE)
        exact.set_servers(mu)

        formula = MG1RetrialCalc(gamma=gamma)
        formula.set_sources(ARRIVAL_RATE)
        formula.set_servers([1.0 / mu, 2.0 / mu**2])

        assert np.isclose(formula.get_orbit_mean(), exact.get_orbit_mean(), rtol=1e-8), ERROR_MSG


def test_mg1_retrial_vs_sim_gamma_service():
    """M/G/1 retrial with Gamma service against the orbit simulator."""
    service_mean = UTILIZATION_FACTOR / ARRIVAL_RATE
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, 1.2)
    b = GammaDistribution.calc_theory_moments(gamma_params, 3)

    calc = MG1RetrialCalc(gamma=RETRIAL_RATE)
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(b)
    calc_results = calc.run()

    sim = RetrialQueueSim(gamma=RETRIAL_RATE)
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(gamma_params, "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    print(f"retrial M/G/1: w1 sim={sim_results.w[0]:.4f}, calc={calc_results.w[0]:.4f}")
    assert np.isclose(sim_results.w[0], calc_results.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_retrial_fast_gamma_recovers_mg1():
    """gamma -> infinity: the retrial queue behaves like the ordinary M/G/1."""
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR
    calc = MM1RetrialCalc(gamma=1e6)
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(mu)

    exact = ErlangCCalc(n=1)
    exact.set_sources(l=ARRIVAL_RATE)
    exact.set_servers(mu=mu)
    exact_results = exact.run()

    assert np.isclose(calc.get_w1(), exact_results.w[0], rtol=1e-4), ERROR_MSG


if __name__ == "__main__":
    test_erlang_a_reduces_to_mm1_impatience()
    test_erlang_a_vs_sim()
    test_erlang_a_theta_to_zero_is_erlang_c()
    test_erlang_a_staffing()
    test_mm1_retrial_vs_sim()
    test_mg1_retrial_formula_matches_exact_mm1()
    test_mg1_retrial_vs_sim_gamma_service()
    test_retrial_fast_gamma_recovers_mg1()
