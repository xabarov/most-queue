"""
Tests for Erlang B (M/M/n/0, M/G/n/0 loss) and Erlang C (M/M/n) calculators.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.erlang import ErlangBCalc, ErlangCCalc
from most_queue.theory.fifo.mmnr import MMnrCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_CV = float(params["service"]["cv"])

NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def test_erlang_b_vs_sim():
    """
    Erlang B (M/M/n/0): state probabilities and blocking vs simulation with zero buffer.
    """
    service_rate = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)

    calc = ErlangBCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(mu=service_rate)
    calc_results = calc.run()

    qs = QsSim(NUM_OF_CHANNELS, buffer=0)
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(service_rate, "M")
    sim_results = qs.run(NUM_OF_JOBS)

    sim_blocking = qs.dropped / qs.arrived
    calc_blocking = calc.get_blocking_probability()
    print(f"blocking: sim={sim_blocking:.4f}, calc={calc_blocking:.4f}")

    probs_print(sim_results.p, calc_results.p, size=NUM_OF_CHANNELS + 1)

    assert np.allclose(
        sim_results.p[: NUM_OF_CHANNELS + 1], calc_results.p, atol=PROBS_ATOL, rtol=PROBS_RTOL
    ), ERROR_MSG
    assert np.isclose(sim_blocking, calc_blocking, atol=PROBS_ATOL), ERROR_MSG

    print_sojourn_moments(sim_results.v, calc_results.v)
    assert np.allclose(sim_results.v[:3], calc_results.v[:3], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_erlang_b_insensitivity():
    """
    Sevastyanov's theorem: blocking in M/G/n/0 depends on the service distribution
    only through its mean. Simulate with Gamma service (CV != 1) and compare with
    the same Erlang B calculation.
    """
    service_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE

    calc = ErlangBCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(mu=1.0 / service_mean)
    calc_results = calc.run()

    gamma_srv = GammaDistribution.get_params_by_mean_and_cv(service_mean, SERVICE_CV)
    qs = QsSim(NUM_OF_CHANNELS, buffer=0)
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(gamma_srv, "Gamma")
    sim_results = qs.run(NUM_OF_JOBS)

    sim_blocking = qs.dropped / qs.arrived
    calc_blocking = calc.get_blocking_probability()
    print(f"blocking (Gamma service, cv={SERVICE_CV}): sim={sim_blocking:.4f}, calc={calc_blocking:.4f}")

    probs_print(sim_results.p, calc_results.p, size=NUM_OF_CHANNELS + 1)

    assert np.isclose(sim_blocking, calc_blocking, atol=PROBS_ATOL), ERROR_MSG
    assert np.allclose(
        sim_results.p[: NUM_OF_CHANNELS + 1], calc_results.p, atol=PROBS_ATOL, rtol=PROBS_RTOL
    ), ERROR_MSG


def test_erlang_c_vs_sim_and_mmnr():
    """
    Erlang C (M/M/n): waiting/sojourn moments and state probabilities vs simulation
    with infinite buffer; state probabilities cross-checked against MMnrCalc with
    a large finite queue.
    """
    service_rate = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)

    calc = ErlangCCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(mu=service_rate)
    calc_results = calc.run()

    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(service_rate, "M")
    sim_results = qs.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG

    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)
    assert np.allclose(sim_results.w[:3], calc_results.w[:3], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.v[:3], calc_results.v[:3], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    # cross-check state probabilities against M/M/n/r with a large queue
    mmnr = MMnrCalc(n=NUM_OF_CHANNELS, r=200)
    mmnr.set_sources(l=ARRIVAL_RATE)
    mmnr.set_servers(mu=service_rate)
    mmnr_results = mmnr.run()
    assert np.allclose(mmnr_results.p[:10], calc_results.p[:10], atol=1e-6), ERROR_MSG


if __name__ == "__main__":
    test_erlang_b_vs_sim()
    test_erlang_b_insensitivity()
    test_erlang_c_vs_sim_and_mmnr()
