"""
Tests for the MAP/PH/1 QBD calculator (and M/PH/1, PH/PH/1 special cases):
cross-validation against exact M/M/1, Pollaczek-Khinchine, GI/M/1 and simulation.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import ErlangDistribution, H2Distribution
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.random.utils.params import ErlangParams, H2Params
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.erlang import ErlangCCalc
from most_queue.theory.fifo.gi_m_1 import GIM1Calc
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.matrix.map_ph1 import MapPh1Calc, MPh1Calc, PhPh1Calc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

H2_SERVICE = H2Params(p1=0.4, mu1=2.0 / UTILIZATION_FACTOR, mu2=0.5 / UTILIZATION_FACTOR)


def test_mph1_reduces_to_mm1():
    """M/PH/1 with a one-phase PH equals the exact M/M/1 (Erlang C, n=1)."""
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR

    calc = MPh1Calc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(PHDistribution.from_exp(mu))
    results = calc.run()

    exact = ErlangCCalc(n=1)
    exact.set_sources(l=ARRIVAL_RATE)
    exact.set_servers(mu=mu)
    exact_results = exact.run()

    assert np.allclose(results.w, exact_results.w[:3], rtol=1e-5), ERROR_MSG
    assert np.allclose(results.p[:15], exact_results.p[:15], atol=1e-10), ERROR_MSG


def test_mph1_matches_pollaczek_khinchine():
    """M/PH/1 with H2 service equals the exact M/G/1 (P-K) waiting moments."""
    calc = MPh1Calc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(PHDistribution.from_h2(H2_SERVICE))
    results = calc.run()

    b = H2Distribution.calc_theory_moments(H2_SERVICE, 4)
    exact = MG1Calc()
    exact.set_sources(l=ARRIVAL_RATE)
    exact.set_servers(b)
    exact_results = exact.run()

    print_waiting_moments(results.w, exact_results.w)
    assert np.allclose(results.w, exact_results.w[:3], rtol=1e-4), ERROR_MSG
    assert np.allclose(results.v, exact_results.v[:3], rtol=1e-4), ERROR_MSG


def test_phph1_matches_gim1():
    """PH(Erlang-2)/M/1 equals the exact GI/M/1 waiting moments."""
    erlang_arr = ErlangParams(r=2, mu=2.0 * ARRIVAL_RATE)
    mu = ARRIVAL_RATE / UTILIZATION_FACTOR

    calc = PhPh1Calc()
    calc.set_sources(PHDistribution.from_erlang(erlang_arr))
    calc.set_servers(PHDistribution.from_exp(mu))
    results = calc.run()

    a = ErlangDistribution.calc_theory_moments(erlang_arr, 4)
    exact = GIM1Calc()
    exact.set_sources(a)
    exact.set_servers(mu=mu)
    exact_results = exact.run()

    print_waiting_moments(results.w, exact_results.w)
    # all three moments: the GIM1Calc higher-moment bug found by this test
    # was fixed (closed form w_k = k! * sigma / (mu*(1-sigma))^k)
    assert np.allclose(results.w, exact_results.w[:3], rtol=1e-3), ERROR_MSG


def test_map_ph1_vs_sim():
    """MAP(MMPP-2)/PH(H2)/1 against simulation with MAP source and PH servers."""
    mmpp = MAP.mmpp([2.0 * ARRIVAL_RATE, 0.4 * ARRIVAL_RATE], np.array([[-0.2, 0.2], [0.3, -0.3]]))
    lam = MAP.arrival_rate(mmpp)
    service_mean = UTILIZATION_FACTOR / lam
    h2_srv = H2Distribution.get_params_by_mean_and_cv(service_mean, 1.2)
    ph_srv = PHDistribution.from_h2(h2_srv)

    calc = MapPh1Calc()
    calc.set_sources(mmpp)
    calc.set_servers(ph_srv)
    calc_results = calc.run()

    sim = QsSim(1, seed=42)
    sim.set_sources(mmpp, "MAP")
    sim.set_servers(ph_srv, "PH")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.allclose(sim_results.w[:2], calc_results.w[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.v[:2], calc_results.v[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_correlation_increases_waiting():
    """
    The point of MAP: positively correlated arrivals (MMPP) give a longer mean
    wait than a renewal process with the SAME interarrival mean and cv.
    """
    mmpp = MAP.mmpp([2.0, 0.2], np.array([[-0.1, 0.1], [0.15, -0.15]]))
    lam = MAP.arrival_rate(mmpp)
    moments = MAP.calc_theory_moments(mmpp, 2)
    cv = float(np.sqrt(moments[1] - moments[0] ** 2) / moments[0])
    assert MAP.lag_correlation(mmpp, 1) > 0.01

    # renewal H2 arrivals with the same mean and cv
    h2_arr = H2Distribution.get_params_by_mean_and_cv(moments[0], cv)
    renewal = MAP.from_ph_renewal(PHDistribution.from_h2(h2_arr))

    service = PHDistribution.from_exp(lam / UTILIZATION_FACTOR)

    w_mmpp = MapPh1Calc()
    w_mmpp.set_sources(mmpp)
    w_mmpp.set_servers(service)

    w_ren = MapPh1Calc()
    w_ren.set_sources(renewal)
    w_ren.set_servers(service)

    w1_mmpp = w_mmpp.run().w[0]
    w1_ren = w_ren.run().w[0]
    print(f"same mean/cv arrivals: renewal w1={w1_ren:.3f}, correlated MMPP w1={w1_mmpp:.3f}")
    assert w1_mmpp > 1.1 * w1_ren, ERROR_MSG


if __name__ == "__main__":
    test_mph1_reduces_to_mm1()
    test_mph1_matches_pollaczek_khinchine()
    test_phph1_matches_gim1()
    test_map_ph1_vs_sim()
    test_correlation_increases_waiting()
