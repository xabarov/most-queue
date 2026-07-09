"""
Tests for the MAP/M/c QBD calculator (EPIC-007): correlated arrivals, c
exponential servers. Cross-validated against Erlang C, MAP/M/1 and simulation.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_waiting_moments, probs_print
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.erlang import ErlangCCalc
from most_queue.theory.matrix.map_mmc import MapMMcCalc
from most_queue.theory.matrix.map_ph1 import MapPh1Calc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_CHANNELS = int(params["num_of_channels"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])


def test_poisson_map_mmc_equals_erlang_c():
    """A one-phase (Poisson) MAP reduces MAP/M/c to the exact Erlang C."""
    mu = ARRIVAL_RATE / (NUM_OF_CHANNELS * UTILIZATION_FACTOR)

    mmc = MapMMcCalc(n=NUM_OF_CHANNELS)
    mmc.set_sources(MAP.poisson(ARRIVAL_RATE))
    mmc.set_servers(mu)
    results = mmc.run()

    erlang = ErlangCCalc(n=NUM_OF_CHANNELS)
    erlang.set_sources(l=ARRIVAL_RATE)
    erlang.set_servers(mu=mu)
    exact = erlang.run()

    assert np.isclose(results.w[0], exact.w[0], rtol=1e-8), ERROR_MSG
    assert np.allclose(results.p[:20], exact.p[:20], atol=1e-9), ERROR_MSG


def test_map_mmc_c1_equals_map_ph1():
    """MAP/M/1 (c=1) equals the single-server MAP/PH/1 with exponential service."""
    mmpp = MAP.mmpp([2.0 * ARRIVAL_RATE, 0.4 * ARRIVAL_RATE], np.array([[-0.2, 0.2], [0.3, -0.3]]))
    lam = MAP.arrival_rate(mmpp)
    mu = lam / UTILIZATION_FACTOR

    mmc = MapMMcCalc(n=1)
    mmc.set_sources(mmpp)
    mmc.set_servers(mu)
    r_mmc = mmc.run()

    ph1 = MapPh1Calc()
    ph1.set_sources(mmpp)
    ph1.set_servers(PHDistribution.from_exp(mu))
    r_ph1 = ph1.run()

    assert np.isclose(r_mmc.w[0], r_ph1.w[0], rtol=1e-6), ERROR_MSG
    assert np.allclose(r_mmc.p[:15], r_ph1.p[:15], atol=1e-8), ERROR_MSG


def test_map_mmc_vs_sim():
    """MAP(MMPP-2)/M/c against simulation with a MAP arrival source."""
    mmpp = MAP.mmpp([2.5 * ARRIVAL_RATE, 0.5 * ARRIVAL_RATE], np.array([[-0.15, 0.15], [0.25, -0.25]]))
    lam = MAP.arrival_rate(mmpp)
    mu = lam / (NUM_OF_CHANNELS * UTILIZATION_FACTOR)

    calc = MapMMcCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(mmpp)
    calc.set_servers(mu)
    calc_results = calc.run()

    sim = QsSim(NUM_OF_CHANNELS, seed=42)
    sim.set_sources(mmpp, "MAP")
    sim.set_servers(mu, "M")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    print_waiting_moments(sim_results.w, calc_results.w)

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.isclose(sim_results.w[0], calc_results.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_map_mmc_correlation_increases_waiting():
    """Correlated arrivals give a longer mean wait than Poisson at the same rate."""
    mu = 1.0
    c = 2
    lam = c * mu * UTILIZATION_FACTOR

    poisson = MapMMcCalc(n=c)
    poisson.set_sources(MAP.poisson(lam))
    poisson.set_servers(mu)
    w_poisson = poisson.run().w[0]

    # a bursty MMPP with the same overall rate lam
    q = np.array([[-0.1, 0.1], [0.1, -0.1]])
    mmpp_raw = MAP.mmpp([3.0, 0.2], q)
    scale = lam / MAP.arrival_rate(mmpp_raw)
    from most_queue.random.map_ph import MAPParams  # pylint: disable=import-outside-toplevel

    mmpp = MAPParams(D0=np.asarray(mmpp_raw.D0) * scale, D1=np.asarray(mmpp_raw.D1) * scale)

    bursty = MapMMcCalc(n=c)
    bursty.set_sources(mmpp)
    bursty.set_servers(mu)
    w_bursty = bursty.run().w[0]

    print(f"MAP/M/{c}: Poisson w1={w_poisson:.4f}, bursty w1={w_bursty:.4f}")
    assert w_bursty > 1.5 * w_poisson, ERROR_MSG


def test_map_mmc_unstable_raises():
    """Utilization >= 1 raises a clear error."""
    mmc = MapMMcCalc(n=2)
    mmc.set_sources(MAP.poisson(5.0))
    mmc.set_servers(2.0)  # rho = 5 / (2*2) = 1.25
    try:
        mmc.run()
        raise AssertionError("expected instability error")
    except ValueError as exc:
        assert "unstable" in str(exc)


if __name__ == "__main__":
    test_poisson_map_mmc_equals_erlang_c()
    test_map_mmc_c1_equals_map_ph1()
    test_map_mmc_vs_sim()
    test_map_mmc_correlation_increases_waiting()
    test_map_mmc_unstable_raises()
