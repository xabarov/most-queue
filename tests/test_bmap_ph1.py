"""
Tests for the BMAP/PH/1 queue (EPIC-008): batch Markovian arrivals, phase-type
service. Cross-validated against BMAP/M/1, MAP/PH/1 and simulation.
"""

import os

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_waiting_moments, probs_print
from most_queue.random.distributions import H2Distribution
from most_queue.random.map_ph import MAP, BMAPParams, PHDistribution, bmap_from_map, bmap_poisson_batch
from most_queue.sim.bmap import BmapPh1Sim
from most_queue.theory.matrix.bmap_m1 import BmapM1Calc
from most_queue.theory.matrix.bmap_ph1 import BmapPh1Calc
from most_queue.theory.matrix.map_ph1 import MapPh1Calc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]
PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

BATCH_PROBS = [0.2, 0.3, 0.1, 0.2, 0.2]


def test_bmap_ph1_exp_equals_bmap_m1():
    """One-phase PH service (exponential) reduces BMAP/PH/1 to BMAP/M/1."""
    lam, mu = 0.5, 2.5
    bmap = bmap_poisson_batch(lam, BATCH_PROBS)

    bph = BmapPh1Calc()
    bph.set_sources(bmap)
    bph.set_servers(PHDistribution.from_exp(mu))
    r_ph = bph.run()

    bm = BmapM1Calc()
    bm.set_sources(bmap)
    bm.set_servers(mu)
    r_m = bm.run()

    assert np.allclose(r_ph.p[:25], r_m.p[:25], atol=1e-7), ERROR_MSG
    assert np.isclose(r_ph.w[0], r_m.w[0], rtol=1e-5), ERROR_MSG


def test_bmap_ph1_size1_equals_map_ph1():
    """A BMAP with only size-1 batches reduces BMAP/PH/1 to MAP/PH/1."""
    mmpp = MAP.mmpp([2.0 * 0.5, 0.4 * 0.5], np.array([[-0.2, 0.2], [0.3, -0.3]]))
    lam = MAP.arrival_rate(mmpp)
    h2 = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(UTILIZATION_FACTOR / lam, 1.4))

    bph = BmapPh1Calc()
    bph.set_sources(bmap_from_map(mmpp))
    bph.set_servers(h2)
    r_ph = bph.run()

    mp = MapPh1Calc()
    mp.set_sources(mmpp)
    mp.set_servers(h2)
    r_mp = mp.run()

    assert np.allclose(r_ph.p[:15], r_mp.p[:15], atol=1e-7), ERROR_MSG
    assert np.isclose(r_ph.w[0], r_mp.w[0], rtol=1e-5), ERROR_MSG


@pytest.mark.slow
def test_bmap_ph1_poisson_batch_vs_sim():
    """Poisson-batch/H2/1 against simulation."""
    lam, b_mean = 0.4, 0.5
    bmap = bmap_poisson_batch(lam, BATCH_PROBS)
    h2 = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(b_mean, 1.3))

    calc = BmapPh1Calc()
    calc.set_sources(bmap)
    calc.set_servers(h2)
    r_calc = calc.run()

    sim = BmapPh1Sim(bmap, h2, seed=42)
    r_sim = sim.run(300_000)

    probs_print(r_sim.p, r_calc.p, size=10)
    print_waiting_moments(r_sim.w, r_calc.w)
    assert np.allclose(r_sim.p[:10], r_calc.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.isclose(r_sim.w[0], r_calc.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


@pytest.mark.slow
def test_bmap_ph1_correlated_batches_vs_sim():
    """MMPP with mixed batch sizes + H2 service against simulation."""
    mmpp = MAP.mmpp([2.0, 0.4], np.array([[-0.2, 0.2], [0.3, -0.3]]))
    d0, d1 = np.asarray(mmpp.D0), np.asarray(mmpp.D1)
    bmap = BMAPParams(D=[d0, 0.5 * d1, 0.5 * d1])  # half size-1, half size-2

    from most_queue.random.map_ph import bmap_arrival_rate  # pylint: disable=import-outside-toplevel

    lam = bmap_arrival_rate(bmap)
    h2 = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(0.6 / lam, 1.2))

    calc = BmapPh1Calc()
    calc.set_sources(bmap)
    calc.set_servers(h2)
    r_calc = calc.run()

    sim = BmapPh1Sim(bmap, h2, seed=7)
    r_sim = sim.run(300_000)

    assert np.allclose(r_sim.p[:10], r_calc.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
    assert np.isclose(r_sim.v[0], r_calc.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_bmap_ph1_exp_equals_bmap_m1()
    test_bmap_ph1_size1_equals_map_ph1()
    test_bmap_ph1_poisson_batch_vs_sim()
    test_bmap_ph1_correlated_batches_vs_sim()
