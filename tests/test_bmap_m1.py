"""
Tests for the BMAP/M/1 queue (EPIC-007): batch Markovian arrivals, single
exponential server. Cross-validated against the exact M^[X]/M/1 (BatchMM1)
and MAP/M/1.
"""

import os

import numpy as np
import yaml

from most_queue.random.map_ph import MAP, bmap_from_map, bmap_poisson_batch
from most_queue.theory.batch.mm1 import BatchMM1
from most_queue.theory.matrix.bmap_m1 import BmapM1Calc
from most_queue.theory.matrix.map_mmc import MapMMcCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

BATCH_PROBS = [0.2, 0.3, 0.1, 0.2, 0.2]  # sizes 1..5, mean 2.9


def test_bmap_m1_equals_batch_mm1():
    """A single-phase Poisson-batch BMAP/M/1 equals the exact M^[X]/M/1."""
    lam, mu = 0.5, 2.5  # mean batch 2.9 -> rho = 0.58

    bmap = BmapM1Calc()
    bmap.set_sources(bmap_poisson_batch(lam, BATCH_PROBS))
    bmap.set_servers(mu)
    r_bmap = bmap.run()

    batch = BatchMM1()
    batch.set_sources(l=lam, batch_probs=BATCH_PROBS)
    batch.set_servers(mu=mu)
    r_batch = batch.run()

    assert np.allclose(r_bmap.p[:30], r_batch.p[:30], atol=1e-8), ERROR_MSG
    assert np.isclose(r_bmap.w[0], r_batch.w[0], rtol=1e-6), ERROR_MSG
    assert np.isclose(r_bmap.v[0], r_batch.v[0], rtol=1e-6), ERROR_MSG


def test_bmap_m1_size1_equals_map_m1():
    """A BMAP with only size-1 batches (a MAP) equals MAP/M/1."""
    mmpp = MAP.mmpp([2.0 * 0.5, 0.4 * 0.5], np.array([[-0.2, 0.2], [0.3, -0.3]]))
    lam = MAP.arrival_rate(mmpp)
    mu = lam / UTILIZATION_FACTOR

    bmap = BmapM1Calc()
    bmap.set_sources(bmap_from_map(mmpp))
    bmap.set_servers(mu)
    r_bmap = bmap.run()

    mapm1 = MapMMcCalc(n=1)
    mapm1.set_sources(mmpp)
    mapm1.set_servers(mu)
    r_map = mapm1.run()

    assert np.allclose(r_bmap.p[:15], r_map.p[:15], atol=1e-7), ERROR_MSG
    assert np.isclose(r_bmap.w[0], r_map.w[0], rtol=1e-5), ERROR_MSG
    assert np.isclose(r_bmap.v[0], r_map.v[0], rtol=1e-5), ERROR_MSG


def test_bmap_m1_sojourn_waiting_consistency():
    """E[V] = E[W] + 1/mu for the single-server queue."""
    lam, mu = 0.4, 3.0
    bmap = BmapM1Calc()
    bmap.set_sources(bmap_poisson_batch(lam, BATCH_PROBS))
    bmap.set_servers(mu)
    r = bmap.run()
    assert np.isclose(r.v[0], r.w[0] + 1.0 / mu, rtol=1e-8), ERROR_MSG


def test_bmap_m1_unstable_raises():
    """Utilization >= 1 raises a clear error."""
    bmap = BmapM1Calc()
    bmap.set_sources(bmap_poisson_batch(0.5, BATCH_PROBS))  # job rate 1.45
    bmap.set_servers(1.0)  # rho = 1.45
    try:
        bmap.run()
        raise AssertionError("expected instability error")
    except ValueError as exc:
        assert "unstable" in str(exc)


if __name__ == "__main__":
    test_bmap_m1_equals_batch_mm1()
    test_bmap_m1_size1_equals_map_m1()
    test_bmap_m1_sojourn_waiting_consistency()
    test_bmap_m1_unstable_raises()
