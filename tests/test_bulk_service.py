"""
Tests for the M/M^[a,b]/1 bulk-service (batch-service) queue and the LLM-style
size-dependent batching scenario. The analytical CTMC calculator is validated
against the discrete-event simulator and the b=1 M/M/1 special case.
"""

import numpy as np
import pytest

from most_queue.sim.bulk_service import BulkServiceSim
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.fifo.mmnr import MMnrCalc

RTOL = 0.05  # simulation cross-validation tolerance (higher-load cases are noisier)
NUM_JOBS = 400_000


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.8])
def test_b1_equals_mm1(rho):
    """A batch size of 1 reduces to M/M/1 exactly."""
    mu, lam = 1.0, rho
    calc = BulkServiceMM1Calc(a=1, b=1)
    calc.set_sources(lam)
    calc.set_servers(mu)
    v = calc.run().v[0]

    mm1 = MMnrCalc(n=1, r=400)
    mm1.set_sources(l=lam)
    mm1.set_servers(mu=mu)
    assert np.isclose(v, mm1.run().v[0], rtol=1e-6)


@pytest.mark.parametrize("b, lam", [(3, 1.5), (5, 2.5), (3, 2.5)])
def test_bulk_service_vs_sim(b, lam):
    """M/M^[1,b]/1 mean sojourn/waiting: CTMC calculator vs simulation."""
    calc = BulkServiceMM1Calc(a=1, b=b, queue_truncation=400)
    calc.set_sources(lam)
    calc.set_servers(1.0)
    r = calc.run()
    assert calc.boundary_mass < 1e-3

    sim = BulkServiceSim(a=1, b=b, seed=1234)
    sim.set_sources(lam, "M")
    sim.set_servers(1.0)
    rs = sim.run(NUM_JOBS)

    assert np.isclose(r.v[0], rs.v[0], rtol=RTOL)
    assert np.isclose(r.w[0], rs.w[0], rtol=RTOL + 0.02)


def test_fixed_batch_size_vs_sim():
    """a = b (serve exactly b at a time) vs simulation."""
    calc = BulkServiceMM1Calc(a=3, b=3)
    calc.set_sources(2.0)
    calc.set_servers(1.0)
    v = calc.run().v[0]

    sim = BulkServiceSim(a=3, b=3, seed=1234)
    sim.set_sources(2.0, "M")
    sim.set_servers(1.0)
    assert np.isclose(v, sim.run(NUM_JOBS).v[0], rtol=RTOL)


def test_llm_batching_optimal_size():
    """
    LLM-style batching: batch service time grows with batch size, so a larger max
    batch improves throughput but adds per-batch latency. The mean response time
    is far lower with batching than without (b=1), matching simulation.
    """
    base, coef = 0.3, 0.08
    mu = lambda i: 1.0 / (base + coef * i)  # noqa: E731
    lam = 2.0

    def et(b):
        c = BulkServiceMM1Calc(a=1, b=b, queue_truncation=400)
        c.set_sources(lam)
        c.set_servers(mu)
        return c.run().v[0]

    et1, et8 = et(1), et(8)
    assert et8 < et1  # batching helps

    sim = BulkServiceSim(a=1, b=8, seed=1234)
    sim.set_sources(lam, "M")
    sim.set_servers(mu)
    assert np.isclose(et8, sim.run(NUM_JOBS).v[0], rtol=RTOL)


if __name__ == "__main__":
    test_b1_equals_mm1(0.6)
    test_bulk_service_vs_sim(3, 1.5)
    test_llm_batching_optimal_size()
