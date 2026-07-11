"""
Tests for Age of Information (AoI) calculators and simulator.

Closed forms (M/M/1 average and peak, preemptive-LCFS M/M/1 average, general
single-server FCFS peak = E[T] + 1/lambda) are cross-validated against the
discrete-event AoISim.
"""

import numpy as np
import pytest

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.aoi import AoISim
from most_queue.theory.aoi import AoICalc, LcfsPreemptiveAoICalc

AOI_RTOL = 0.04
NUM_UPDATES = 400_000


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.8])
def test_mm1_aoi_vs_sim(rho):
    """M/M/1 average AoI and PAoI closed forms vs simulation."""
    mu, lam = 1.0, rho
    calc = AoICalc()
    calc.set_sources(lam)
    calc.set_servers(mu=mu)
    r = calc.run()

    sim = AoISim(1, "FCFS", seed=1234)
    sim.set_sources(lam, "M")
    sim.set_servers(mu, "M")
    rs = sim.run(NUM_UPDATES)

    assert np.isclose(r.avg_aoi, rs.avg_aoi, rtol=AOI_RTOL)
    assert np.isclose(r.peak_aoi, rs.peak_aoi, rtol=AOI_RTOL)
    # exact closed-form check (no simulation noise)
    assert np.isclose(r.avg_aoi, (1 / mu) * (1 + 1 / rho + rho**2 / (1 - rho)))
    assert np.isclose(r.peak_aoi, 1 / (mu - lam) + 1 / lam)


@pytest.mark.parametrize("rho", [0.3, 0.6])
def test_mg1_peak_aoi_vs_sim(rho):
    """General single-server FCFS PAoI = E[T] + 1/lambda (M/G/1 with H2 service)."""
    b = gamma_moments_by_mean_and_cv(1.0, 2.0)  # mean 1, CV = 2
    lam = rho / b[0]
    calc = AoICalc()
    calc.set_sources(lam)
    calc.set_servers(b=b)
    r = calc.run()

    sim = AoISim(1, "FCFS", seed=1234)
    sim.set_sources(lam, "M")
    sim.set_servers(H2Distribution.get_params(b), "H")
    rs = sim.run(NUM_UPDATES)

    assert np.isclose(r.peak_aoi, rs.peak_aoi, rtol=AOI_RTOL)
    assert r.avg_aoi is None  # M/G/1 average AoI is not moment-closed -> use the simulator


@pytest.mark.parametrize("rho", [0.4, 0.9, 1.5])
def test_lcfs_preemptive_aoi_vs_sim(rho):
    """Preemptive-LCFS M/M/1 average AoI = (1/mu)(1 + 1/rho); stable for any rho."""
    mu, lam = 1.0, rho
    calc = LcfsPreemptiveAoICalc()
    calc.set_sources(lam)
    calc.set_servers(mu)
    r = calc.run()

    sim = AoISim(1, "LCFS-PR", seed=1234)
    sim.set_sources(lam, "M")
    sim.set_servers(mu, "M")
    rs = sim.run(NUM_UPDATES)

    assert np.isclose(r.avg_aoi, rs.avg_aoi, rtol=AOI_RTOL)
    assert np.isclose(r.avg_aoi, (1 / mu) * (1 + 1 / rho))


def test_aoi_sim_reproducible():
    """Seeded AoISim is reproducible."""

    def one():
        s = AoISim(1, "FCFS", seed=99)
        s.set_sources(0.5, "M")
        s.set_servers(1.0, "M")
        return s.run(50_000).avg_aoi

    assert one() == one()


if __name__ == "__main__":
    test_mm1_aoi_vs_sim(0.6)
    test_lcfs_preemptive_aoi_vs_sim(0.9)
