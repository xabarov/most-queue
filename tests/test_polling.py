"""
Tests for M/G/1 polling systems: the pseudo-conservation law and symmetric
per-queue mean waiting time, cross-validated against the polling simulator.
"""

import numpy as np
import pytest

from most_queue.sim.polling import PollingSim
from most_queue.theory.polling import PollingCalc

EXP_MOMENTS = [1.0, 2.0, 6.0]  # exponential service, mean 1
RTOL = 0.05
NJOBS = 600_000


def _sim_waits(Q, lams, discipline, r1=0.5, seed=1234):
    s = PollingSim(Q, discipline, seed=seed)
    s.set_sources(lams, "M")
    s.set_servers([1.0] * Q, "M")
    s.set_switchover(r1, "D")
    res = s.run(NJOBS)
    return [res.w[i][0] for i in range(Q)]


@pytest.mark.parametrize("discipline", ["exhaustive", "gated"])
def test_symmetric_mean_wait_vs_sim(discipline):
    """Symmetric polling: per-queue mean wait W = (pseudo-conservation sum)/rho."""
    Q, lam, r1 = 3, 0.2, 0.5
    calc = PollingCalc(discipline)
    calc.set_sources([lam] * Q)
    calc.set_servers(EXP_MOMENTS)
    calc.set_switchover(r1)
    r = calc.run()

    w_sim = np.mean(_sim_waits(Q, [lam] * Q, discipline, r1))
    assert np.isclose(r.mean_wait_symmetric, w_sim, rtol=RTOL)


@pytest.mark.parametrize("lams", [[0.2, 0.2, 0.2], [0.3, 0.2, 0.1]])
def test_pseudo_conservation_law_vs_sim(lams):
    """The load-weighted sum of mean waits matches the pseudo-conservation law."""
    r1 = 0.5
    calc = PollingCalc("exhaustive")
    calc.set_sources(lams)
    calc.set_servers(EXP_MOMENTS)
    calc.set_switchover(r1)
    pcl = calc.run().pseudo_conservation_sum

    w_sim = _sim_waits(3, lams, "exhaustive", r1)
    sum_sim = sum(lams[i] * EXP_MOMENTS[0] * w_sim[i] for i in range(3))
    assert np.isclose(pcl, sum_sim, rtol=RTOL)


def test_gated_worse_than_exhaustive():
    """Gated has a larger load-weighted wait than exhaustive (extra discipline term)."""
    Q, lam, r1 = 3, 0.2, 0.5

    def pcl(disc):
        c = PollingCalc(disc)
        c.set_sources([lam] * Q)
        c.set_servers(EXP_MOMENTS)
        c.set_switchover(r1)
        return c.run().pseudo_conservation_sum

    assert pcl("gated") > pcl("exhaustive")


def test_symmetric_detection():
    """Asymmetric systems return the pseudo-conservation sum but no single symmetric wait."""
    calc = PollingCalc("exhaustive")
    calc.set_sources([0.3, 0.2, 0.1])
    calc.set_servers(EXP_MOMENTS)
    calc.set_switchover(0.5)
    r = calc.run()
    assert r.mean_wait_symmetric is None
    assert r.pseudo_conservation_sum > 0


if __name__ == "__main__":
    test_symmetric_mean_wait_vs_sim("exhaustive")
    test_pseudo_conservation_law_vs_sim([0.3, 0.2, 0.1])
