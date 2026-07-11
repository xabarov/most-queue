"""
Tests for mean-field load-balancing calculators (power-of-d, JSQ, JIQ) and the
finite-N dispatching simulator.
"""

import numpy as np
import pytest

from most_queue.sim.load_balancing import LoadBalancingSim
from most_queue.theory.load_balancing import LoadBalancingMeanField


def _mf(policy, rho, d=2, mu=1.0):
    c = LoadBalancingMeanField(policy, d)
    c.set_sources(rho)
    c.set_servers(mu)
    return c.run()


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_d1_is_mm1(rho):
    """Random dispatch (d=1) makes each queue an independent M/M/1."""
    r = _mf("power-of-d", rho, d=1)
    assert np.isclose(r.w, 1.0 / (1.0 - rho))  # mu = 1
    assert np.isclose(_mf("random", rho).w, 1.0 / (1.0 - rho))


def test_power_of_two_tail_is_double_exponential():
    """power-of-d mean-field tail: s_k = rho ** ((d^k - 1)/(d-1))."""
    rho = 0.9
    r = _mf("power-of-d", rho, d=2)
    for k in range(6):
        assert np.isclose(r.tail[k], rho ** (2**k - 1))


def test_jiq_jsq_zero_wait():
    """JIQ and JSQ achieve asymptotically zero waiting below capacity (W -> 1/mu)."""
    for policy in ("jiq", "jsq"):
        r = _mf(policy, 0.9)
        assert np.isclose(r.w, 1.0)  # mu = 1 -> W = service time only
        assert r.wait < 1e-9


@pytest.mark.parametrize("policy, d", [("power-of-d", 2), ("power-of-d", 3), ("jiq", 2), ("jsq", 2)])
def test_meanfield_matches_finite_n_sim(policy, d):
    """The mean-field response time matches a large-N simulation."""
    rho = 0.9
    r = _mf(policy, rho, d=d)
    sim = LoadBalancingSim(200, policy, d, seed=1234)
    sim.set_sources(rho)
    sim.set_servers(1.0)
    w_sim = sim.run(300_000).v[0]
    assert np.isclose(r.w, w_sim, rtol=0.05)


if __name__ == "__main__":
    test_d1_is_mm1(0.9)
    test_power_of_two_tail_is_double_exponential()
    test_jiq_jsq_zero_wait()
