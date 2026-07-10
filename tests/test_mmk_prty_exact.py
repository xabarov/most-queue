"""
Tests for the exact truncated-CTMC solver of M/M/k with m preemptive-resume
priority classes, and its use as a noise-free reference for RDR-A.
"""

import numpy as np
import pytest

from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.priority.preemptive.mmk_prty_exact import MMkPriorityExact
from most_queue.theory.priority.preemptive.rdr_a import RDRAPriorityCalc


def test_exact_reduces_to_mmn_single_class():
    """With one class the exact CTMC is M/M/k."""
    ex = MMkPriorityExact(n=2, truncation=200)
    ex.set_sources([1.2])
    ex.set_servers([1.0])
    v_exact = ex.run().v[0][0]

    mmn = MMnrCalc(n=2, r=400)
    mmn.set_sources(l=1.2)
    mmn.set_servers(mu=1.0)
    assert np.isclose(v_exact, mmn.run().v[0], rtol=1e-4)


@pytest.mark.parametrize(
    "n, lambdas, mus",
    [
        (2, [0.4, 0.4, 0.4], [1.0, 1.0, 1.0]),  # m=3 equal mu
        (2, [0.3, 0.3, 0.3], [1.2, 1.0, 0.8]),  # m=3 heterogeneous mu
        (2, [0.25, 0.25, 0.25, 0.25], [1.0, 1.0, 1.0, 1.0]),  # m=4 identical, rho=0.5
        (3, [0.4, 0.4, 0.4, 0.4], [1.0, 1.0, 1.0, 1.0]),  # M/M/3, m=4, rho=0.53
    ],
)
def test_rdr_a_matches_exact_ctmc(n, lambdas, mus):
    """
    RDR-A per-class mean response time must match the exact CTMC very closely
    (this is a noise-free check, far tighter than validation against simulation).
    """
    ex = MMkPriorityExact(n=n)
    ex.set_sources(lambdas)
    ex.set_servers(mus)
    res_exact = ex.run()
    assert ex.boundary_mass < 1e-2, f"truncation too tight: boundary mass {ex.boundary_mass:.2e}"
    v_exact = [float(res_exact.v[i][0]) for i in range(len(lambdas))]

    ra = RDRAPriorityCalc(n=n)
    ra.set_sources(lambdas)
    ra.set_servers(mus)
    res_a = ra.run()
    v_a = [float(np.asarray(res_a.v[i][0]).real) for i in range(len(lambdas))]

    for i, (a, e) in enumerate(zip(v_a, v_exact)):
        assert np.isclose(a, e, rtol=0.03), f"class {i}: RDR-A {a:.4f} vs exact {e:.4f}"


def test_response_variance_highest_class_closed_form():
    """
    Highest class in M/M/1 sees an M/M/1: sojourn ~ Exp(mu - lam), so
    E[T^2] = 2 / (mu - lam)^2. Validates the tagged-job second-moment method.
    """
    lam, mu = 0.5, 1.0
    ex = MMkPriorityExact(n=1, truncation=[400, 1], with_variance=True)
    ex.set_sources([lam, 0.1])
    ex.set_servers([mu, 1.0])
    e_t2 = ex.run().v[0][1]
    assert np.isclose(e_t2, 2.0 / (mu - lam) ** 2, rtol=1e-4)


@pytest.mark.parametrize(
    "n, lambdas, mus",
    [
        (1, [0.4, 0.3], [1.0, 1.0]),
        (2, [0.4, 0.4, 0.4], [1.0, 1.0, 1.0]),
        (2, [0.3, 0.3, 0.3], [1.2, 1.0, 0.8]),
    ],
)
def test_response_variance_mean_consistency(n, lambdas, mus):
    """
    The tagged-job absorbing-chain mean must equal the Little's-law mean for every
    class (the mean is discipline-invariant), a noise-free check that the auxiliary
    chain used for the second moment is constructed correctly.
    """
    ex = MMkPriorityExact(n=n, with_variance=True)
    ex.set_sources(lambdas)
    ex.set_servers(mus)
    res = ex.run()
    assert ex.tagged_mean_check is not None
    for i, (little, tagged) in enumerate(ex.tagged_mean_check):
        assert np.isclose(little, tagged, rtol=1e-3), f"class {i}: Little {little} vs tagged {tagged}"
        # second moment must exceed the square of the mean (Var >= 0)
        assert res.v[i][1] >= little**2


if __name__ == "__main__":
    test_exact_reduces_to_mmn_single_class()
    test_rdr_a_matches_exact_ctmc(2, [0.4, 0.4, 0.4], [1.0, 1.0, 1.0])
    test_response_variance_highest_class_closed_form()
