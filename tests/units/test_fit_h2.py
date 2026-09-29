"""
Unit tests for H2 moment fitting (most_queue.random.utils.fit.fit_h2 --
Aliev's method), including the boundary ("one phase distribution") branch
taken when the third raw moment is at or below the minimum achievable for
the given mean/cv.

Regression for a bug found while building the SLA/deadline-violation layer
(EPIC-021): that branch used to return H2 parameters whose implied mean was
off from the target by orders of magnitude -- see
docs/roadmaps/slo_deadline_roadmap.md sec. 11 for the root-cause writeup.
"""

import math

import numpy as np
import pytest

from most_queue.random.utils.fit import fit_h2


def _h2_moments(params, num=3):
    """Raw moments of an H2(p1, mu1, mu2) distribution, computed directly."""
    p1, mu1, mu2 = params.p1, params.mu1, params.mu2
    return [math.factorial(k) * (p1 / mu1 ** (k) + (1.0 - p1) / mu2 ** (k)) for k in range(1, num + 1)]


def _cv(moments):
    return math.sqrt(moments[1] - moments[0] ** 2) / moments[0]


# (mean, m2, m3) triples with cv >= 1. The first two are comfortably interior
# (regular bisection path); the last two sit at/near the H2-feasibility
# boundary and previously triggered the buggy "one phase distribution"
# branch (reproduced from a priority M/G/1 waiting time and an MMPP/PH/1
# LLM-serving example respectively).
INTERIOR_CASES = [
    (1.0, 3.25, 17.875),  # H2 with cv=1.5 (used in examples/llm_serving_slo.py)
    (2.5, 20.0, 250.0),
]

BOUNDARY_CASES = [
    (0.7622563090385152, 2.4362247108555835, 11.675882483075384),  # priority class-0 wait
    (19.941593275971666, 923.2401001924018, 63503.63412899185),  # LLM free-tier wait
]


@pytest.mark.parametrize("moments", INTERIOR_CASES)
def test_fit_h2_interior_matches_all_three_moments(moments):
    """Away from the boundary, fit_h2 reproduces mean, variance and skew."""
    assert _cv(list(moments)) >= 1.0
    params = fit_h2(list(moments))
    fitted = _h2_moments(params)
    assert np.allclose(fitted, moments, rtol=1e-5), (moments, fitted)


@pytest.mark.parametrize("moments", BOUNDARY_CASES)
def test_fit_h2_boundary_matches_mean_and_variance(moments):
    """
    At/below the feasibility boundary, mean and variance (both always
    feasible for cv >= 1) must still match exactly; the third moment clamps
    to the nearest achievable value rather than the (infeasible) target.
    """
    params = fit_h2(list(moments))
    fitted = _h2_moments(params)
    assert np.isclose(fitted[0], moments[0], rtol=1e-6), (moments, fitted)
    assert np.isclose(fitted[1], moments[1], rtol=1e-6), (moments, fitted)
    # Regression guard for the actual bug: the old code implied a mean off
    # by orders of magnitude (e.g. ~1e-7 instead of ~0.76).
    assert fitted[0] > 0.5 * moments[0]


@pytest.mark.parametrize("moments", INTERIOR_CASES + BOUNDARY_CASES)
def test_fit_h2_params_are_valid(moments):
    """p1 in [0, 1], both rates strictly positive."""
    params = fit_h2(list(moments))
    assert 0.0 <= params.p1 <= 1.0
    assert params.mu1 > 0
    assert params.mu2 > 0


def test_fit_h2_degenerate_for_cv_below_one():
    """cv < 1 cannot be represented by H2; fit_h2 returns the documented degenerate sentinel."""
    # Erlang-3-like moments, cv = 1/sqrt(3) < 1.
    params = fit_h2([1.5, 2.5, 4.6875])
    assert params.p1 == 0 and params.mu1 == 0 and params.mu2 == 0


if __name__ == "__main__":
    for m in INTERIOR_CASES:
        test_fit_h2_interior_matches_all_three_moments(m)
    for m in BOUNDARY_CASES:
        test_fit_h2_boundary_matches_mean_and_variance(m)
    for m in INTERIOR_CASES + BOUNDARY_CASES:
        test_fit_h2_params_are_valid(m)
    test_fit_h2_degenerate_for_cv_below_one()
    print("all fit_h2 tests passed")
