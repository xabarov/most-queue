"""
Unit tests for the heterogeneous (independent, non-identically distributed)
maximum-of-n building block (most_queue.theory.utils.max_dist, EPIC-030).
"""

import numpy as np
import pytest

from most_queue.random.utils.params import ParetoParams
from most_queue.theory.utils.max_dist import heterogeneous_max_moments, pareto_max_moments


def test_reduces_exactly_to_iid_pareto_closed_form():
    """n identical Pareto branches must match the exact closed-form pareto_max_moments."""
    params = ParetoParams(alpha=5.0, K=1.0)
    n = 4
    exact = pareto_max_moments(params, n, 3)
    het = heterogeneous_max_moments([("pareto", params)] * n, 3)
    assert np.allclose(exact, het, rtol=1e-5)


def test_matches_monte_carlo_for_genuinely_mixed_branches():
    """Different families/params per branch, cross-validated against direct sampling."""
    rng = np.random.default_rng(0)
    alpha1, k1 = 5.0, 1.0
    alpha2, k2 = 6.0, 1.5
    gamma_shape, gamma_scale = 2.0, 1.0  # mean=2, var=2 -> raw moments [2, 6, 24]

    branches = [
        ("pareto", ParetoParams(alpha=alpha1, K=k1)),
        ("pareto", ParetoParams(alpha=alpha2, K=k2)),
        ("gamma", [2.0, 6.0, 24.0]),
    ]
    theory = heterogeneous_max_moments(branches, 3)

    n = 2_000_000
    x1 = (rng.pareto(alpha1, n) + 1) * k1
    x2 = (rng.pareto(alpha2, n) + 1) * k2
    x3 = rng.gamma(gamma_shape, gamma_scale, n)
    mx = np.maximum(np.maximum(x1, x2), x3)
    mc = [np.mean(mx**k) for k in (1, 2, 3)]

    assert np.allclose(theory, mc, rtol=0.01)


def test_more_branches_does_not_decrease_the_mean():
    """Adding another (independent) branch to the max cannot lower E[max]."""
    params = ParetoParams(alpha=5.0, K=1.0)
    m3 = heterogeneous_max_moments([("pareto", params)] * 3, 1)[0]
    m4 = heterogeneous_max_moments([("pareto", params)] * 4, 1)[0]
    assert m4 >= m3


def test_empty_branches_rejected():
    with pytest.raises(ValueError):
        heterogeneous_max_moments([], 3)


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        heterogeneous_max_moments([("bogus", None)], 1)


if __name__ == "__main__":
    test_reduces_exactly_to_iid_pareto_closed_form()
    test_matches_monte_carlo_for_genuinely_mixed_branches()
    test_more_branches_does_not_decrease_the_mean()
    test_empty_branches_rejected()
    test_unknown_family_rejected()
    print("all heterogeneous-max tests passed")
