"""
Unit tests for k-th order statistic moments, i.i.d. Pareto closed form and
heterogeneous (independent, non-identically distributed) numerical case
(most_queue.theory.utils.max_dist, EPIC-031).
"""

import numpy as np
import pytest

from most_queue.random.utils.params import ParetoParams
from most_queue.theory.utils.max_dist import (
    heterogeneous_kth_order_moments,
    heterogeneous_max_moments,
    pareto_kth_order_moments,
    pareto_max_moments,
)


def test_pareto_kth_order_reduces_to_max_at_k_equals_n():
    params = ParetoParams(alpha=5.0, K=1.0)
    n = 5
    exact_max = pareto_max_moments(params, n, 3)
    kth_at_n = pareto_kth_order_moments(params, n, n, 3)
    assert np.allclose(exact_max, kth_at_n, rtol=1e-9)


def test_pareto_kth_order_moments_increase_with_k():
    params = ParetoParams(alpha=5.0, K=1.0)
    n = 5
    means = [pareto_kth_order_moments(params, n, k, 1)[0] for k in range(1, n + 1)]
    assert all(means[i] <= means[i + 1] for i in range(len(means) - 1))


def test_heterogeneous_kth_order_matches_pareto_closed_form():
    """Identical Pareto branches: the numerical heterogeneous path must match
    the exact closed form at every k, not just k=n."""
    params = ParetoParams(alpha=5.0, K=1.0)
    n = 3
    for k in range(1, n + 1):
        exact = pareto_kth_order_moments(params, n, k, 2)
        het = heterogeneous_kth_order_moments([("pareto", params)] * n, k, 2)
        assert np.allclose(exact, het, rtol=1e-4)


def test_heterogeneous_kth_order_reduces_to_max_at_k_equals_n():
    branches = [
        ("pareto", ParetoParams(alpha=5.0, K=1.0)),
        ("pareto", ParetoParams(alpha=6.0, K=1.5)),
        ("gamma", [2.0, 6.0, 24.0]),
    ]
    n = len(branches)
    kth = heterogeneous_kth_order_moments(branches, n, 3)
    mx = heterogeneous_max_moments(branches, 3)
    assert np.allclose(kth, mx, rtol=1e-6)


def test_heterogeneous_kth_order_matches_monte_carlo():
    rng = np.random.default_rng(3)
    alpha1, k1 = 5.0, 1.0
    alpha2, k2 = 6.0, 1.5

    branches = [
        ("pareto", ParetoParams(alpha=alpha1, K=k1)),
        ("pareto", ParetoParams(alpha=alpha2, K=k2)),
        ("gamma", [2.0, 6.0, 24.0]),
    ]

    n = 2_000_000
    x1 = (rng.pareto(alpha1, n) + 1) * k1
    x2 = (rng.pareto(alpha2, n) + 1) * k2
    x3 = rng.gamma(2.0, 1.0, n)
    stacked = np.sort(np.vstack([x1, x2, x3]), axis=0)

    for k in (1, 2, 3):
        theory = heterogeneous_kth_order_moments(branches, k, 2)
        mc = [np.mean(stacked[k - 1] ** m) for m in (1, 2)]
        assert np.allclose(theory, mc, rtol=0.01)


def test_kth_order_moments_increase_with_k_heterogeneous():
    branches = [
        ("pareto", ParetoParams(alpha=5.0, K=1.0)),
        ("pareto", ParetoParams(alpha=6.0, K=1.5)),
        ("gamma", [2.0, 6.0, 24.0]),
    ]
    means = [heterogeneous_kth_order_moments(branches, k, 1)[0] for k in range(1, len(branches) + 1)]
    assert all(means[i] <= means[i + 1] for i in range(len(means) - 1))


def test_invalid_k_rejected():
    params = ParetoParams(alpha=5.0, K=1.0)
    with pytest.raises(ValueError):
        pareto_kth_order_moments(params, n=3, k=0, num=1)
    with pytest.raises(ValueError):
        pareto_kth_order_moments(params, n=3, k=4, num=1)
    with pytest.raises(ValueError):
        heterogeneous_kth_order_moments([("pareto", params)] * 3, k=0, num=1)
    with pytest.raises(ValueError):
        heterogeneous_kth_order_moments([("pareto", params)] * 3, k=4, num=1)


if __name__ == "__main__":
    test_pareto_kth_order_reduces_to_max_at_k_equals_n()
    test_pareto_kth_order_moments_increase_with_k()
    test_heterogeneous_kth_order_matches_pareto_closed_form()
    test_heterogeneous_kth_order_reduces_to_max_at_k_equals_n()
    test_heterogeneous_kth_order_matches_monte_carlo()
    test_kth_order_moments_increase_with_k_heterogeneous()
    test_invalid_k_rejected()
    print("all k-th order statistic tests passed")
