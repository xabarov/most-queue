"""
Unit tests for the exact maximum-of-n-iid-Pareto layer
(most_queue.theory.utils.max_dist.pareto_max_moments / pareto_max_tail).

EPIC-022: unlike the light-tailed (H2/Gamma/Erlang) approximations in
MaxDistribution, this is an exact closed form (via the Beta function), no
quadrature -- see docs/roadmaps/fork_join_heavy_tail_roadmap.md sec. 2 for
the derivation.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from most_queue.random.distributions import ParetoDistribution
from most_queue.random.utils.params import ParetoParams
from most_queue.theory.utils.max_dist import pareto_max_moments, pareto_max_tail

PARAMS = ParetoParams(alpha=3.5, K=2.0)


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_pareto_max_moments_matches_numeric_integration(n):
    """E[max^k] matches direct numeric integration over the U=F(X) substitution."""
    formula = pareto_max_moments(PARAMS, n=n, num=3)
    for k in range(1, 4):
        integral, _ = quad(lambda u, k=k: (1 - u) ** (-k / PARAMS.alpha) * u ** (n - 1), 0, 1, limit=500, points=[1.0])
        numeric = PARAMS.K**k * n * integral
        assert np.isclose(formula[k - 1], numeric, rtol=1e-6), (n, k, formula[k - 1], numeric)


def test_pareto_max_moments_n1_reduces_to_plain_pareto():
    """max of a single Pareto is the Pareto itself."""
    formula = pareto_max_moments(PARAMS, n=1, num=3)
    direct = ParetoDistribution.calc_theory_moments(PARAMS, num=3)
    assert np.allclose(formula, direct, rtol=1e-6)


def test_pareto_max_moments_raises_when_moment_does_not_exist():
    """E[max^k] does not exist for k >= alpha, regardless of n."""
    params = ParetoParams(alpha=2.0, K=1.0)
    with pytest.raises(ValueError, match="does not exist"):
        pareto_max_moments(params, n=3, num=3)
    # k=1 alone is fine (1 < alpha=2)
    assert len(pareto_max_moments(params, n=3, num=1)) == 1


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_pareto_max_tail_boundaries(n):
    """P(max > K) = 1 (CDF is 0 at the support's left edge); P(max > x) -> 0 as x -> inf."""
    assert np.isclose(pareto_max_tail(PARAMS, n, PARAMS.K), 1.0)
    assert pareto_max_tail(PARAMS, n, 1e12) < 1e-9


def test_pareto_max_tail_n1_reduces_to_plain_pareto():
    for x in (2.0, 3.0, 10.0, 100.0):
        assert np.isclose(pareto_max_tail(PARAMS, 1, x), ParetoDistribution.get_tail(PARAMS, x))


def test_pareto_max_tail_monotone_in_x_and_n():
    xs = [2.0, 3.0, 5.0, 10.0, 50.0]
    tails = [pareto_max_tail(PARAMS, 5, x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(tails, tails[1:]))
    # more parallel components -> larger max -> heavier tail at a fixed x
    x = 5.0
    tail_n1 = pareto_max_tail(PARAMS, 1, x)
    tail_n10 = pareto_max_tail(PARAMS, 10, x)
    assert tail_n10 > tail_n1


def test_invalid_n_rejected():
    with pytest.raises(ValueError):
        pareto_max_moments(PARAMS, n=0, num=1)
    with pytest.raises(ValueError):
        pareto_max_tail(PARAMS, n=0, x=1.0)


if __name__ == "__main__":
    for n in (1, 2, 5, 20):
        test_pareto_max_moments_matches_numeric_integration(n)
        test_pareto_max_tail_boundaries(n)
    test_pareto_max_moments_n1_reduces_to_plain_pareto()
    test_pareto_max_moments_raises_when_moment_does_not_exist()
    test_pareto_max_tail_n1_reduces_to_plain_pareto()
    test_pareto_max_tail_monotone_in_x_and_n()
    test_invalid_n_rejected()
    print("all pareto max_dist tests passed")
