"""
Tests for the SLA / deadline-violation probability layer
(most_queue.theory.utils.sla): fit-based P(W > deadline) and the SLO
quantile, checked against the exact M/M/1 closed form and basic sanity
properties (monotonicity, round-trip).
"""

import numpy as np

from most_queue.random.distributions import ErlangDistribution, GammaDistribution, H2Distribution
from most_queue.random.utils.params import ErlangParams
from most_queue.theory.utils.sla import (
    deadline_violation_prob,
    fit_from_moments,
    mm1_deadline_violation_prob,
    slo_quantile,
)


def _mm1_wait_moments(lam: float, mu: float, num: int = 3) -> list[float]:
    """Raw moments of the M/M/1 FCFS waiting time: E[W^k] = rho * k! / (mu*(1-rho))^k."""
    rho = lam / mu
    mu_eff = mu * (1.0 - rho)
    import math

    return [rho * math.factorial(k) / mu_eff**k for k in range(1, num + 1)]


def test_mm1_fit_matches_exact_away_from_zero():
    """
    Fitting H2 to the M/M/1 waiting-time moments and reading off the tail
    must match the closed-form P(W > t) = rho * exp(-mu*(1-rho)*t) for any
    t > 0. (At t = 0 the fit necessarily disagrees: the true distribution has
    an atom at zero, which no continuous H2 density can reproduce -- that is
    exactly why `mm1_deadline_violation_prob` exists as a separate exact
    building block.)
    """
    lam, mu = 0.7, 1.0
    moments = _mm1_wait_moments(lam, mu)
    for deadline in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        exact = mm1_deadline_violation_prob(lam, mu, deadline)
        fit = deadline_violation_prob(moments, deadline)
        assert np.isclose(exact, fit, rtol=1e-6, atol=1e-8), (deadline, exact, fit)


def test_deadline_violation_prob_monotone_and_bounded():
    """P(W > D) is non-increasing in D and stays within [0, 1]."""
    moments = _mm1_wait_moments(0.8, 1.0)
    deadlines = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    probs = [deadline_violation_prob(moments, d) for d in deadlines]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(b <= a + 1e-9 for a, b in zip(probs, probs[1:]))


def test_slo_quantile_roundtrip_h2_and_gamma():
    """slo_quantile is the (approximate) inverse of deadline_violation_prob."""
    # cv >= 1 -> auto-selects H2
    moments_h2 = _mm1_wait_moments(0.7, 1.0)
    cv = (moments_h2[1] - moments_h2[0] ** 2) ** 0.5 / moments_h2[0]
    assert cv >= 1.0

    # cv < 1 -> auto-selects Gamma (Erlang-3 has cv = 1/sqrt(3) < 1)
    moments_gamma = ErlangDistribution.calc_theory_moments(ErlangParams(r=3, mu=2.0), num=3)

    for moments in (moments_h2, moments_gamma):
        for p in (0.01, 0.05, 0.2):
            d = slo_quantile(moments, p)
            back = deadline_violation_prob(moments, d)
            assert np.isclose(back, p, rtol=1e-3, atol=1e-4), (moments, p, d, back)


def test_auto_family_selection():
    """family='auto' picks H2 for cv >= 1 and Gamma for cv < 1."""
    moments_h2 = _mm1_wait_moments(0.7, 1.0)
    _, dist_class = fit_from_moments(moments_h2, family="auto")
    assert dist_class is H2Distribution

    moments_gamma = ErlangDistribution.calc_theory_moments(ErlangParams(r=3, mu=2.0), num=3)
    _, dist_class = fit_from_moments(moments_gamma, family="auto")
    assert dist_class is GammaDistribution


def test_family_h2_rejects_cv_below_one():
    """Forcing family='h2' on a cv < 1 moment set must fail loudly, not silently degrade."""
    moments_gamma = ErlangDistribution.calc_theory_moments(ErlangParams(r=3, mu=2.0), num=3)
    try:
        fit_from_moments(moments_gamma, family="h2")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_mm1_fit_matches_exact_away_from_zero()
    test_deadline_violation_prob_monotone_and_bounded()
    test_slo_quantile_roundtrip_h2_and_gamma()
    test_auto_family_selection()
    test_family_h2_rejects_cv_below_one()
