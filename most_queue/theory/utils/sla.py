"""
SLA / deadline-violation probability layer.

Turns raw moments of a waiting/sojourn time -- as returned by almost every
calculator in the library (``QueueResults.w``/``.v``,
``MulticlassResults.w``/``.v``) -- into a deadline-violation probability
``P(W > D)`` or an SLO quantile ``D_p`` such that ``P(W > D_p) = p``.

Approach: fit a 2-3 moment distribution (H2 for cv >= 1, Gamma for cv < 1)
and evaluate its closed-form tail/CDF. This is an approximation -- see
``docs/roadmaps/slo_deadline_roadmap.md`` for accuracy caveats in heavy
traffic / far tails. The one exact (non-fit) building block provided here,
``mm1_deadline_violation_prob``, is a correctness regression anchor, not a
general-purpose entry point.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.random.utils.fit import fit_gamma, fit_h2
from most_queue.theory.utils.tail import upper_bound

Family = Literal["auto", "h2", "gamma"]


def _coeff_of_variation(moments: list[float]) -> float:
    """cv = std(W) / E[W] from the first two raw moments."""
    m1, m2 = moments[0], moments[1]
    if m1 <= 0:
        raise ValueError(f"mean (moments[0]) must be positive, got {m1}")
    var = m2 - m1 * m1
    if var < 0:
        # numerical noise for near-deterministic moments
        var = 0.0
    return math.sqrt(var) / m1


def fit_from_moments(moments: list[float], family: Family = "auto") -> tuple[Any, Any]:
    """
    Fit a distribution to raw moments, return ``(params, dist_class)``.

    ``dist_class`` exposes ``get_cdf(params, t)`` (and ``get_tail`` for H2).

    - ``family="auto"``: cv >= 1 -> H2, cv < 1 -> Gamma.
    - ``family="h2"``: force H2 -- requires at least 3 raw moments and raises
      ``ValueError`` if cv < 1 (an H2 mixture cannot represent cv < 1;
      silently falling back would hide a wrong family choice from the
      caller).
    - ``family="gamma"``: force Gamma -- works for any cv > 0 (only the
      first two raw moments are used).

    Uses ``fit_h2`` (Aliev's method), not ``fit_h2_clx``: the latter solves a
    cubic in raw moments and can return complex-valued parameters for moment
    triples outside the strict H2-feasible region even when cv >= 1, which
    ``fit_h2`` avoids by construction (real bisection, degenerates to a
    single-phase fit when the third moment is out of range).

    Defense-in-depth: a boundary bug in ``fit_h2``'s "one phase distribution"
    branch (third raw moment at or below the minimum achievable for the
    given mean/cv -- observed on priority waiting-time moments, whose
    residual-busy-period structure often lands near this boundary) used to
    return parameters whose implied mean was off from the target by orders
    of magnitude; this was found and fixed at the root in ``fit_h2`` itself
    (see ``docs/roadmaps/slo_deadline_roadmap.md`` sec. 11). The implied-mean
    check kept here still guards against any future regression: it falls
    back to Gamma for ``family="auto"`` (Gamma has no such failure mode for
    cv >= 1), or raises for an explicit ``family="h2"`` request, rather than
    silently returning a wrong fit.
    """
    if len(moments) < 2:
        raise ValueError("at least two raw moments (mean, second moment) are required")

    cv = _coeff_of_variation(moments)
    resolved = ("h2" if cv >= 1.0 else "gamma") if family == "auto" else family

    if resolved == "h2":
        if cv < 1.0:
            raise ValueError(f"H2 cannot represent cv < 1 (got cv={cv:.4f}); use family='gamma' or 'auto'")
        if len(moments) < 3:
            raise ValueError("family='h2' requires at least three raw moments")
        params = fit_h2(moments[:3])
        # fit_h2 only ever returns real values ("Only real parameters are
        # selected"); H2Params types them as float | complex for fit_h2_clx.
        p1, mu1, mu2 = float(params.p1.real), float(params.mu1.real), float(params.mu2.real)
        implied_mean = p1 / mu1 + (1.0 - p1) / mu2
        if not math.isclose(implied_mean, moments[0], rel_tol=0.05):
            if family == "auto":
                return fit_gamma(moments[:2]), GammaDistribution
            raise RuntimeError(
                f"fit_h2 degenerated near the H2-feasibility boundary (implied mean "
                f"{implied_mean:.4g} vs target mean {moments[0]:.4g}); use family='gamma' or 'auto'"
            )
        return params, H2Distribution

    if resolved == "gamma":
        params = fit_gamma(moments[:2])
        return params, GammaDistribution

    raise ValueError(f"unknown family: {family!r}; expected 'auto', 'h2' or 'gamma'")


def deadline_violation_prob(moments: list[float], deadline: float, family: Family = "auto") -> float:
    """
    P(W > deadline), fitted from raw moments (see ``fit_from_moments``).
    """
    if deadline < 0:
        raise ValueError("deadline must be non-negative")
    params, dist_class = fit_from_moments(moments, family)
    if hasattr(dist_class, "get_tail"):
        return float(dist_class.get_tail(params, deadline))
    return 1.0 - float(dist_class.get_cdf(params, deadline))


def slo_quantile(moments: list[float], p: float, family: Family = "auto") -> float:
    """
    Smallest deadline D_p such that P(W > D_p) < p (the (1-p) quantile of W).
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    params, dist_class = fit_from_moments(moments, family)

    def cdf_fn(t: float) -> float:
        return dist_class.get_cdf(params, t)

    start = max(moments[0], 1e-6)
    return upper_bound(cdf_fn, p=p, start=start)


def mm1_deadline_violation_prob(lam: float, mu: float, deadline: float) -> float:
    """
    Exact P(W > deadline) for the M/M/1 FCFS waiting time:
    P(W > t) = rho * exp(-mu * (1 - rho) * t), rho = lam / mu.

    Used as a correctness anchor for ``deadline_violation_prob`` (fit-based).
    """
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be positive")
    if deadline < 0:
        raise ValueError("deadline must be non-negative")
    rho = lam / mu
    if rho >= 1:
        raise ValueError(f"system is unstable: rho={rho} must be < 1")
    return rho * math.exp(-mu * (1.0 - rho) * deadline)
