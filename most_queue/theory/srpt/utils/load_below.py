"""
Helpers for size-based M/G/1 formulas.

``build_pdf_cdf`` — returns ``(pdf_fn, cdf_fn)`` callables for the
distribution identified by *kendall_notation*.

``load_below`` / ``upper_bound`` — numeric building blocks used by the
grid precomputation in ``_SizeBasedCalcBase``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.integrate import quad

from most_queue.random.distributions import (
    DeterministicDistribution,
    ErlangDistribution,
    ExpDistribution,
    GammaDistribution,
    H2Distribution,
    NormalDistribution,
    ParetoDistribution,
    UniformDistribution,
)

PdfFn = Callable[[float], float]
CdfFn = Callable[[float], float]

# Distributions that expose static get_pdf/get_cdf.
# CoxDistribution is intentionally excluded: it has no density helpers.
KENDALL_TO_CLASS = {
    "M": ExpDistribution,
    "H": H2Distribution,
    "E": ErlangDistribution,
    "Gamma": GammaDistribution,
    "Pa": ParetoDistribution,
    "Uniform": UniformDistribution,
    "Norm": NormalDistribution,
    "D": DeterministicDistribution,
}


def get_distribution_class(kendall_notation: str):
    """Resolve distribution class by Kendall notation."""
    dist_class = KENDALL_TO_CLASS.get(kendall_notation)
    if dist_class is None:
        raise ValueError(
            f"Unsupported kendall notation for size-based formulas: {kendall_notation!r}. "
            f"Supported: {list(KENDALL_TO_CLASS)}"
        )
    return dist_class


def get_theory_moments(params, kendall_notation: str, num: int = 3) -> list[float]:
    """Raw moments E[S], E[S?], … from the selected distribution."""
    dist_class = get_distribution_class(kendall_notation)
    return dist_class.calc_theory_moments(params, num)


# ---------------------------------------------------------------------------
# Private helpers for distributions that lack static get_pdf / get_cdf
# ---------------------------------------------------------------------------


def _exp_pdf(mu: float, t: float) -> float:
    if t < 0:
        return 0.0
    return float(mu) * math.exp(-float(mu) * t)


def _exp_cdf(mu: float, t: float) -> float:
    if t < 0:
        return 0.0
    return 1.0 - math.exp(-float(mu) * t)


def _erlang_pdf(params, t: float) -> float:
    """Erlang PDF (r >= 1)."""
    if t < 0:
        return 0.0
    r, mu = params.r, params.mu
    return (mu**r) * (t ** (r - 1)) * math.exp(-mu * t) / math.factorial(r - 1)


def _normal_pdf(params, t: float) -> float:
    """Truncated-normal PDF (t >= 0).  Mass below 0 is ignored."""
    if t < 0:
        return 0.0
    from scipy.stats import norm  # lazy import — only needed for Norm distribution

    return float(norm.pdf(t, loc=params.mean, scale=params.std_dev))


def _normal_cdf(params, t: float) -> float:
    if t < 0:
        return 0.0
    from scipy.stats import norm  # lazy import

    return float(norm.cdf(t, loc=params.mean, scale=params.std_dev))


def _deterministic_pdf(_params, _t: float) -> float:
    raise ValueError(
        "Deterministic distribution ('D') does not have a density function; "
        "size-based SRPT/SJF/PSJF formulas require a continuous service distribution."
    )


def _deterministic_cdf(params, t: float) -> float:
    """CDF of D(b): step at b.  *params* is the deterministic value b (float)."""
    return 0.0 if t < float(params) else 1.0


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_pdf_cdf(params, kendall_notation: str) -> tuple[PdfFn, CdfFn]:
    """
    Build numeric ``(pdf_fn, cdf_fn)`` callables for *kendall_notation*.

    For distributions that already expose static ``get_pdf`` / ``get_cdf``
    (H2, Gamma, Pareto, Uniform), those methods are used directly.
    For Exp, Erlang, Norm and Deterministic, dedicated helpers are provided.

    Raises ``ValueError`` for unsupported notations (e.g. ``"C"`` / Cox).
    """
    get_distribution_class(kendall_notation)  # validates notation early

    if kendall_notation == "M":
        return lambda t: _exp_pdf(params, t), lambda t: _exp_cdf(params, t)
    if kendall_notation == "E":
        return (
            lambda t: _erlang_pdf(params, t),
            lambda t: ErlangDistribution.get_cdf(params, t),
        )
    if kendall_notation == "Norm":
        return lambda t: _normal_pdf(params, t), lambda t: _normal_cdf(params, t)
    if kendall_notation == "D":
        return (
            lambda t: _deterministic_pdf(params, t),
            lambda t: _deterministic_cdf(params, t),
        )

    # Remaining distributions (H, Gamma, Pa, Uniform) have get_pdf / get_cdf.
    dist_class = KENDALL_TO_CLASS[kendall_notation]
    return (
        lambda t: dist_class.get_pdf(params, t),
        lambda t: dist_class.get_cdf(params, t),
    )


# ---------------------------------------------------------------------------
# Numeric primitives
# ---------------------------------------------------------------------------


def upper_bound(cdf_fn: CdfFn, p: float = 1e-7, start: float = 1.0, max_iter: int = 100) -> float:
    """
    Find the smallest x such that the tail 1 ? CDF(x) < p.

    Uses exponential expansion to bracket the root, then 80 bisection steps.
    """
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    if start <= 0:
        raise ValueError("start must be positive")

    left = 0.0
    right = float(start)

    for _ in range(max_iter):
        if 1.0 - float(cdf_fn(right)) < p:
            break
        left = right
        right *= 2.0
    else:
        raise ValueError("Could not find a finite upper integration bound")

    for _ in range(80):
        mid = 0.5 * (left + right)
        if 1.0 - float(cdf_fn(mid)) < p:
            right = mid
        else:
            left = mid

    return right


def load_below(l: float, pdf_fn: PdfFn, x_upper: float) -> float:
    """
    Partial load ?_x = ? · ??^{x_upper} t f(t) dt.

    Used by predictor models; ordinary calculators use the precomputed grid.
    """
    if x_upper <= 0:
        return 0.0
    integral, _ = quad(lambda t: t * pdf_fn(t), 0.0, float(x_upper), limit=300)
    return float(l) * integral
