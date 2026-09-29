"""
Helpers for size-based M/G/1 formulas.

``build_pdf_cdf`` -- returns ``(pdf_fn, cdf_fn)`` callables for the
distribution identified by *kendall_notation*.

``load_below`` -- numeric building block used by the grid precomputation in
``_SizeBasedCalcBase``. ``upper_bound`` is re-exported from
``most_queue.theory.utils.tail`` (moved there -- it is a generic
tail-bisection primitive, also used by the SLA/deadline-violation layer).
"""

from __future__ import annotations

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
from most_queue.theory.utils.tail import upper_bound

PdfFn = Callable[[float], float]
CdfFn = Callable[[float], float]

__all__ = [
    "get_distribution_class",
    "get_theory_moments",
    "build_pdf_cdf",
    "upper_bound",
    "load_below",
]

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
    """Raw moments E[S], E[S^2], ... from the selected distribution."""
    dist_class = get_distribution_class(kendall_notation)
    return dist_class.calc_theory_moments(params, num)


def build_pdf_cdf(params, kendall_notation: str) -> tuple[PdfFn, CdfFn]:
    """
    Build numeric ``(pdf_fn, cdf_fn)`` callables for *kendall_notation*.

    All supported distributions expose static ``get_pdf`` / ``get_cdf``
    on their class (see ``most_queue.random.distributions``).

    Raises ``ValueError`` for unsupported notations (e.g. ``"C"`` / Cox).
    """
    dist_class = get_distribution_class(kendall_notation)
    return (
        lambda t: dist_class.get_pdf(params, t),
        lambda t: dist_class.get_cdf(params, t),
    )


# ---------------------------------------------------------------------------
# Numeric primitives
# ---------------------------------------------------------------------------


def load_below(l: float, pdf_fn: PdfFn, x_upper: float) -> float:
    """
    Partial load rho_x = lambda * integral_0^{x_upper} t f(t) dt.

    Used by predictor models; ordinary calculators use the precomputed grid.
    """
    if x_upper <= 0:
        return 0.0
    integral, _ = quad(lambda t: t * pdf_fn(t), 0.0, float(x_upper), limit=300)
    return float(l) * integral
