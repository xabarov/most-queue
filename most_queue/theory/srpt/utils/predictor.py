"""
Predictor models for SPJF formulas.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

from scipy.integrate import quad

from most_queue.theory.srpt.utils.load_below import load_below

PdfFn = Callable[[float], float]


class Predictor(Protocol):
    """
    Interface expected by MG1SpjfCalc.
    """

    def marginal_y_pdf(self, y: float, service_pdf_fn: PdfFn) -> float:
        """
        Marginal PDF of predicted size g_Y(y).
        """

    def load_below_y(self, l: float, service_pdf_fn: PdfFn, y_upper: float) -> float:
        """
        Partial load ?'_y for predicted classes with Y <= y.
        """


class PerfectPredictor:
    """
    Perfect predictions: Y = X.
    """

    def joint_pdf(self, x: float, y: float, service_pdf_fn: PdfFn) -> float:
        """Joint PDF f(x, y); concentrated on the diagonal y = x."""
        if x <= 0 or y <= 0:
            return 0.0
        if abs(x - y) > 1e-9:
            return 0.0
        return service_pdf_fn(x)

    def marginal_y_pdf(self, y: float, service_pdf_fn: PdfFn) -> float:
        """Marginal PDF of Y -- equals the service-time PDF under perfect predictions."""
        if y < 0:
            return 0.0
        return service_pdf_fn(y)

    def load_below_y(self, l: float, service_pdf_fn: PdfFn, y_upper: float) -> float:
        """Partial load rho'_y = lambda * integral_0^{y_upper} t f(t) dt."""
        return load_below(l, service_pdf_fn, y_upper)


class ExpNoisePredictor:
    """
    Y | X=x ~ Exp(rate=1/x), x > 0.
    """

    @staticmethod
    def _cond_pdf(y: float, x: float) -> float:
        """Conditional PDF g(y | X=x) = Exp(rate=1/x)."""
        if x <= 0 or y < 0:
            return 0.0
        return (1.0 / x) * math.exp(-y / x)

    @staticmethod
    def _cond_cdf(y: float, x: float) -> float:
        """Conditional CDF G(y | X=x) = 1 - exp(-y/x)."""
        if x <= 0:
            return 0.0
        if y < 0:
            return 0.0
        return 1.0 - math.exp(-y / x)

    def joint_pdf(self, x: float, y: float, service_pdf_fn: PdfFn) -> float:
        """Joint PDF f(x) * g(y|x) where g(y|x) = Exp(1/x)."""
        if x <= 0 or y < 0:
            return 0.0
        return service_pdf_fn(x) * self._cond_pdf(y, x)

    def marginal_y_pdf(self, y: float, service_pdf_fn: PdfFn) -> float:
        """Marginal PDF of Y obtained by integrating out the true size X."""
        if y < 0:
            return 0.0
        value, _ = quad(lambda x: service_pdf_fn(x) * self._cond_pdf(y, x), 0.0, math.inf, limit=400)
        return value

    def load_below_y(self, l: float, service_pdf_fn: PdfFn, y_upper: float) -> float:
        """Partial load rho'_y = lambda * integral_0^inf x f(x) P(Y<=y | X=x) dx."""
        if y_upper <= 0:
            return 0.0
        integral, _ = quad(
            lambda x: x * service_pdf_fn(x) * self._cond_cdf(y_upper, x),
            0.0,
            math.inf,
            limit=400,
        )
        return float(l) * integral


class LognormalNoisePredictor:
    """
    Y | X=x ~ LogNormal(log(x), sigma), x > 0.
    """

    def __init__(self, sigma: float = 0.5) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma

    def _cond_pdf(self, y: float, x: float) -> float:
        if x <= 0 or y <= 0:
            return 0.0
        coeff = 1.0 / (y * self.sigma * math.sqrt(2.0 * math.pi))
        exponent = -((math.log(y) - math.log(x)) ** 2) / (2.0 * self.sigma * self.sigma)
        return coeff * math.exp(exponent)

    def _cond_cdf(self, y: float, x: float) -> float:
        if x <= 0:
            return 0.0
        if y <= 0:
            return 0.0
        z = (math.log(y) - math.log(x)) / self.sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def joint_pdf(self, x: float, y: float, service_pdf_fn: PdfFn) -> float:
        """Joint PDF f(x) * g(y|x) where g(y|x) = LogNormal(log x, sigma)."""
        if x <= 0 or y <= 0:
            return 0.0
        return service_pdf_fn(x) * self._cond_pdf(y, x)

    def marginal_y_pdf(self, y: float, service_pdf_fn: PdfFn) -> float:
        """Marginal PDF of Y obtained by integrating out the true size X."""
        if y <= 0:
            return 0.0
        value, _ = quad(lambda x: service_pdf_fn(x) * self._cond_pdf(y, x), 0.0, math.inf, limit=400)
        return value

    def load_below_y(self, l: float, service_pdf_fn: PdfFn, y_upper: float) -> float:
        """Partial load rho'_y = lambda * integral_0^inf x f(x) P(Y<=y | X=x) dx."""
        if y_upper <= 0:
            return 0.0
        integral, _ = quad(
            lambda x: x * service_pdf_fn(x) * self._cond_cdf(y_upper, x),
            0.0,
            math.inf,
            limit=400,
        )
        return float(l) * integral
