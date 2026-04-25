"""
M/G/1 SPJF calculator (Mitzenmacher 2020).
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.srpt.utils.load_below import build_pdf_cdf, get_theory_moments
from most_queue.theory.srpt.utils.predictor import PerfectPredictor, Predictor


class MG1SpjfCalc(BaseQueue):
    """
    Numeric calculator for M/G/1 SPJF with a pluggable predictor model.

    Formula (Mitzenmacher 2020)::

        E[W^SPJF(y)] = lam * E[S^2] / [2(1 - rho'_y)^2]
        E[W^SPJF]    = int_0^inf g_Y(y) * E[W^SPJF(y)] dy
        E[T^SPJF]    = E[W^SPJF] + b[0]

    where rho'_y and g_Y(y) are supplied by the predictor object.

    With ``PerfectPredictor`` (Y = X) the result equals ``MG1SjfCalc``.

    Usage::

        calc = MG1SpjfCalc()
        calc.set_sources(1.0)
        calc.set_servers(h2_params, "H")
        calc.set_predictor(ExpNoisePredictor())
        result = calc.run()
    """

    def __init__(self) -> None:
        super().__init__(n=1)
        self.l: float | None = None
        self.b: list[float] | None = None
        self.pdf_fn = None
        self.cdf_fn = None
        self.predictor: Predictor = PerfectPredictor()

    def set_sources(self, l: float) -> None:  # pylint: disable=arguments-differ
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, params, kendall_notation: str = "H") -> None:  # pylint: disable=arguments-differ
        pdf_fn, cdf_fn = build_pdf_cdf(params, kendall_notation)
        self.pdf_fn = pdf_fn
        self.cdf_fn = cdf_fn
        self.b = get_theory_moments(params, kendall_notation, 3)
        self.is_servers_set = True

    def set_predictor(self, predictor: Predictor | None) -> None:
        """Set the prediction model. Pass ``None`` to restore perfect predictions."""
        self.predictor = PerfectPredictor() if predictor is None else predictor

    def conditional_mean_wait(self, y: float) -> float:
        """Conditional mean wait E[W^SPJF(y)] for a job with predicted size y."""
        if y <= 0:
            return 0.0
        rho_y = self.predictor.load_below_y(self.l, self.pdf_fn, y)
        denom = 1.0 - rho_y
        if denom <= 1e-10:
            raise ValueError(f"load >= 1 at y={y}: integral diverges")
        return (self.l * self.b[1]) / (2.0 * denom * denom)

    def run(self) -> QueueResults:
        """Compute E[W^SPJF] and E[T^SPJF] using the configured predictor model."""
        start = self._measure_time()
        self._check_if_servers_and_sources_set()

        utilization = self.l * self.b[0]
        if utilization >= 1.0:
            raise ValueError("System is unstable: utilization must be < 1")

        ew, _ = quad(
            lambda y: (self.predictor.marginal_y_pdf(float(y), self.pdf_fn) * self.conditional_mean_wait(float(y))),
            0.0,
            math.inf,
            limit=600,
            epsabs=1e-9,
            epsrel=1e-7,
        )
        et = ew + self.b[0]

        self.w = [ew]
        self.v = [et]
        result = QueueResults(v=self.v, w=self.w, p=None, utilization=utilization)
        self._set_duration(result, start)
        return result
