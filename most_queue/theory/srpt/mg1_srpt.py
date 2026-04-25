"""
M/G/1 SRPT calculator (Schrage-Miller 1966).
"""

from __future__ import annotations

from scipy.integrate import quad

from most_queue.structs import QueueResults
from most_queue.theory.srpt._base import _SizeBasedCalcBase


class MG1SrptCalc(_SizeBasedCalcBase):
    """
    Numeric calculator for M/G/1 SRPT.

    Formula (Schrage-Miller 1966)::

        E[T^SRPT(x)] = [? ??? t?f(t)dt + x?(1-F(x))] / [2(1-?_x)?]
                     + ??? dt / (1 - ?_t)

        E[T^SRPT] = ??^? f(x) · E[T^SRPT(x)] dx
        E[W^SRPT] = E[T^SRPT] - b[0]

    All inner integrals are resolved via precomputed grids (see
    ``_SizeBasedCalcBase._build_grids``), so the overall computation
    is a single outer ``scipy.integrate.quad`` call with O(1) interpolated
    lookups — no nested quadrature.
    """

    def conditional_mean_response(self, x: float) -> float:
        """Conditional mean sojourn time E[T^SRPT(x)] for a job of size x."""
        if x <= 0:
            return 0.0
        rho_x = self._rho_interp(x)
        denom = 1.0 - rho_x
        if denom <= 1e-10:
            raise ValueError(f"load ? 1 at x={x}: integral diverges")

        int_t2f = self._t2f_interp(x)
        tail = max(0.0, 1.0 - self._cdf_interp(x))
        first_term = (self.l * int_t2f + x * x * tail) / (2.0 * denom * denom)
        second_term = self._second_int_interp(x)
        return first_term + second_term

    def run(self) -> QueueResults:
        start = self._measure_time()
        self._check_if_servers_and_sources_set()
        utilization = self._check_stability()
        self._build_grids()

        et, _ = quad(
            lambda x: self.pdf_fn(x) * self.conditional_mean_response(float(x)),
            0.0,
            self.x_max,
            limit=300,
        )
        ew = et - self.b[0]

        self.v = [et]
        self.w = [ew]
        result = QueueResults(v=self.v, w=self.w, p=None, utilization=utilization)
        self._set_duration(result, start)
        return result
