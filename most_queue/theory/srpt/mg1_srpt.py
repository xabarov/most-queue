"""
M/G/1 SRPT calculator (Schrage-Miller 1966).
"""

from __future__ import annotations

from most_queue.structs import QueueResults
from most_queue.theory.srpt._base import _SizeBasedCalcBase


class MG1SrptCalc(_SizeBasedCalcBase):
    """
    Numeric calculator for M/G/1 SRPT.

    Conditional mean **sojourn** (Schrage-Miller; see e.g. Bansal, OR Letters 2004)::

        E[T^SRPT(x)]
            = [lam * int_0^x t^2 f(t) dt + lam * x^2 (1-F(x))] / [2(1-rho_x)^2]
              + int_0^x dt / (1 - rho_t)

    (equivalently ``lam * int_0^x t * (1-F(t)) dt / (1-rho_x)^2 + int_0^x ...``).

    Unconditional::

        E[T^SRPT] = int_0^inf f(x) * E[T^SRPT(x)] dx
        E[W^SRPT] = E[T^SRPT] - b[0]

    Inner pieces use precomputed grids on ``_SizeBasedCalcBase``.
    """

    def conditional_mean_response(self, x: float) -> float:
        """Conditional mean sojourn time E[T^SRPT(x)] for a job of size x."""
        if x <= 0:
            return 0.0
        rho_x = self._rho_interp(x)
        denom = 1.0 - rho_x
        if denom <= 1e-10:
            raise ValueError(f"load >= 1 at x={x}: integral diverges")

        int_t2f = self._t2f_interp(x)
        tail = max(0.0, 1.0 - self._cdf_interp(x))
        # Both service-moment and tail terms carry a factor lambda (not x^2 alone).
        first_term = (self.l * int_t2f + self.l * x * x * tail) / (2.0 * denom * denom)
        second_term = self._second_int_interp(x)
        return first_term + second_term

    def run(self) -> QueueResults:
        """Compute E[T^SRPT] and E[W^SRPT] averaged over the job-size distribution."""
        start = self._measure_time()
        self._check_if_servers_and_sources_set()
        utilization = self._check_stability()
        self._build_grids()

        et = self._integrate_pdf_times_conditional(self.conditional_mean_response)
        ew = et - self.b[0]

        self.v = [et]
        self.w = [ew]
        result = QueueResults(v=self.v, w=self.w, p=None, utilization=utilization)
        self._set_duration(result, start)
        return result
