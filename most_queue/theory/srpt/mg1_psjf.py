"""
M/G/1 PSJF calculator (preemptive shortest job first).
"""

from __future__ import annotations

from most_queue.structs import QueueResults
from most_queue.theory.srpt._base import _SizeBasedCalcBase


class MG1PsjfCalc(_SizeBasedCalcBase):
    """
    Numeric calculator for M/G/1 PSJF.

    Formula::

        E[T^PSJF(x)] = lam * int_0^x t^2 f(t) dt / [2(1 - rho_x)^2]  +  x / (1 - rho_x)
        E[T^PSJF]    = int_0^inf f(x) * E[T^PSJF(x)] dx
        E[W^PSJF]    = E[T^PSJF] - b[0]

    Unlike SRPT, the priority rank is the *original* job size (not remaining
    work), so there is no int_0^x dt/(1-rho_t) term.
    """

    def conditional_mean_response(self, x: float) -> float:
        """Conditional mean sojourn time E[T^PSJF(x)] for a job of size x."""
        if x <= 0:
            return 0.0
        rho_x = self._rho_interp(x)
        denom = 1.0 - rho_x
        if denom <= 1e-10:
            raise ValueError(f"load >= 1 at x={x}: integral diverges")

        int_t2f = self._t2f_interp(x)
        first_term = (self.l * int_t2f) / (2.0 * denom * denom)
        second_term = x / denom
        return first_term + second_term

    def run(self) -> QueueResults:
        """Compute E[T^PSJF] and E[W^PSJF] averaged over the job-size distribution."""
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
