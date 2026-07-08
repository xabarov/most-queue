"""
M/G/1 FB (Foreground-Background, also known as LAS -- Least Attained Service)
calculator.

FB is the canonical *blind* size-based discipline: the server always works on
the job(s) with the least attained service, with no knowledge of job sizes.
Jobs sharing the minimal attained service are served in processor-sharing mode.

References:
    Nuyens M., Wierman A. The Foreground-Background queue: A survey.
        Performance Evaluation, 65, 2008. doi:10.1016/j.peva.2007.06.028.
    Harchol-Balter M. Performance Modeling and Design of Computer Systems.
        Cambridge University Press, 2013 (Ch. 33).
"""

from __future__ import annotations

from most_queue.structs import QueueResults
from most_queue.theory.srpt._base import _SizeBasedCalcBase


class MG1FbCalc(_SizeBasedCalcBase):
    """
    Numeric calculator for M/G/1 FB (LAS).

    Formula::

        E[T^FB(x)] = lam * E[min(X, x)^2] / [2 (1 - rho_bar_x)^2] + x / (1 - rho_bar_x)

    where rho_bar_x = lam * E[min(X, x)] is the load of jobs truncated at x:

        E[min(X, x)]   = int_0^x t f(t) dt + x (1 - F(x))
        E[min(X, x)^2] = int_0^x t^2 f(t) dt + x^2 (1 - F(x))

    Unlike PSJF (which uses only the load of jobs *smaller* than x), FB
    "sees" every job up to x units of its work, hence the truncated moments.
    """

    def conditional_mean_response(self, x: float) -> float:
        """Conditional mean sojourn time E[T^FB(x)] for a job of size x."""
        if x <= 0:
            return 0.0
        tail = 1.0 - self._cdf_interp(x)
        rho_bar = self._rho_interp(x) + self.l * x * tail
        denom = 1.0 - rho_bar
        if denom <= 1e-10:
            raise ValueError(f"truncated load >= 1 at x={x}: integral diverges")

        min2 = self._t2f_interp(x) + x * x * tail
        first_term = (self.l * min2) / (2.0 * denom * denom)
        second_term = x / denom
        return first_term + second_term

    def run(self) -> QueueResults:
        """Compute E[T^FB] and E[W^FB] averaged over the job-size distribution."""
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
