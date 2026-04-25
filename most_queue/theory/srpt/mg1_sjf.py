"""
M/G/1 SJF calculator (non-preemptive shortest job first).
"""

from __future__ import annotations

from scipy.integrate import quad

from most_queue.structs import QueueResults
from most_queue.theory.srpt._base import _SizeBasedCalcBase


class MG1SjfCalc(_SizeBasedCalcBase):
    """
    Numeric calculator for M/G/1 SJF (non-preemptive size priority).

    Formula (Conway-Maxwell-Miller continuous-priority variant)::

        E[W^SJF(x)] = lam * E[S^2] / [2 * (1 - rho_x)^2]
        E[W^SJF]    = int_0^inf f(x) * E[W^SJF(x)] dx
        E[T^SJF]    = E[W^SJF] + b[0]
    """

    def conditional_mean_wait(self, x: float) -> float:
        """Conditional mean wait E[W^SJF(x)] for a job of size x."""
        if x <= 0:
            return 0.0
        rho_x = self._rho_interp(x)
        denom = 1.0 - rho_x
        if denom <= 1e-10:
            raise ValueError(f"load >= 1 at x={x}: integral diverges")
        return (self.l * self.b[1]) / (2.0 * denom * denom)

    def run(self) -> QueueResults:
        """Compute E[W^SJF] and E[T^SJF] averaged over the job-size distribution."""
        start = self._measure_time()
        self._check_if_servers_and_sources_set()
        utilization = self._check_stability()
        self._build_grids()

        ew, _ = quad(
            lambda x: self.pdf_fn(x) * self.conditional_mean_wait(float(x)),
            0.0,
            self.x_max,
            limit=300,
        )
        et = ew + self.b[0]

        self.w = [ew]
        self.v = [et]
        result = QueueResults(v=self.v, w=self.w, p=None, utilization=utilization)
        self._set_duration(result, start)
        return result
