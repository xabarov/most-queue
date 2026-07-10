"""
Analytical Age of Information (AoI) calculators for single-server queues.

Two exact results are provided as closed forms:

* **M/M/1 FCFS** — time-average AoI Δ̄ = (1/μ)(1 + 1/ρ + ρ²/(1−ρ)) and
  peak AoI (PAoI) = E[T] + 1/λ.
* **preemptive-LCFS M/M/1** — time-average AoI Δ̄ = (1/μ)(1 + 1/ρ); stable for
  any ρ (a fresh update always preempts a stale one).

For a general single-server **FCFS** queue the PAoI equals E[T] + 1/λ (the mean
sojourn plus the mean interarrival) — exact whenever every update is delivered in
order. The average AoI of a general M/G/1 is not a function of the service
moments alone (it needs the full service distribution); use the discrete-event
`most_queue.sim.aoi.AoISim` for those cases.
"""

import time

from most_queue.structs import AoIResults
from most_queue.theory.base_queue import BaseQueue


class AoICalc(BaseQueue):
    """
    AoI for a single-server FCFS queue. Exponential service gives both the exact
    average AoI and PAoI (M/M/1); a general service (given by raw moments) gives
    the exact PAoI = E[T] + 1/λ via the Pollaczek–Khinchine mean sojourn, while
    the average AoI is left to simulation (not moment-closed for M/G/1).
    """

    def __init__(self):
        super().__init__(n=1)
        self.l = None
        self.mu = None
        self.b = None  # service raw moments (for M/G/1)

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival (update generation) rate."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float | None = None, b: list[float] | None = None):  # pylint: disable=arguments-differ
        """
        :param mu: exponential service rate (M/M/1), or
        :param b: service-time raw moments [E[S], E[S^2], ...] (M/G/1).
        """
        if mu is None and b is None:
            raise ValueError("provide either mu (M/M/1) or b (M/G/1 service moments)")
        self.mu = mu
        self.b = b
        self.is_servers_set = True

    def _mean_sojourn(self):
        if self.b is not None:
            es, es2 = self.b[0], self.b[1]
            ro = self.l * es
            if ro >= 1:
                raise ValueError(f"unstable: rho={ro:.3f} >= 1")
            return es + self.l * es2 / (2.0 * (1.0 - ro))  # Pollaczek–Khinchine
        ro = self.l / self.mu
        if ro >= 1:
            raise ValueError(f"unstable: rho={ro:.3f} >= 1")
        return 1.0 / (self.mu - self.l)

    def run(self) -> AoIResults:
        """Compute AoI. Returns average AoI (M/M/1 only) and PAoI (always)."""
        self._check_if_servers_and_sources_set()
        start = time.process_time()

        e_t = self._mean_sojourn()
        peak = e_t + 1.0 / self.l

        avg = None
        if self.mu is not None:  # exponential service -> exact M/M/1 average AoI
            ro = self.l / self.mu
            avg = (1.0 / self.mu) * (1.0 + 1.0 / ro + ro * ro / (1.0 - ro))

        res = AoIResults(avg_aoi=avg, peak_aoi=peak)
        res.duration = time.process_time() - start
        return res


class LcfsPreemptiveAoICalc(BaseQueue):
    """
    Time-average AoI of a preemptive-LCFS M/M/1 queue: Δ̄ = (1/μ)(1 + 1/ρ).
    A fresh update always preempts (and discards) the stale one in service, so the
    system is stable for any load and delivers minimal age among M/M/1 policies.
    """

    def __init__(self):
        super().__init__(n=1)
        self.l = None
        self.mu = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival (update generation) rate."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """:param mu: exponential service rate."""
        self.mu = mu
        self.is_servers_set = True

    def run(self) -> AoIResults:
        """Compute the average AoI (stable for any rho)."""
        self._check_if_servers_and_sources_set()
        start = time.process_time()
        ro = self.l / self.mu
        avg = (1.0 / self.mu) * (1.0 + 1.0 / ro)
        res = AoIResults(avg_aoi=avg, peak_aoi=None)
        res.duration = time.process_time() - start
        return res
