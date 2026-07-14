"""
Machine repair problem (machine interference) with warm spares — the closed
reliability classic (Palm, 1947; Benson & Cox, 1951; survey — e.g. Jain et
al., Computers & OR, 2013).

M identical machines must be operating; S warm spares stand by. An operating
machine fails at rate xi, a warm spare at rate xi_s <= xi; failed units queue
for one of R repairmen (repair rate eta each). A repaired unit becomes a
spare (or goes straight to operation if fewer than M are running).

The number of failed units j is a finite birth-death process:
    birth  b(j) = xi * min(M, M + S - j) + xi_s * max(0, S - j)
    death  d(j) = eta * min(j, R)
solved exactly by the product form of birth-death chains.
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MachineRepairResults:
    """Machine repair problem results."""

    availability: float = 0.0  # P(all M machines are operating)
    mean_failed: float = 0.0  # E[j]
    mean_operating: float = 0.0  # E[number of operating machines]
    repairmen_utilization: float = 0.0  # E[min(j, R)] / R
    failure_throughput: float = 0.0  # long-run failures per unit time
    p: list = field(default_factory=list)  # distribution of failed units
    duration: float = 0.0


class MachineRepairCalc:
    """
    Finite-source machine repair problem with R repairmen and S warm spares.

    :param n_machines: M — machines that should be operating.
    :param n_repairmen: R — repair crew size.
    :param n_spares: S — warm spares.
    """

    def __init__(self, n_machines: int, n_repairmen: int = 1, n_spares: int = 0):
        self.m = int(n_machines)
        self.r = int(n_repairmen)
        self.s = int(n_spares)
        if min(self.m, self.r) < 1 or self.s < 0:
            raise ValueError("Need n_machines >= 1, n_repairmen >= 1, n_spares >= 0")
        self.xi = None
        self.eta = None
        self.xi_s = None
        self.is_sources_set = False
        self.results = None

    def set_sources(self, xi: float, eta: float, xi_s: float | None = None):
        """
        :param xi: failure rate of an operating machine.
        :param eta: repair rate of one repairman.
        :param xi_s: failure rate of a warm spare (default 0 — cold standby).
        """
        self.xi = xi
        self.eta = eta
        self.xi_s = 0.0 if xi_s is None else xi_s
        self.is_sources_set = True

    def _birth(self, j: int) -> float:
        operating = min(self.m, self.m + self.s - j)
        spares = max(0, self.s - j)
        return self.xi * operating + self.xi_s * spares

    def _death(self, j: int) -> float:
        return self.eta * min(j, self.r)

    def run(self) -> MachineRepairResults:
        """
        Solve the finite birth-death chain exactly.
        """
        start = time.process_time()
        if not self.is_sources_set:
            raise ValueError("Sources are not set. Please use set_sources() method.")

        n_top = self.m + self.s
        weights = np.zeros(n_top + 1)
        weights[0] = 1.0
        for j in range(1, n_top + 1):
            weights[j] = weights[j - 1] * self._birth(j - 1) / self._death(j)
        p = weights / weights.sum()

        js = np.arange(n_top + 1)
        operating = np.minimum(self.m, self.m + self.s - js)
        self.results = MachineRepairResults(
            availability=float(p[: self.s + 1].sum()),
            mean_failed=float(np.dot(js, p)),
            mean_operating=float(np.dot(operating, p)),
            repairmen_utilization=float(np.dot(np.minimum(js, self.r), p)) / self.r,
            failure_throughput=float(sum(self._birth(j) * p[j] for j in range(n_top + 1))),
            p=[float(x) for x in p],
            duration=time.process_time() - start,
        )
        return self.results
