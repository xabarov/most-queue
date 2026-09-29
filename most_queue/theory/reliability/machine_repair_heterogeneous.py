"""
Machine repair problem with warm spares and TWO heterogeneous repairmen
(repair rates eta_a != eta_b) -- an extension of MachineRepairCalc.

With a single repair rate, the number of failed units j alone is a Markov
state (birth-death). With two different rates that is no longer true: at
j=1 the repair progresses at a different speed depending on *which*
repairman is engaged, so the chain needs to know that too. This is the
finite-source analogue of the classic M/M/2-heterogeneous-servers technique
(Krishnamoorthi, Operations Research, 1963, doi:10.1287/opre.11.3.321),
applied here to the machine-repair-with-warm-spares model (matching
Optimization and Engineering, 2012, doi:10.1007/s11081-012-9195-1).

Fastest-available-first dispatch: repairman A (eta_a >= eta_b after
normalisation) is engaged first whenever it is idle. States:
    j=0            : both idle                        -- 1 state
    j=1            : "A busy" or "B busy"              -- 2 states
    j=2..n_top     : both busy (queue if j > 2)        -- 1 state each
"A busy" (j=1) is reached only from j=0 (a fresh failure, dispatched to the
fast repairman); "B busy" (j=1) is reached only from j=2 when A finishes
first. At j=2 with either death, control passes to the other single-busy
substate; at j>=3 both deaths return to the same (j-1) both-busy state,
since the queue immediately re-engages whichever repairman frees up.

Reduces exactly to MachineRepairCalc(n_repairmen=2, ...) when eta_a=eta_b.
"""

import time

import numpy as np

from most_queue.theory.reliability.machine_repair import MachineRepairResults
from most_queue.theory.reliability.utils import ctmc_stationary


class MachineRepairHeterogeneousCalc:
    """
    Finite-source machine repair problem with exactly two heterogeneous
    repairmen and S warm spares.

    :param n_machines: M -- machines that should be operating.
    :param n_spares: S -- warm spares.
    """

    def __init__(self, n_machines: int, n_spares: int = 0):
        self.m = int(n_machines)
        self.s = int(n_spares)
        if self.m < 1 or self.s < 0:
            raise ValueError("Need n_machines >= 1, n_spares >= 0")
        self.xi = None
        self.xi_s = None
        self.eta_a = None
        self.eta_b = None
        self.is_sources_set = False
        self.results = None

    def set_sources(self, xi: float, eta_a: float, eta_b: float, xi_s: float | None = None):
        """
        :param xi: failure rate of an operating machine.
        :param eta_a: repair rate of one repairman.
        :param eta_b: repair rate of the other repairman (any order -- the
            faster one is dispatched first regardless of which argument it
            came in as).
        :param xi_s: failure rate of a warm spare (default 0 -- cold standby).
        """
        self.xi = xi
        self.eta_a, self.eta_b = max(eta_a, eta_b), min(eta_a, eta_b)
        self.xi_s = 0.0 if xi_s is None else xi_s
        self.is_sources_set = True

    def _birth(self, j: int) -> float:
        operating = min(self.m, self.m + self.s - j)
        spares = max(0, self.s - j)
        return self.xi * operating + self.xi_s * spares

    def _idx_j1a(self) -> int:
        return 1

    def _idx_j1b(self) -> int:
        return 2

    def _idx(self, j: int) -> int:
        """State index for j >= 2 (both repairmen busy)."""
        return j + 1

    def run(self) -> MachineRepairResults:
        """Solve the exact (non-birth-death) finite CTMC."""
        start = time.process_time()
        if not self.is_sources_set:
            raise ValueError("Sources are not set. Please use set_sources() method.")

        n_top = self.m + self.s
        n_states = n_top + 2  # j=0, j=1 (A/B split), j=2..n_top
        eta_a, eta_b = self.eta_a, self.eta_b

        transitions = []

        def add(src, dst, rate):
            if rate > 0:
                transitions.append((src, dst, rate))

        # j = 0
        add(0, self._idx_j1a(), self._birth(0))

        # j = 1, split by which repairman is engaged
        add(self._idx_j1a(), self._idx(2) if n_top >= 2 else self._idx_j1a(), self._birth(1))
        add(self._idx_j1a(), 0, eta_a)
        add(self._idx_j1b(), self._idx(2) if n_top >= 2 else self._idx_j1b(), self._birth(1))
        add(self._idx_j1b(), 0, eta_b)

        # j = 2: both busy, deaths return to the *other* single-busy substate
        if n_top >= 2:
            if n_top >= 3:
                add(self._idx(2), self._idx(3), self._birth(2))
            add(self._idx(2), self._idx_j1b(), eta_a)  # A finished first, B continues
            add(self._idx(2), self._idx_j1a(), eta_b)  # B finished first, A continues

        # j = 3 .. n_top: both busy throughout, queue immediately re-engages
        for j in range(3, n_top + 1):
            if j < n_top:
                add(self._idx(j), self._idx(j + 1), self._birth(j))
            add(self._idx(j), self._idx(j - 1), eta_a + eta_b)

        pi = ctmc_stationary(transitions, n_states)

        # Marginal distribution over j = 0..n_top (merge the j=1 split).
        p = np.zeros(n_top + 1)
        p[0] = pi[0]
        p[1] = pi[self._idx_j1a()] + pi[self._idx_j1b()]
        for j in range(2, n_top + 1):
            p[j] = pi[self._idx(j)]

        js = np.arange(n_top + 1)
        operating = np.minimum(self.m, self.m + self.s - js)

        utilization_a = float(pi[self._idx_j1a()] + sum(pi[self._idx(j)] for j in range(2, n_top + 1)))
        utilization_b = float(pi[self._idx_j1b()] + sum(pi[self._idx(j)] for j in range(2, n_top + 1)))

        self.results = MachineRepairResults(
            availability=float(p[: self.s + 1].sum()),
            mean_failed=float(np.dot(js, p)),
            mean_operating=float(np.dot(operating, p)),
            repairmen_utilization=(utilization_a + utilization_b) / 2.0,
            failure_throughput=float(sum(self._birth(j) * p[j] for j in range(n_top + 1))),
            p=[float(x) for x in p],
            utilization_a=utilization_a,
            utilization_b=utilization_b,
            duration=time.process_time() - start,
        )
        return self.results
