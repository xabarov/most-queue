"""
M/M/1 queue with working breakdowns (Kalidass & Kasturi, Computers &
Industrial Engineering, 2012).

The server alternates between a normal phase (service rate mu) and a
defective phase (reduced rate mu_d < mu) instead of stopping completely:
breakdowns occur at rate xi (in the normal phase, busy or idle), repairs at
rate eta. Degradation-instead-of-outage is the natural model for clouds and
virtualized services.

Exact solution of the truncated two-phase CTMC on states (k jobs, phase).

Stability: lambda < (mu * eta + mu_d * xi) / (xi + eta) (mean service rate in
the random environment).
"""

import time

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary

NORMAL, DEGRADED = 0, 1


class MM1WorkingBreakdownsCalc(BaseQueue):
    """
    M/M/1 with working breakdowns: reduced service rate during repair.
    """

    def __init__(self):
        super().__init__(n=1)
        self.l = None
        self.mu = None
        self.mu_d = None
        self.xi = None
        self.eta = None
        self.degraded_prob = None
        self.results = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float, mu_d: float, xi: float, eta: float):  # pylint: disable=arguments-differ
        """
        :param mu: normal service rate.
        :param mu_d: degraded service rate during a breakdown (mu_d <= mu).
        :param xi: breakdown rate (normal -> degraded).
        :param eta: repair rate (degraded -> normal).
        """
        if mu_d > mu:
            raise ValueError("mu_d must not exceed mu")
        self.mu = mu
        self.mu_d = mu_d
        self.xi = xi
        self.eta = eta
        self.is_servers_set = True

    def run(self, tail_tol: float = 1e-10, k_start: int = 400, k_max: int = 25600) -> QueueResults:
        """
        Solve the truncated two-phase CTMC.
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        mean_rate = (self.mu * self.eta + self.mu_d * self.xi) / (self.xi + self.eta)
        if self.l >= mean_rate:
            raise ValueError(f"Unstable: lambda={self.l} >= mean service rate {mean_rate:.4f}")

        rates = {NORMAL: self.mu, DEGRADED: self.mu_d}
        k_cap = k_start
        while True:

            def idx(k, ph):
                return 2 * k + ph

            trans = []
            for k in range(k_cap + 1):
                for ph in (NORMAL, DEGRADED):
                    if k < k_cap:
                        trans.append((idx(k, ph), idx(k + 1, ph), self.l))
                    if k > 0 and rates[ph] > 0:
                        trans.append((idx(k, ph), idx(k - 1, ph), rates[ph]))
                trans.append((idx(k, NORMAL), idx(k, DEGRADED), self.xi))
                trans.append((idx(k, DEGRADED), idx(k, NORMAL), self.eta))
            pi = ctmc_stationary(trans, 2 * (k_cap + 1))
            if pi[2 * k_cap :].sum() < tail_tol or k_cap >= k_max:
                break
            k_cap *= 2

        pi2 = pi.reshape(k_cap + 1, 2)
        k_marg = pi2.sum(axis=1)
        mean_jobs = float(np.dot(np.arange(k_cap + 1), k_marg))

        self.degraded_prob = float(pi2[:, DEGRADED].sum())
        busy = float(sum(pi2[k].sum() for k in range(1, k_cap + 1)))

        self.results = QueueResults(
            v=[mean_jobs / self.l],
            p=[float(x) for x in k_marg],
            utilization=busy,
            duration=time.process_time() - start,
        )
        return self.results
