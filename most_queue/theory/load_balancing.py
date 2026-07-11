"""
Mean-field (large-N) analysis of load-balancing / dispatching policies.

For a large pool of N identical servers fed by Poisson arrivals with per-server
utilization rho, the stationary fraction of servers with at least k jobs,
s_k, has a closed form in the N -> infinity mean-field limit:

* **power-of-d** (JSQ(d), sample d servers and join the shortest):
  s_k = rho ** ((d**k - 1) / (d - 1)).  For d = 1 (random dispatch) this is the
  geometric M/M/1 tail s_k = rho**k; for d >= 2 it decays *doubly* exponentially
  in k (the "power of two choices").
* **JSQ** (join the shortest of all) and **JIQ** (join an idle server if any):
  in the mean-field limit below capacity the queue length is at most 1
  (s_1 = rho, s_k = 0 for k >= 2) — asymptotically zero waiting.

The mean number of jobs per server is L = sum_{k>=1} s_k and the mean response
time follows from Little's law with per-server arrival rate rho*mu:
W = L / (rho * mu).
"""

import time
from dataclasses import dataclass, field

from most_queue.theory.base_queue import BaseQueue


@dataclass
class LoadBalancingResults:
    """Mean-field load-balancing results."""

    w: float  # mean response time (sojourn)
    wait: float  # mean waiting time
    mean_number_per_server: float  # L = sum_k s_k
    tail: list = field(default_factory=list)  # s_k = P(server has >= k jobs), k = 0,1,2,...
    duration: float = 0.0


class LoadBalancingMeanField(BaseQueue):
    """
    Mean-field response time of a load-balancing policy over a large server pool.

    :param policy: "power-of-d" (a.k.a. JSQ(d)), "jsq", "jiq", or "random" (= power-of-d with d=1).
    :param d: number of sampled servers for the power-of-d policy (ignored otherwise).
    """

    def __init__(self, policy: str = "power-of-d", d: int = 2):
        super().__init__(n=1)
        self.policy = policy.lower()
        self.d = d
        self.rho = None
        self.mu = None

    def set_sources(self, rho: float):  # pylint: disable=arguments-differ
        """:param rho: per-server offered load (utilization), 0 < rho < 1."""
        if not 0 < rho < 1:
            raise ValueError("rho must be in (0, 1)")
        self.rho = rho
        self.is_sources_set = True

    def set_servers(self, mu: float = 1.0):  # pylint: disable=arguments-differ
        """:param mu: per-server service rate."""
        self.mu = mu
        self.is_servers_set = True

    def _tail(self, kmax: int) -> list:
        rho = self.rho
        if self.policy in ("jsq", "jiq"):
            # mean-field limit below capacity: at most one job per server
            s = [0.0] * (kmax + 1)
            s[0] = 1.0
            s[1] = rho
            return s
        d = 1 if self.policy == "random" else self.d
        s = [1.0]  # s_0
        for k in range(1, kmax + 1):
            expo = k if d == 1 else (d**k - 1) // (d - 1)
            s.append(rho**expo)
        return s

    def run(self) -> LoadBalancingResults:
        """Compute the mean-field tail, mean number per server and response time."""
        self._check_if_servers_and_sources_set()
        start = time.process_time()

        # tail decays fast; sum enough terms (double-exponential for d>=2)
        kmax = 2000 if (self.policy in ("random",) or (self.policy == "power-of-d" and self.d == 1)) else 60
        tail = self._tail(kmax)
        L = sum(tail[1:])  # mean number of jobs at a server
        w = L / (self.rho * self.mu)
        wait = w - 1.0 / self.mu

        res = LoadBalancingResults(
            w=w,
            wait=max(wait, 0.0),
            mean_number_per_server=L,
            tail=tail[: min(len(tail), 12)],
        )
        res.duration = time.process_time() - start
        return res
