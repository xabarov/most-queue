"""
M/G/1 accumulating priority queue (linear APQ).

Each class k accumulates priority linearly while waiting: a customer that
arrived at time tau has instantaneous priority rate_k * (t - tau); at every
service completion the waiting customer with the largest accumulated priority
is served (non-preemptive). This is Kleinrock's delay-dependent priority
discipline (Kleinrock, Naval Research Logistics Quarterly, 1964), whose
modern treatment — including waiting-time distributions — is the accumulating
priority queue of Stanford, Taylor & Ziedins (Queueing Systems, 2013).

Mean waiting times are exact via Kleinrock's recursion. Order classes by
accumulation rate b_(1) <= ... <= b_(N) and compute upward:

    W_k = [W0/(1-rho) - sum_{i<k} rho_i W_i (1 - b_i/b_k)]
          / [1 - sum_{i>k} rho_i (1 - b_k/b_i)]

where W0 = sum_i lambda_i E[S_i^2]/2. Limits: equal rates give FIFO
(W = W0/(1-rho)); extreme rate ratios give the classic non-preemptive
priority (Cobham) waits. The load-weighted conservation law
sum_k rho_k W_k = rho W0/(1-rho) holds for any rates.

Class 0 is the highest-priority class (library convention), so accumulation
rates must be passed in non-increasing order.
"""

import time

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.utils.conv import conv_moments


class MG1AccumulatingPriorityCalc(BaseQueue):
    """
    Exact mean waiting/sojourn times for the M/G/1 linear accumulating
    priority queue (non-preemptive).
    """

    def __init__(self):
        super().__init__(n=1)
        self.l = None
        self.b = None
        self.rates = None

    def set_sources(self, l: list[float]):  # pylint: disable=arguments-differ
        """
        :param l: arrival rate of each class (class 0 — highest priority).
        """
        self.l = [float(x) for x in l]
        self.is_sources_set = True

    def set_servers(self, b: list[list[float]], rates: list[float]):  # pylint: disable=arguments-differ
        """
        :param b: raw service moments per class, b[k] = [E[S], E[S^2], ...].
        :param rates: priority accumulation rate of each class, aligned with
            `l` (class 0 — highest priority, i.e. the largest rate).
        """
        if len(rates) != len(b):
            raise ValueError("rates must have one entry per class")
        if any(r <= 0 for r in rates):
            raise ValueError("accumulation rates must be positive")
        if any(rates[i] < rates[i + 1] for i in range(len(rates) - 1)):
            raise ValueError("class 0 is the highest priority: rates must be non-increasing")
        self.b = [list(x) for x in b]
        self.rates = [float(r) for r in rates]
        self.is_servers_set = True

    def run(self) -> PriorityResults:
        """
        Kleinrock's recursion for the mean waits, lowest accumulation rate
        first.
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        n_classes = len(self.l)
        rho_k = [self.l[k] * self.b[k][0] for k in range(n_classes)]
        rho = sum(rho_k)
        if rho >= 1.0:
            raise ValueError(f"Unstable: total load {rho:.4f} >= 1")
        w0 = sum(self.l[k] * self.b[k][1] / 2.0 for k in range(n_classes))

        # Recursion runs from the lowest accumulation rate upward
        order = sorted(range(n_classes), key=lambda k: self.rates[k])
        w_sorted: dict[int, float] = {}
        for pos, k in enumerate(order):
            b_k = self.rates[k]
            num = w0 / (1.0 - rho)
            for prev_pos in range(pos):
                i = order[prev_pos]
                num -= rho_k[i] * w_sorted[i] * (1.0 - self.rates[i] / b_k)
            den = 1.0
            for nxt_pos in range(pos + 1, n_classes):
                i = order[nxt_pos]
                den -= rho_k[i] * (1.0 - b_k / self.rates[i])
            w_sorted[k] = num / den

        w = [[w_sorted[k]] for k in range(n_classes)]
        v = [conv_moments(self.b[k], [w_sorted[k]], 1) for k in range(n_classes)]

        self.results = PriorityResults(
            w=w,
            v=[[float(x) for x in vk] for vk in v],
            utilization=float(rho),
            duration=time.process_time() - start,
        )
        return self.results
