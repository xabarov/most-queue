"""
M/G/1 polling systems: a single server cyclically visits Q queues, incurring a
switchover time between queues.

Because switchovers are non-productive, polling is not work-conserving, but a
**pseudo-conservation law** (Boxma & Groenevelt) fixes the load-weighted sum of
mean waiting times exactly:

  sum_i rho_i W_i = rho/(1-rho) * sum_i lambda_i E[S_i^2]/2
                  + rho * E[R^2] / (2 E[R])
                  + E[R] / (2(1-rho)) * (rho^2 - sum_i rho_i^2)
                  + Y_discipline

where R is the total switchover time per cycle, rho = sum_i rho_i, and the
discipline term is Y = 0 for exhaustive service and
Y = E[R]/(1-rho) * sum_i rho_i^2 for gated service.

For a **symmetric** system (identical queues) all W_i are equal, so the exact
per-queue mean wait is W = (pseudo-conservation sum) / rho. General asymmetric
per-queue waits are obtained from the paired `PollingSim`.
"""

import time
from dataclasses import dataclass, field

from most_queue.theory.base_queue import BaseQueue


@dataclass
class PollingResults:
    """Polling results."""

    pseudo_conservation_sum: float  # sum_i rho_i W_i (exact, any system)
    mean_wait_symmetric: float | None = None  # per-queue mean wait if the system is symmetric
    mean_cycle: float = 0.0  # E[C] = E[R] / (1 - rho)
    utilization: float = 0.0
    per_queue_wait: list = field(default_factory=list)
    duration: float = 0.0


class PollingCalc(BaseQueue):
    """
    M/G/1 cyclic polling with switchover times.

    :param discipline: "exhaustive" (serve a queue until empty) or "gated" (serve
        only the jobs present at the polling instant).
    """

    def __init__(self, discipline: str = "exhaustive"):
        super().__init__(n=1)
        self.discipline = discipline.lower()
        if self.discipline not in ("exhaustive", "gated"):
            raise ValueError("discipline must be 'exhaustive' or 'gated'")
        self.lams = None
        self.b = None  # per-queue service raw moments
        self.r1 = None  # per-queue mean switchover
        self.r_cv2 = 0.0  # squared CV of each switchover (i.i.d. per queue)

    def set_sources(self, class_arrival_rates: list):  # pylint: disable=arguments-differ
        """:param class_arrival_rates: arrival rate lambda_i for each queue."""
        self.lams = list(class_arrival_rates)
        self.is_sources_set = True

    def set_servers(self, b: list):  # pylint: disable=arguments-differ
        """:param b: per-queue service raw moments; either one list [E[S], E[S^2], ...]
        shared by all queues, or a list of such lists, one per queue."""
        if b and not hasattr(b[0], "__len__"):
            b = [list(b)] * len(self.lams)
        self.b = [list(x) for x in b]
        self.is_servers_set = True

    def set_switchover(self, mean: float, cv2: float = 0.0):
        """:param mean: mean switchover time between two queues (same for all).
        :param cv2: squared coefficient of variation of a switchover (0 = deterministic)."""
        self.r1 = mean
        self.r_cv2 = cv2

    def run(self) -> PollingResults:
        """Compute the pseudo-conservation sum and (if symmetric) per-queue wait."""
        self._check_if_servers_and_sources_set()
        if self.r1 is None:
            raise RuntimeError("call set_switchover() before run()")
        start = time.process_time()

        Q = len(self.lams)
        rho_i = [self.lams[i] * self.b[i][0] for i in range(Q)]
        rho = sum(rho_i)
        if rho >= 1:
            raise ValueError(f"unstable: rho = {rho:.3f} >= 1")

        s = Q * self.r1  # mean total switchover per cycle
        var_r_total = Q * (self.r1**2 * self.r_cv2)  # i.i.d. switchovers
        e_r2_total = var_r_total + s**2  # E[R_total^2]

        term1 = rho / (1 - rho) * sum(self.lams[i] * self.b[i][1] / 2 for i in range(Q))
        term2 = rho * e_r2_total / (2 * s)
        term3 = s / (2 * (1 - rho)) * (rho**2 - sum(r * r for r in rho_i))
        y = 0.0 if self.discipline == "exhaustive" else s / (1 - rho) * sum(r * r for r in rho_i)
        pcl = term1 + term2 + term3 + y

        # symmetric system -> all W_i equal
        symmetric = len({(round(self.lams[i], 12), tuple(round(x, 12) for x in self.b[i])) for i in range(Q)}) == 1
        w_sym = pcl / rho if symmetric else None
        per_queue = [w_sym] * Q if w_sym is not None else []

        res = PollingResults(
            pseudo_conservation_sum=pcl,
            mean_wait_symmetric=w_sym,
            mean_cycle=s / (1 - rho),
            utilization=rho,
            per_queue_wait=per_queue,
        )
        res.duration = time.process_time() - start
        return res
