"""
M/G/n with egalitarian Processor Sharing (PS), n >= 1 identical servers.

While at most `n` jobs are present, each gets a dedicated server (rate
`1/b1`); once the count exceeds `n`, the combined capacity `n/b1` is shared
equally among all jobs currently present. Generalizes
``most_queue.theory.fifo.mg1_ps.MG1PSCalc`` (n=1) to n servers -- the same
BCMP/Kelly insensitivity to the service-time distribution shape beyond its
mean applies to a "symmetric" (PS) discipline node, and the state-dependent
("load-dependent") aggregate throughput function `min(n, j)/b1` is the same
device used throughout Jackson/BCMP queueing-network theory to represent a
multi-server station as a single node [Baskett et al. 1975]. The resulting
queue-length distribution is therefore IDENTICAL to the classical M/M/n
(Erlang-C) one -- PS and FCFS differ in how individual jobs experience the
system, not in the aggregate occupancy process.

Only the queue-length distribution and mean sojourn/waiting are exact here.
Job-size-conditional higher sojourn moments are NOT available for n > 1 and are
not a follow-up away: Yashkov's recursion, which `MG1PSCalc` now implements, is
specific to the single-server case -- it is built on the M/G/1-FCFS waiting-time
distribution, and the multi-server PS sojourn time has no such representation.
The insensitivity that makes the queue length here identical to M/M/n does not
extend to the conditional sojourn time, so the n=1 result cannot simply be
rescaled.

References:
    Kleinrock L. Time-shared Systems: A Theoretical Treatment. JACM, 14(2),
        1967. doi:10.1145/321386.321388.
    Baskett F., Chandy K.M., Muntz R.R., Palacios F.G. Open, Closed, and Mixed
        Networks of Queues with Different Classes of Customers. JACM, 22(2),
        1975. doi:10.1145/321879.321887 (load-dependent service centers,
        insensitivity).
    Yashkov S.F. Processor-Sharing Queues: Some Progress in Analysis. Queueing
        Systems, 2, 1987. doi:10.1007/bf01182931 (higher moments; implemented
        for n=1 in most_queue.theory.fifo.mg1_ps, not available for n>1).
"""

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MGnPSCalc(BaseQueue):
    """
    M/G/n egalitarian Processor Sharing: n identical servers, shared equally
    once the number present exceeds n.

    :param n: number of identical servers.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        super().__init__(n=n, calc_params=calc_params)

        self.l = None  # arrival intensity
        self.b = None  # service time raw moments

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set servers
        :param b: raw moments of service time distribution (only b[0] is used —
            PS characteristics are insensitive to the rest)
        """
        if not b or b[0] <= 0:
            raise ValueError("Service time moments must be non-empty with positive mean")
        self.b = list(b)
        self.is_servers_set = True

    def _death_rate(self, j: int) -> float:
        """Aggregate completion rate at occupancy j: min(n, j) / b1 (load-dependent throughput)."""
        return min(self.n, j) / self.b[0]

    def _utilization(self) -> float:
        self._check_if_servers_and_sources_set()
        ro = self.l * self.b[0] / self.n
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")
        return ro

    def get_p(self) -> list[float]:
        """
        Exact queue-length distribution: a state-dependent birth-death product
        form (unbounded -- PS never blocks admission), truncated at
        ``calc_params.p_num`` for normalization. Identical to the classical
        M/M/n (Erlang-C) queue-length distribution.
        """
        self._utilization()  # validates stability
        num_probs = self.calc_params.p_num
        log_p = np.zeros(num_probs)
        log_ratio = 0.0
        for j in range(1, num_probs):
            log_ratio += np.log(self.l) - np.log(self._death_rate(j))
            log_p[j] = log_ratio
        p = np.exp(log_p - log_p.max())
        p /= p.sum()
        self.p = p.tolist()
        return self.p

    def get_n_mean(self) -> float:
        """Exact mean number in system, direct summation over the queue-length distribution."""
        p = self.get_p()
        return float(sum(j * pj for j, pj in enumerate(p)))

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """Mean sojourn time (first moment only), via Little's law: E[V] = E[N] / lambda."""
        self.v = [self.get_n_mean() / self.l]
        return self.v

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """Mean sharing delay (first moment only): E[V] - b1."""
        v = self.get_v(num)
        self.w = [v[0] - self.b[0]]
        return self.w

    def run(self, num_of_moments: int = 1) -> QueueResults:
        """
        Run calculation. Only first moments of v and w are produced.
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
