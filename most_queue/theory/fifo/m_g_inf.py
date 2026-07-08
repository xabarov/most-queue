"""
Infinite-server queue M/G/inf (covers M/M/inf as a special case).

The stationary number of busy servers is Poisson with mean a = l * b1 and is
insensitive to the service distribution beyond its mean; the transient
distribution is Poisson as well.

References:
    Takacs L. Introduction to the Theory of Queues. Oxford University Press, 1962.
    Sevastyanov B.A. An Ergodic Theorem for Markov Processes and Its Application to
        Telephone Systems with Refusals. Theory of Probability and Its Applications,
        2(1), 1957. doi:10.1137/1102005.
"""

import math

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MGInfCalc(BaseQueue):
    """
    Infinite-server system M/G/inf: every arriving job starts service immediately,
    so waiting time is identically zero and sojourn time equals service time.

    The number of servers is unlimited; the `n` attribute inherited from BaseQueue
    is fixed to 1 and not used.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)

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
        :param b: raw moments of service time distribution (only b[0] affects
            the state probabilities — insensitivity; the rest are returned as
            sojourn time moments)
        """
        if not b or b[0] <= 0:
            raise ValueError("Service time moments must be non-empty with positive mean")
        self.b = list(b)
        self.is_servers_set = True

    def get_offered_load(self) -> float:
        """
        Offered load a = l * b1 — the mean number of busy servers.
        """
        self._check_if_servers_and_sources_set()
        return self.l * self.b[0]

    def get_p(self) -> list[float]:
        """
        Get probabilities of states (number of busy servers): Poisson with mean a.
        """
        a = self.get_offered_load()
        num_probs = self.calc_params.p_num
        p = [0.0] * num_probs
        term = math.exp(-a)
        for k in range(num_probs):
            p[k] = term
            term *= a / (k + 1)
        self.p = p
        return p

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time: identically zero (a server is always free).
        """
        self.w = [0.0] * num
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: equal to service time moments.
        """
        self._check_if_servers_and_sources_set()
        self.v = list(self.b[:num])
        return self.v

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation of the queue system.

        Returns:
            QueueResults; utilization is reported as 0.0 (infinitely many servers),
            the mean number of busy servers is available via get_offered_load().
        """
        start = self._measure_time()
        with self._validate_state():
            num_of_moments = min(num_of_moments, len(self.b))
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.mean_jobs_in_system = self.get_offered_load()
            self.ro = 0.0

        result = QueueResults(p=p, w=w, v=v, utilization=0.0)
        self._set_duration(result, start)
        return result
