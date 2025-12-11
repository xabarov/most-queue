"""
Calc M/M/1 queue with exponential impatience.
"""

import math

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MM1Impatience(BaseQueue):
    """
    Calc M/M/1 queue with exponential impatience.
    """

    def __init__(
        self,
        gamma: float,
        calc_params: CalcParams | None = None,
    ):
        """
        Initialization of the MM1Impatience class.
        :param gamma: Impatience rate.
        :param calc_params: Calculation parameters.
        """

        super().__init__(n=1, calc_params=calc_params)

        if calc_params is None:
            calc_params = CalcParams()

        self.l = None
        self.mu = None
        self.gamma = gamma
        self.tol = calc_params.tolerance
        self.probs_max_num = calc_params.p_num

        self.probs = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        self.l = l

        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """
        Set servers
        :param mu: service rate
        """
        self.mu = mu

        self.is_servers_set = True

    def run(self) -> QueueResults:
        """
        Run calculation of queueing system.

        Returns:
            QueueResults with calculated values.
        """
        start = self._measure_time()

        p = self.get_p()
        w1 = self.get_w1()
        v1 = self.get_v1()

        # Note: Utilization calculation may need to account for self.gamma
        # (impatience parameter) in future improvements for more accurate results.

        utilization = self.l / self.mu
        result = QueueResults(p=p, v=[v1, 0, 0], w=[w1, 0, 0], utilization=utilization)
        self._set_duration(result, start)
        return result

    def get_p(self) -> list[float]:
        """
        Get the probabilities of states.
        :return: List of probabilities.
        """
        self.probs = self.probs or self._calc_p()
        return self.probs

    def get_N(self):
        """
        Get average number of jobs in the system.
        """
        self.probs = self.probs or self._calc_p()
        N = 0
        for i, p in enumerate(self.probs):
            N += i * p

        return N

    def get_Q(self):
        """
        Get average number of jobs in the queue.
        """

        self.probs = self.probs or self._calc_p()
        Q = 0
        for i, p in enumerate(self.probs):
            if i == 0:
                continue
            Q += (i - 1) * p

        return Q

    def get_w1(self):
        """
        Get average waiting time.
        """
        return self.get_Q() / self.l

    def get_v1(self):
        """
        Get average sojourn time.
        """
        return self.get_N() / self.l

    def _calc_p(self) -> list[float]:
        """
        Probabilities of states in the system
        """

        self._check_if_servers_and_sources_set()

        p0 = self._calc_p0()
        ps = [p0]

        for i in range(1, self.probs_max_num):
            chisl = math.pow(self.l, i)
            znam = self.mu
            j = 1
            while j < i:
                znam *= self.mu + j * self.gamma
                j += 1

            pi = p0 * chisl / znam
            ps.append(pi)
            if pi < self.tol:
                break
        return ps

    def _calc_p0(self) -> float:
        """
        Probability of zero state in the system
        """
        summ = 0
        elem_old = self.l
        elem_new = 0

        i = 1
        while math.fabs(elem_new - elem_old) > self.tol:
            chisl = math.pow(self.l, i)
            znam = self.mu
            j = 1
            while j < i:
                znam *= self.mu + j * self.gamma
                j += 1

            elem_old = elem_new
            elem_new = chisl / znam
            summ += elem_new

            i += 1

        return 1.0 / (1.0 + summ)
