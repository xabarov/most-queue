"""
Calculate queue M/M/n/r
"""

import math

from most_queue.rand_distribution import ExpDistribution
from most_queue.theory.base_queue import BaseQueue, QueueResults
from most_queue.theory.utils.conv import conv_moments


class MMnrCalc(BaseQueue):
    """
    Calculate queue M/M/n/r
    """

    def __init__(self, n: int, r: int):
        """
        :param l: arrival intensity
        :param mu: service intensity
        :param n: number of servers
        :param r: number of places in the queue (including servers)
        """

        super().__init__(n=n)

        self.r = r  # number of places in the queue (including servers)

        self.l = None  # arrival intensity
        self.mu = None  # service intensity
        self.w = None
        self.p = None
        self.ro = None  # utilization factor

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
        Run calculation of the queue system.
        """

        p = self.get_p()
        w = self.get_w()
        v = self.get_v()

        utilization = self.l / (self.mu * self.n)

        return QueueResults(p=p, w=w, v=v, utilization=utilization)

    def get_utilization(self) -> float:
        """
        Calculate utilization factor of the system.
        """

        return self.ro

    def get_busy_probability(self) -> float:
        """
        Calculate probability that all servers are busy and there is no free place in the queue
        """

        self.p = self.p or self._calc_p()
        chisl = math.pow(self.ro, self.n + self.r) * self.p[0]
        znam = math.factorial(self.n) * math.pow(self.n, self.r)
        return chisl / znam

    def get_mean_queue_length(self) -> float:
        """
        Calculate mean queue length
        """

        self.p = self.p or self._calc_p()
        summ = 0
        for i in range(1, self.r + 1):
            summ += i * math.pow(self.ro / self.n, i)
        return self.p[self.n] * summ

    def get_w(self, num=3) -> list[float]:
        """
        Calculate initial moments of waiting time in the queue
        """
        self.p = self.p or self._calc_p()
        qs = self._get_qs(q_num=num)
        w = [0] * num
        for k in range(num):
            w[k] = qs[k] / pow(self.l, k + 1)

        self.w = w
        return w

    def get_v(self) -> list[float]:
        """
        Calculate  initial moments of sojourn time in the queue
        """
        if self.w is None:
            self.w = self.get_w()
        b = ExpDistribution.calc_theory_moments(self.mu)
        v = conv_moments(self.w, b)
        return v

    def get_p(self) -> list[float]:
        """
        Get probabilities of states
        :return: list of probabilities of states from 0 to n+r
        """
        self.p = self.p or self._calc_p()
        return self.p

    def _get_qs(self, q_num=3):

        q_s = []
        for k in range(1, q_num + 1):
            summ = 0
            for nn in range(k, self.r + 1):
                summ += (math.factorial(nn) / math.factorial(nn - k)) * self.p[nn + self.n]
            q_s.append(summ)
        return q_s

    def _calc_p(self) -> list[float]:
        """
        Calc probability of states
        :return: list of probabilities of states from 0 to n+r
        """

        self._check_if_servers_and_sources_set()
        self.ro = self.l / self.mu  # utilization factor

        p = [0] * (self.n + self.r + 1)
        summ1 = 0
        for i in range(self.n):
            summ1 += pow(self.ro, i) / math.factorial(i)

        chisl = 1 - pow(self.ro / self.n, self.r + 1)
        coef = pow(self.ro, self.n) / math.factorial(self.n)
        znam = 1 - (self.ro / self.n)

        p[0] = 1.0 / (summ1 + coef * chisl / znam)

        for i in range(self.n):
            p[i] = pow(self.ro, i) * p[0] / math.factorial(i)

        for i in range(self.n, self.n + self.r + 1):
            p[i] = pow(self.ro, i) * p[0] / (math.factorial(self.n) * pow(self.n, i - self.n))

        self.p = p

        return p
