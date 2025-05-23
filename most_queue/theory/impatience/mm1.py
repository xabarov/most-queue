"""
Calc M/M/1 queue with exponential impatience.
"""

import math


class MM1Impatience:
    """
    Calc M/M/1 queue with exponential impatience.
    """

    def __init__(self, l: float, mu: float, gamma: float,
                 tol: float = 1e-12, probs_max_num: int = 100000):
        """
        Initialization of the MM1Impatience class.
        :param l: Arrival rate
        :param mu: Service rate
        :param gamma: Impatience rate
        :param tol: Tolerance for convergence in iterative calculations. Default is 1e-12.
        :param probs_max_num: Maximum number of probabilities to calculate. Default is 100000.
        """
        self.l = l
        self.mu = mu
        self.gamma = gamma
        self.tol = tol
        self.probs_max_num = probs_max_num

        self.probs = self._calc_p()

    def get_p(self) -> list[float]:
        """
        Get the probabilities of states.
        :return: List of probabilities.
        """
        return self.probs

    def get_N(self):
        """
        Get average number of jobs in the system.
        """
        N = 0
        for i, p in enumerate(self.probs):
            N += i * p

        return N

    def get_Q(self):
        """
        Get average number of jobs in the queue.
        """
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
        p0 = self._calc_p0()
        ps = [p0]

        for i in range(1, self.probs_max_num):
            chisl = math.pow(self.l, i)
            znam = self.mu
            j = 1
            while j < i:
                znam *= (self.mu + j * self.gamma)
                j += 1

            pi = p0 * chisl / znam
            ps.append(pi)
            if pi < self.tol:
                break
        self.p = ps
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
                znam *= (self.mu + j * self.gamma)
                j += 1

            elem_old = elem_new
            elem_new = chisl / znam
            summ += elem_new

            i += 1

        return 1.0 / (1.0 + summ)
