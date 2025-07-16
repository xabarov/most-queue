"""
Calculation of M/M/1 QS with batch arrival
"""

from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class BatchMM1(BaseQueue):
    """
    Calculation of M/M/1 QS with batch arrival
    """

    def __init__(self, calc_params: CalcParams | None = None):
        """
        Init parameters for M/M/1 QS with batch arrival
        :param calc_params: calculation parameters
        :type calc_params: CalcParams | None
        """

        super().__init__(n=1, calc_params=calc_params)

        self.lam = None
        self.mu = None
        self.ls = None
        self.p_num = self.calc_params.p_num
        self.tol = self.calc_params.tolerance

        self.l = None
        self.l2 = None
        self.ro_tilda = None
        self.ro = None

    def set_sources(self, l: float, batch_probs: list[float]):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        self.lam = l
        self.ls = list(batch_probs)
        moments = self.calc_l_moments()
        self.l = moments[0]
        self.l2 = moments[1]

        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """
        Set servers
        :param mu: service rate
        """
        self.mu = mu

        self.is_servers_set = True

    def calc_l(self):
        """
        Mean batch size
        """
        return sum(self.ls) / len(self.ls)

    def calc_l_moments(self):
        """
        Initial moments of the number of jobs in the batch
        """
        moments = [0.0] * len(self.ls)
        for i, prob in enumerate(self.ls):
            for j, _ in enumerate(moments):
                moments[j] += pow(i + 1, j + 1) * prob
        return moments

    def calc_L(self):
        """
        Returns list of probabilities
        of arrival more than i applications in a group
        """
        Ls = [1]
        summ = 0
        for prob in self.ls:
            summ += prob
            Ls.append(1.0 - summ)

        return Ls

    def get_p(self):
        """
        Probs of QS states
        """

        self._check_if_servers_and_sources_set()

        self.ro_tilda = self.ro_tilda or self.lam / self.mu
        self.ro = self.ro or self.l * self.ro_tilda

        p0 = 1.0 - self.ro
        Ls = self.calc_L()
        rs = [1]

        ps = [p0]

        for i in range(1, self.p_num):
            summ = 0
            for j in range(i):
                num = i - j - 1
                if num < len(self.ls):
                    summ += rs[j] * Ls[num]

            r_tek = self.ro_tilda * summ
            rs.append(r_tek)

            ps.append(p0 * r_tek)
            if ps[i] < self.tol:
                break

        return ps

    def get_N(self):
        """
        Mean jobs in QS
        """

        self.ro_tilda = self.ro_tilda or self.lam / self.mu
        self.ro = self.ro or self.l * self.ro_tilda

        return (self.l2 + self.l) * self.ro_tilda / (2 * (1.0 - self.ro))

    def get_Q(self):
        """
        Mean queue length
        """
        N = self.get_N()
        return N - self.ro

    def get_w1(self):
        """
        Mean wait time
        """
        return self.get_Q() / self.ro_tilda

    def get_v1(self):
        """
        Mean sojourn time
        """
        return self.get_N() / (self.l * self.lam)
