"""
Calculation of M/M/1 QS with batch arrival
"""

from most_queue.structs import QueueResults
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
        self.l_moments = None
        self.ro_tilda = None

    def set_sources(self, l: float, batch_probs: list[float]):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        self.lam = l
        self.ls = list(batch_probs)
        self.l_moments = self._calc_moments_of_job_in_batch()

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
        Run calculation.

        Returns:
            QueueResults with calculated values.
        """
        start = self._measure_time()

        v = [self.get_v1()]
        w = [self.get_w1()]
        p = self.get_p()

        result = QueueResults(utilization=self.ro, v=v, w=w, p=p)
        self._set_duration(result, start)
        return result

    def calc_mean_batch_size(self):
        """
        Mean batch size
        """
        return sum(self.ls) / len(self.ls)

    def get_p(self):
        """
        Probs of QS states
        """

        self._check_if_servers_and_sources_set()

        self.ro_tilda = self.ro_tilda or self.lam / self.mu
        self.ro = self.ro or self.l_moments[0] * self.ro_tilda

        p0 = 1.0 - self.ro
        Ls = self._calc_big_ls()
        rs = [1]

        ps = [p0]

        for i in range(1, self.calc_params.p_num):
            summ = 0
            for j in range(i):
                num = i - j - 1
                if num < len(self.ls):
                    summ += rs[j] * Ls[num]

            r_tek = self.ro_tilda * summ
            rs.append(r_tek)

            ps.append(p0 * r_tek)
            if ps[i] < self.calc_params.tolerance:
                break

        return ps

    def get_mean_jobs_in_system(self):
        """
        Mean jobs in QS
        """

        self._check_if_servers_and_sources_set()

        self.ro_tilda = self.ro_tilda or self.lam / self.mu
        self.ro = self.ro or self.l_moments[0] * self.ro_tilda

        return (self.l_moments[0] + self.l_moments[1]) * self.ro_tilda / (2 * (1.0 - self.ro))

    def get_mean_jobs_on_queue(self):
        """
        Mean queue length
        """
        self.mean_jobs_in_system = self.mean_jobs_in_system or self.get_mean_jobs_in_system()
        return self.mean_jobs_in_system - self.ro

    def get_w1(self):
        """
        Mean wait time
        """
        self.mean_jobs_on_queue = self.mean_jobs_on_queue or self.get_mean_jobs_on_queue()
        return self.mean_jobs_on_queue / self.ro_tilda

    def get_v1(self):
        """
        Mean sojourn time
        """
        self.mean_jobs_in_system = self.mean_jobs_in_system or self.get_mean_jobs_in_system()
        return self.mean_jobs_in_system / (self.l_moments[0] * self.lam)

    def _calc_moments_of_job_in_batch(self):
        """
        raw moments of the number of jobs in the batch
        """
        moments = [0.0] * len(self.ls)
        for i, prob in enumerate(self.ls):
            for j, _ in enumerate(moments):
                moments[j] += pow(i + 1, j + 1) * prob
        return moments

    def _calc_big_ls(self) -> list[float]:
        """
        Returns list of probabilities
        of arrival more than i job in a group
        """
        Ls = [1]
        summ = 0
        for prob in self.ls:
            summ += prob
            Ls.append(1.0 - summ)

        return Ls
