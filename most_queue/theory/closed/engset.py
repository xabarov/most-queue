"""
Calculation of the Engset model for M/M/1 with a finite number of sources.
"""

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.diff5dots import diff5dots


class Engset(BaseQueue):
    """
    Calculation of the Engset model for M/M/1 with a finite number of sources.
    """

    def __init__(self):
        """
        lam - arrival rate of requests from each source
        mu - service rate
        m - number of request sources
        """

        super().__init__(n=1)
        self.lam = None
        self.mu = None
        self.m_i = None
        self.m = None
        self.p0 = None

    def set_sources(self, l: float, number_of_sources: int):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        :param number_of_sources: number of sources
        """
        self.lam = l
        self.m = number_of_sources
        self._calc_m_i()
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
        Run calculations for Engset model.

        Returns:
            QueueResults with calculated values.
        """
        start = self._measure_time()

        v = self.get_v()
        w = self.get_w()
        p = self.get_p()

        result = QueueResults(v=v, w=w, p=p, utilization=self.ro)
        self._set_duration(result, start)
        return result

    def get_p(self) -> list[float]:
        """
        Get probabilities of states of the system
        """

        self._check_if_servers_and_sources_set()

        self.ro = self.ro or self.lam / self.mu

        summ = 0
        for i, mm in enumerate(self.m_i):
            summ += mm * pow(self.ro, i)

        ps = []
        ps.append(1.0 / summ)

        for i in range(1, self.m + 1):
            ps.append(ps[0] * self.m_i[i] * pow(self.ro, i))

        self.p = ps
        self.p0 = ps[0]
        return ps

    def get_mean_jobs_in_system(self) -> float:
        """
        Get average number of jobs in the system
        """
        self.p = self.p or self.get_p()

        mean_jobs_in_system = 0
        for i, mm in enumerate(self.m_i):
            mean_jobs_in_system += i * mm * pow(self.ro, i)
        mean_jobs_in_system *= self.p0

        self.mean_jobs_in_system = mean_jobs_in_system

        return mean_jobs_in_system

    def get_mean_jobs_on_queue(self):
        """
        Get average number of jobs in the queue
        """

        self.mean_jobs_in_system = self.mean_jobs_in_system or self.get_mean_jobs_in_system()
        return self.mean_jobs_in_system - (1.0 - self.p0)

    def get_readiness(self):
        """
        Get probability that a randomly chosen source can send a request,
        i.e. the readiness coefficient
        """

        self.p = self.p or self.get_p()

        return self.mu * (1.0 - self.p0) / (self.lam * self.m)

    def get_w1(self):
        """
        Get average waiting time without diff the Laplace-Stieltjes transform
        """

        self.mean_jobs_on_queue = self.mean_jobs_on_queue or self.get_mean_jobs_on_queue()
        return self.mean_jobs_on_queue / self._get_lam_big_d()

    def get_v1(self):
        """
        Get average sojourn time without diff the Laplace-Stieltjes transform
        """

        self.mean_jobs_in_system = self.mean_jobs_in_system or self.get_mean_jobs_in_system()
        return self.mean_jobs_in_system / self._get_lam_big_d()

    def get_w(self):
        """
        Get waiting time raw moments through the diff Laplace-Stieltjes transform
        """

        if self.w:
            return self.w

        self.p = self.p or self.get_p()
        self.mean_jobs_in_system = self.mean_jobs_in_system or self.get_mean_jobs_in_system()

        h = 0.01
        ss = [x * h for x in range(5)]

        ws_dots = [self._ws(s, self.p0, self.mean_jobs_in_system) for s in ss]
        w_diff = diff5dots(ws_dots, h)

        self.w = [-w_diff[0], w_diff[1], -w_diff[2]]

        return self.w

    def get_v(self):
        """
        Get sojourn time raw moments trough convolution with service
        and diff Laplace-Stieltjes transform of waiting time
        """

        if self.v:
            return self.v

        self.w = self.w or self.get_w()
        b = [1.0 / self.mu, 2.0 / pow(self.mu, 2), 6.0 / pow(self.mu, 3)]
        self.v = conv_moments(self.w, b)
        return self.v

    def _calc_m_i(self):
        m_i = []
        m_i.append(1)
        prod = 1
        for i in range(self.m):
            prod *= self.m - i
            m_i.append(prod)
        self.m_i = m_i

    def _get_lam_big_d(self):
        lam_big_d = self.lam * (self.m - self.get_mean_jobs_in_system())
        return lam_big_d

    def _ws(self, s, p0, N):
        """
        Laplace-Stieltjes transform of waiting time
        """
        summ = 0
        for i in range(self.m):
            summ += pow(self.lam, i) * self.m_i[i + 1] / pow(self.mu + s, i)

        return summ * p0 / (self.m - N)
