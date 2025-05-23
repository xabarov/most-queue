"""
Calculation of the Engset model for M/M/1 with a finite number of sources.
"""
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.diff5dots import diff5dots


class Engset:
    """
    Calculation of the Engset model for M/M/1 with a finite number of sources.
    """

    def __init__(self, lam: float, mu: float, m: int):
        """
        lam - arrival rate of requests from each source
        mu - service rate
        m - number of request sources
        """
        self.lam = lam
        self.mu = mu
        self.ro = lam / mu
        self.m = m
        self._calc_m_i()

        self.p = None
        self.p0 = None

    def get_p(self) -> list[float]:
        """
        Get probabilities of states of the system
        """
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

    def get_N(self) -> float:
        """
        Get average number of jobs in the system
        """

        if self.p is None:
            self.get_p()

        N = 0
        for i, mm in enumerate(self.m_i):
            N += i * mm * pow(self.ro, i)
        N *= self.p0

        return N

    def get_Q(self):
        """
        Get average number of jobs in the queue
        """
        if self.p is None:
            self.get_p()

        return self.get_N() - (1.0 - self.p0)

    def get_kg(self):
        """
        Get probability that a randomly chosen source can send a request,
        i.e. the readiness coefficient
        """
        if self.p is None:
            self.get_p()

        return self.mu * (1.0 - self.p0) / (self.lam * self.m)

    def get_w1(self):
        """
        Get average waiting time without diff the Laplace-Stieltjes transform
        """
        return self.get_Q() / self._get_lam_big_d()

    def get_v1(self):
        """
        Get average sojourn time without diff the Laplace-Stieltjes transform
        """
        return self.get_N() / self._get_lam_big_d()

    def get_w(self):
        """
        Get waiting time initial moments through the diff Laplace-Stieltjes transform
        """

        if self.p is None:
            self.get_p()

        h = 0.01
        ss = [x * h for x in range(5)]

        N = self.get_N()

        ws_dots = [self._ws(s, self.p0, N) for s in ss]
        w_diff = diff5dots(ws_dots, h)

        return [-w_diff[0], w_diff[1], -w_diff[2]]

    def get_v(self):
        """
        Get sojourn time initial moments trough convolution with service 
        and diff Laplace-Stieltjes transform of waiting time
        """
        w = self.get_w()
        b = [1.0 / self.mu, 2.0 / pow(self.mu, 2), 6.0 / pow(self.mu, 3)]
        return conv_moments(w, b)

    def _calc_m_i(self):
        m_i = []
        m_i.append(1)
        prod = 1
        for i in range(self.m):
            prod *= (self.m - i)
            m_i.append(prod)
        self.m_i = m_i

    def _get_lam_big_d(self):
        lam_big_d = self.lam * (self.m - self.get_N())
        return lam_big_d

    def _ws(self, s, p0, N):
        """
        Laplace-Stieltjes transform of waiting time
        """
        summ = 0
        for i in range(self.m):
            summ += pow(self.lam, i) * self.m_i[i + 1] / pow(self.mu + s, i)

        return summ * p0 / (self.m - N)
