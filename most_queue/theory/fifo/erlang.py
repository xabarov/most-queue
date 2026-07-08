"""
Erlang B (loss system M/M/n/0, valid for M/G/n/0) and Erlang C (delay system M/M/n) calculators.

References:
    Erlang A.K. Solution of Some Problems in the Theory of Probabilities of Significance
        in Automatic Telephone Exchanges. Elektroteknikeren, 13, 1917.
    Sevastyanov B.A. An Ergodic Theorem for Markov Processes and Its Application to
        Telephone Systems with Refusals. Theory of Probability and Its Applications,
        2(1), 1957. doi:10.1137/1102005 (insensitivity of the loss formula: blocking in
        M/G/n/0 depends on the service distribution only through its mean).
    Jagerman D.L. Some Properties of the Erlang Loss Function. Bell System Technical
        Journal, 53(3), 1974 (stable recursion for B(n, a)).
"""

import math

from most_queue.random.distributions import ExpDistribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.conv import conv_moments


def erlang_b_formula(a: float, n: int) -> float:
    """
    Blocking probability B(n, a) via the numerically stable recursion
    B(k, a) = a*B(k-1, a) / (k + a*B(k-1, a)), B(0, a) = 1.

    :param a: offered load a = l / mu (in Erlangs)
    :param n: number of servers
    """
    b = 1.0
    for k in range(1, n + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c_formula(a: float, n: int) -> float:
    """
    Probability of waiting C(n, a) in M/M/n, expressed through Erlang B:
    C = n*B / (n - a*(1 - B)). Requires a < n.

    :param a: offered load a = l / mu (in Erlangs)
    :param n: number of servers
    """
    if a >= n:
        raise ValueError(f"System is unstable: offered load a={a} must be < n={n}")
    b = erlang_b_formula(a, n)
    return n * b / (n - a * (1.0 - b))


class ErlangBCalc(BaseQueue):
    """
    Loss system M/M/n/0 (Erlang B).

    No queue: a job arriving when all n servers are busy is lost. State probabilities
    form a truncated Poisson distribution. By Sevastyanov's insensitivity theorem the
    state probabilities (and blocking) are valid for M/G/n/0 with any service
    distribution with the same mean, so mu may be read as 1 / (mean service time).
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        :param n: number of servers
        """
        super().__init__(n=n, calc_params=calc_params, buffer=0)

        self.l = None  # arrival intensity
        self.mu = None  # service intensity

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """
        Set servers
        :param mu: service rate (1 / mean service time)
        """
        if mu <= 0:
            raise ValueError(f"Service rate must be positive, got {mu}")
        self.mu = mu
        self.is_servers_set = True

    def get_blocking_probability(self) -> float:
        """
        Probability that an arriving job finds all servers busy and is lost.
        By PASTA it equals the stationary probability of state n.
        """
        self._check_if_servers_and_sources_set()
        return erlang_b_formula(self.l / self.mu, self.n)

    def get_p(self) -> list[float]:
        """
        Get probabilities of states (number of busy servers), truncated Poisson:
        p[k] = (a^k / k!) / sum_{i=0..n} (a^i / i!), k = 0..n
        """
        self._check_if_servers_and_sources_set()
        a = self.l / self.mu
        terms = [math.pow(a, k) / math.factorial(k) for k in range(self.n + 1)]
        norm = sum(terms)
        self.p = [t / norm for t in terms]
        return self.p

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time: identically zero (accepted jobs start service at once).
        """
        self.w = [0.0] * num
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time of ACCEPTED jobs: equal to service time moments.
        """
        self.v = ExpDistribution.calc_theory_moments(self.mu, num)
        return self.v

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation of the queue system.

        Returns:
            QueueResults; utilization is the carried load per server: a*(1 - B)/n.
        """
        start = self._measure_time()
        with self._validate_state():
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            a = self.l / self.mu
            blocking = p[self.n]
            utilization = a * (1.0 - blocking) / self.n
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result


class ErlangCCalc(BaseQueue):
    """
    Delay system M/M/n with infinite queue (Erlang C).

    Waiting time is 0 with probability 1 - C(n, a) and exponential Exp(n*mu - l)
    with probability C(n, a), which gives all raw moments in closed form.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        :param n: number of servers
        """
        super().__init__(n=n, calc_params=calc_params)

        self.l = None  # arrival intensity
        self.mu = None  # service intensity

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """
        Set servers
        :param mu: service rate
        """
        if mu <= 0:
            raise ValueError(f"Service rate must be positive, got {mu}")
        self.mu = mu
        self.is_servers_set = True

    def get_waiting_probability(self) -> float:
        """
        Probability that an arriving job has to wait, C(n, a).
        """
        self._check_if_servers_and_sources_set()
        return erlang_c_formula(self.l / self.mu, self.n)

    def get_p(self) -> list[float]:
        """
        Get probabilities of states 0..p_num-1. For k < n Poisson-like terms,
        for k >= n geometric tail with ratio a/n.
        """
        self._check_if_servers_and_sources_set()
        a = self.l / self.mu
        if a >= self.n:
            raise ValueError(f"System is unstable: offered load a={a} must be < n={self.n}")
        rho = a / self.n

        summ = sum(math.pow(a, k) / math.factorial(k) for k in range(self.n))
        p0 = 1.0 / (summ + math.pow(a, self.n) / (math.factorial(self.n) * (1.0 - rho)))

        num_probs = self.calc_params.p_num
        p = [0.0] * num_probs
        for k in range(min(self.n, num_probs)):
            p[k] = p0 * math.pow(a, k) / math.factorial(k)
        if self.n < num_probs:
            p_n = p0 * math.pow(a, self.n) / math.factorial(self.n)
            for k in range(self.n, num_probs):
                p[k] = p_n * math.pow(rho, k - self.n)
        self.p = p
        return p

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time: w_k = C(n, a) * k! / (n*mu - l)^k.
        """
        self._check_if_servers_and_sources_set()
        c = self.get_waiting_probability()
        rate = self.n * self.mu - self.l
        self.w = [c * math.factorial(k) / math.pow(rate, k) for k in range(1, num + 1)]
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: waiting + independent exponential service.
        """
        w = self.w if self.w is not None else self.get_w(num)
        b = ExpDistribution.calc_theory_moments(self.mu, num)
        self.v = conv_moments(w, b, num=num)
        return self.v

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation of the queue system.
        """
        start = self._measure_time()
        with self._validate_state():
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            utilization = self.l / (self.mu * self.n)
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
