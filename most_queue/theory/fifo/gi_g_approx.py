"""
Two-moment approximations for GI/G/1 and GI/G/m mean waiting time.

These are APPROXIMATIONS: only the first waiting/sojourn moment is produced
(w and v lists have length 1), state probabilities are not computed.

References:
    Kingman J.F.C. Some Inequalities for the Queue GI/G/1. Biometrika, 49(3/4),
        1962. doi:10.1093/biomet/49.3-4.315 (upper bound, exact in heavy traffic).
    Kraemer W., Langenbach-Belz M. Approximate Formulae for the Delay in the
        Queueing System GI/G/1. Proc. 8th International Teletraffic Congress
        (ITC-8), Melbourne, 1976 (cv-dependent correction factor; reduces to the
        exact Pollaczek-Khinchine formula for M/G/1).
    Allen A.O. Probability, Statistics, and Queueing Theory with Computer Science
        Applications. 2nd ed., Academic Press, 1990 (Allen-Cunneen GI/G/m formula;
        exact for M/M/m).
    Kimura T. A Two-Moment Approximation for the Mean Waiting Time in the GI/G/s
        Queue. Management Science, 32(6), 1986. doi:10.1287/mnsc.32.6.751
        (system interpolation over D/M/s, M/D/s, M/M/s; NOT implemented here --
        requires exact D/M/s and M/D/s waiting times).
"""

import math

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fifo.erlang import erlang_c_formula


def _rates_and_cv2(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """
    Derive (lambda, b1, cv2_arrival, cv2_service) from raw moments of
    interarrival (a) and service (b) times.
    """
    if len(a) < 2 or len(b) < 2:
        raise ValueError("At least two raw moments are required for both arrival and service times")
    lam = 1.0 / a[0]
    cv2_a = (a[1] - a[0] ** 2) / (a[0] ** 2)
    cv2_b = (b[1] - b[0] ** 2) / (b[0] ** 2)
    if cv2_a < 0 or cv2_b < 0:
        raise ValueError("Negative variance derived from moments; check the input moments")
    return lam, b[0], cv2_a, cv2_b


def kingman_bound_w1(a: list[float], b: list[float]) -> float:
    """
    Kingman's upper bound for the mean waiting time in GI/G/1:
    E[W] <= lambda * (Var[A] + Var[B]) / (2 * (1 - rho)).
    Asymptotically exact in heavy traffic (rho -> 1).
    """
    lam, b1, cv2_a, cv2_b = _rates_and_cv2(a, b)
    rho = lam * b1
    if rho >= 1:
        raise ValueError(f"System is unstable: utilization rho={rho} must be < 1")
    var_a = cv2_a * a[0] ** 2
    var_b = cv2_b * b1**2
    return lam * (var_a + var_b) / (2.0 * (1.0 - rho))


def klb_w1(a: list[float], b: list[float]) -> float:
    """
    Kraemer & Langenbach-Belz (1976) approximation for the mean waiting time
    in GI/G/1: the Kingman-style two-moment term times a correction factor
    g(rho, cv2_a, cv2_b). Exact for M/G/1 (cv2_a = 1 gives g = 1).
    """
    lam, b1, cv2_a, cv2_b = _rates_and_cv2(a, b)
    rho = lam * b1
    if rho >= 1:
        raise ValueError(f"System is unstable: utilization rho={rho} must be < 1")

    base = rho * b1 * (cv2_a + cv2_b) / (2.0 * (1.0 - rho))

    if cv2_a <= 1:
        g = math.exp(-2.0 * (1.0 - rho) * (1.0 - cv2_a) ** 2 / (3.0 * rho * (cv2_a + cv2_b)))
    else:
        g = math.exp(-(1.0 - rho) * (cv2_a - 1.0) / (cv2_a + 4.0 * cv2_b))
    return base * g


class GIG1ApproxCalc(BaseQueue):
    """
    GI/G/1: two-moment approximation of the mean waiting/sojourn time.

    Methods: "klb" (default, Kraemer & Langenbach-Belz; exact for M/G/1) or
    "kingman" (upper bound). Only first moments are produced.
    """

    def __init__(self, approximation: str = "klb", calc_params: CalcParams | None = None):
        """
        :param approximation: "klb" or "kingman"
        """
        super().__init__(n=1, calc_params=calc_params)
        if approximation not in ("klb", "kingman"):
            raise ValueError(f'approximation must be "klb" or "kingman", got {approximation}')
        self.approximation = approximation

        self.a = None  # interarrival time raw moments
        self.b = None  # service time raw moments

    def set_sources(self, a: list[float]):  # pylint: disable=arguments-differ
        """
        Set sources
        :param a: raw moments of interarrival time distribution (at least two)
        """
        self.a = list(a)
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set servers
        :param b: raw moments of service time distribution (at least two)
        """
        self.b = list(b)
        self.is_servers_set = True

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Approximate mean waiting time. Only the first moment is available.
        """
        self._check_if_servers_and_sources_set()
        if self.approximation == "kingman":
            w1 = kingman_bound_w1(self.a, self.b)
        else:
            w1 = klb_w1(self.a, self.b)
        self.w = [w1]
        return self.w

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Approximate mean sojourn time: E[V] = E[W] + b1.
        """
        w = self.w if self.w is not None else self.get_w()
        self.v = [w[0] + self.b[0]]
        return self.v

    def run(self, num_of_moments: int = 1) -> QueueResults:
        """
        Run calculation. num_of_moments is accepted for interface compatibility;
        only the first moment is produced (approximation).
        """
        start = self._measure_time()
        with self._validate_state():
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            utilization = self.b[0] / self.a[0]
            self.ro = utilization

        result = QueueResults(w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result


class GIGmApproxCalc(BaseQueue):
    """
    GI/G/m: Allen-Cunneen two-moment approximation of the mean waiting time:
    E[W] ~= C(m, a) / (m*mu - lambda) * (cv2_a + cv2_b) / 2,
    where C(m, a) is the Erlang C waiting probability computed for the
    equivalent M/M/m system. Exact for M/M/m (cv2_a = cv2_b = 1).
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        :param n: number of servers
        """
        super().__init__(n=n, calc_params=calc_params)

        self.a = None  # interarrival time raw moments
        self.b = None  # service time raw moments

    def set_sources(self, a: list[float]):  # pylint: disable=arguments-differ
        """
        Set sources
        :param a: raw moments of interarrival time distribution (at least two)
        """
        self.a = list(a)
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set servers
        :param b: raw moments of service time distribution (at least two)
        """
        self.b = list(b)
        self.is_servers_set = True

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Approximate mean waiting time (Allen-Cunneen). Only the first moment.
        """
        self._check_if_servers_and_sources_set()
        lam, b1, cv2_a, cv2_b = _rates_and_cv2(self.a, self.b)
        mu = 1.0 / b1
        offered = lam / mu
        if offered >= self.n:
            raise ValueError(f"System is unstable: offered load a={offered} must be < n={self.n}")
        c_wait = erlang_c_formula(offered, self.n)
        w_mmn = c_wait / (self.n * mu - lam)
        self.w = [w_mmn * (cv2_a + cv2_b) / 2.0]
        return self.w

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Approximate mean sojourn time: E[V] = E[W] + b1.
        """
        w = self.w if self.w is not None else self.get_w()
        self.v = [w[0] + self.b[0]]
        return self.v

    def run(self, num_of_moments: int = 1) -> QueueResults:
        """
        Run calculation. Only the first moment is produced (approximation).
        """
        start = self._measure_time()
        with self._validate_state():
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            utilization = self.b[0] / (self.a[0] * self.n)
            self.ro = utilization

        result = QueueResults(w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
