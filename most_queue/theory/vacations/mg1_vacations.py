"""
Classic M/G/1 vacation models with exhaustive service:
multiple vacations and N-policy.

Both rest on the Fuhrmann-Cooper stochastic decomposition: the stationary
waiting time is the independent sum of the ordinary M/G/1 (Pollaczek-Khinchine)
waiting time and an additional delay determined by the vacation policy.

References:
    Fuhrmann S.W., Cooper R.B. Stochastic Decompositions in the M/G/1 Queue with
        Generalized Vacations. Operations Research, 33(5), 1985.
        doi:10.1287/opre.33.5.1117.
    Doshi B.T. Queueing systems with vacations -- a survey. Queueing Systems, 1,
        1986. doi:10.1007/bf01149327.
    Takagi H. Queueing Analysis, Vol. 1: Vacation and Priority Systems.
        North-Holland, 1991.
"""

import math

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.utils.conv import conv_moments


class MG1MultipleVacationsCalc(BaseQueue):
    """
    M/G/1 with multiple vacations, exhaustive service: whenever the system
    empties, the server leaves for a vacation; if on return the queue is still
    empty, it immediately takes another vacation.

    Decomposition: W = W_{M/G/1} + V_res, where V_res is the stationary
    residual (equilibrium) vacation time, independent of W_{M/G/1}:
    E[V_res^k] = v_{k+1} / ((k+1) * v_1).

    Note: k moments of the waiting time require k+1 raw moments of the
    vacation time. State probabilities are not computed (moments only).
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)

        self.l = None  # arrival intensity
        self.b = None  # service time raw moments
        self.vacation = None  # vacation time raw moments

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
        :param b: raw moments of service time distribution
        """
        self.b = list(b)
        self.is_servers_set = True

    def set_vacations(self, vacation: list[float]):
        """
        Set vacations
        :param vacation: raw moments of the vacation time distribution
            (k+1 moments are needed for k waiting time moments)
        """
        if not vacation or vacation[0] <= 0:
            raise ValueError("Vacation moments must be non-empty with positive mean")
        self.vacation = list(vacation)

    def _residual_vacation_moments(self, num: int) -> list[float]:
        """
        Raw moments of the equilibrium (residual) vacation time:
        r_k = v_{k+1} / ((k+1) * v_1), k = 1..num.
        """
        if self.vacation is None:
            raise ValueError("Vacations are not set. Use set_vacations() method.")
        if len(self.vacation) < num + 1:
            raise ValueError(f"Need at least {num + 1} vacation moments for {num} waiting time moments")
        v1 = self.vacation[0]
        return [self.vacation[k] / ((k + 1) * v1) for k in range(1, num + 1)]

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time: convolution of the Pollaczek-Khinchine
        waiting time moments with the residual vacation moments.
        """
        self._check_if_servers_and_sources_set()

        mg1 = MG1Calc()
        mg1.set_sources(l=self.l)
        mg1.set_servers(self.b)
        w0 = mg1.get_w(num)

        r = self._residual_vacation_moments(num)
        self.w = conv_moments(w0, r, num=num)
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: waiting + independent service.
        """
        w = self.w if self.w is not None else self.get_w(num)
        self.v = conv_moments(w, self.b, num=num)
        return self.v

    def run(self, num_of_moments: int = 3) -> QueueResults:
        """
        Run calculation of the queue system.
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self.l * self.b[0]
            if utilization >= 1:
                raise ValueError(f"System is unstable: utilization rho={utilization} must be < 1")
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.ro = utilization

        result = QueueResults(w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result


class MG1NPolicyCalc(BaseQueue):
    """
    M/G/1 under N-policy: the server switches off when the system empties and
    resumes service only when N jobs have accumulated, then serves exhaustively.

    Decomposition: W = W_{M/G/1} + D, where the additional delay D is the time
    to accumulate J more arrivals, J uniform on {0, ..., N-1}, i.e. an Erlang
    mixture with LST (1/N) * sum_j (l/(l+s))^j. In particular
    E[D] = (N-1)/(2*l). N = 1 reduces exactly to the ordinary M/G/1.
    """

    def __init__(self, big_n: int, calc_params: CalcParams | None = None):
        """
        :param big_n: threshold N of the policy (server resumes at N jobs)
        """
        super().__init__(n=1, calc_params=calc_params)
        if not isinstance(big_n, int) or big_n < 1:
            raise ValueError(f"N must be an integer >= 1, got {big_n}")
        self.big_n = big_n

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
        :param b: raw moments of service time distribution
        """
        self.b = list(b)
        self.is_servers_set = True

    def _additional_delay_moments(self, num: int) -> list[float]:
        """
        Raw moments of the Erlang mixture D: E[D^k] = (1/N) * sum_{j=0}^{N-1}
        E[Erlang(j, l)^k], with E[Erlang(j, l)^k] = j(j+1)...(j+k-1) / l^k.
        """
        moments = []
        for k in range(1, num + 1):
            summ = 0.0
            for j in range(1, self.big_n):
                summ += math.prod(range(j, j + k)) / self.l**k
            moments.append(summ / self.big_n)
        return moments

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time: convolution of the Pollaczek-Khinchine
        waiting time moments with the accumulation delay moments.
        """
        self._check_if_servers_and_sources_set()

        mg1 = MG1Calc()
        mg1.set_sources(l=self.l)
        mg1.set_servers(self.b)
        w0 = mg1.get_w(num)

        if self.big_n == 1:
            self.w = w0
            return self.w

        d = self._additional_delay_moments(num)
        self.w = conv_moments(w0, d, num=num)
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: waiting + independent service.
        """
        w = self.w if self.w is not None else self.get_w(num)
        self.v = conv_moments(w, self.b, num=num)
        return self.v

    def run(self, num_of_moments: int = 3) -> QueueResults:
        """
        Run calculation of the queue system.
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self.l * self.b[0]
            if utilization >= 1:
                raise ValueError(f"System is unstable: utilization rho={utilization} must be < 1")
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.ro = utilization

        result = QueueResults(w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
