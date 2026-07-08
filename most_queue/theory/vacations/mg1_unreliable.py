"""
M/G/1 with an unreliable server (breakdowns and repairs, preemptive-resume).

The server fails according to a Poisson process with rate xi WHILE SERVING
(no failures when idle); repair takes a random time R, after which service of
the interrupted job resumes from the interruption point. The system is then an
ordinary M/G/1 queue in which the service time is replaced by the COMPLETION
TIME C = B + sum of repairs during the service.

Given B = b the number of failures is Poisson(xi*b), so the cumulants of C are
kappa_1 = b*(1 + xi*r1), kappa_k = xi*b*r_k (k >= 2), which yields closed-form
raw moments of C from the raw moments of B and R.

References:
    Avi-Itzhak B., Naor P. Some Queuing Problems with the Service Station
        Subject to Breakdown. Operations Research, 11(3), 1963.
        doi:10.1287/opre.11.3.303.
"""

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.utils.conv import conv_moments


def completion_time_moments(b: list[float], xi: float, r: list[float], num: int = 3) -> list[float]:
    """
    Raw moments of the completion time C (service + repairs, preemptive-resume).

    :param b: raw moments of service time (num moments required)
    :param xi: failure rate while serving
    :param r: raw moments of repair time (num moments required)
    :param num: number of moments to compute (up to 3)
    """
    if xi < 0:
        raise ValueError(f"Failure rate must be non-negative, got {xi}")
    if xi == 0:
        return list(b[:num])
    if len(b) < num or len(r) < num:
        raise ValueError(f"Need at least {num} moments of both service and repair times")

    g = 1.0 + xi * r[0]  # E[C | B=b] = g * b
    c = [g * b[0]]
    if num > 1:
        # E[C^2 | b] = (g*b)^2 + xi*b*r2
        c.append(g * g * b[1] + xi * r[1] * b[0])
    if num > 2:
        # E[C^3 | b] = (g*b)^3 + 3*(g*b)*(xi*b*r2) + xi*b*r3
        c.append(g**3 * b[2] + 3.0 * g * xi * r[1] * b[1] + xi * r[2] * b[0])
    if num > 3:
        # raw4 from cumulants k1=g*b, k2=xi*b*r2, k3=xi*b*r3, k4=xi*b*r4:
        # raw4 = k4 + 4*k3*k1 + 3*k2^2 + 6*k2*k1^2 + k1^4, then E over B
        c.append(
            xi * r[3] * b[0]
            + (4.0 * xi * r[2] * g + 3.0 * xi * xi * r[1] * r[1]) * b[1]
            + 6.0 * xi * r[1] * g * g * b[2]
            + g**4 * b[3]
        )
    if num > 4:
        raise ValueError("Only up to 4 completion time moments are supported")
    return c


class MG1UnreliableCalc(BaseQueue):
    """
    M/G/1 with server breakdowns (Avi-Itzhak-Naor): waiting time is the M/G/1
    waiting time computed for the completion-time distribution; sojourn time is
    waiting plus the (own) completion time.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)

        self.l = None  # arrival intensity
        self.b = None  # service time raw moments
        self.xi = None  # failure rate while serving
        self.r = None  # repair time raw moments

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
        if not b or b[0] <= 0:
            raise ValueError("Service time moments must be non-empty with positive mean")
        self.b = list(b)
        self.is_servers_set = True

    def set_breakdowns(self, xi: float, repair: list[float]):
        """
        Set breakdowns
        :param xi: Poisson failure rate while the server is busy
        :param repair: raw moments of the repair time distribution
        """
        if xi < 0:
            raise ValueError(f"Failure rate must be non-negative, got {xi}")
        if not repair or repair[0] <= 0:
            raise ValueError("Repair time moments must be non-empty with positive mean")
        self.xi = xi
        self.r = list(repair)

    def get_completion_moments(self, num: int = 3) -> list[float]:
        """
        Raw moments of the completion time (service + repairs).
        """
        self._check_if_servers_and_sources_set()
        if self.xi is None:
            raise ValueError("Breakdowns are not set. Use set_breakdowns() method.")
        return completion_time_moments(self.b, self.xi, self.r, num=num)

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time: M/G/1 waiting time for the completion time
        (num waiting moments require num + 1 service and repair moments).
        """
        c = self.get_completion_moments(num + 1)
        mg1 = MG1Calc()
        mg1.set_sources(l=self.l)
        mg1.set_servers(c)
        self.w = mg1.get_w(num)
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: waiting + own completion time.
        """
        w = self.w if self.w is not None else self.get_w(num)
        c = self.get_completion_moments(num)
        self.v = conv_moments(w, c, num=num)
        return self.v

    def run(self, num_of_moments: int = 3) -> QueueResults:
        """
        Run calculation of the queue system.
        """
        start = self._measure_time()
        with self._validate_state():
            c = self.get_completion_moments(num_of_moments)
            utilization = self.l * c[0]
            if utilization >= 1:
                raise ValueError(f"System is unstable: effective load rho={utilization} must be < 1")
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.ro = utilization

        result = QueueResults(w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
