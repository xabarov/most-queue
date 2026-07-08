"""
M/G/1 with egalitarian Processor Sharing (PS).

The stationary number of jobs is geometric (1 - rho) * rho^k, insensitive to
the service distribution beyond its mean (BCMP). The conditional mean sojourn
time of a job of size x is exactly x / (1 - rho), so the slowdown is uniform:
every job is stretched by the same factor 1 / (1 - rho).

Only mean sojourn/waiting values are produced: higher PS sojourn moments
require the Yashkov/Ott transform machinery and are left for a follow-up.

References:
    Kleinrock L. Time-shared Systems: A Theoretical Treatment. JACM, 14(2),
        1967. doi:10.1145/321386.321388.
    Baskett F., Chandy K.M., Muntz R.R., Palacios F.G. Open, Closed, and Mixed
        Networks of Queues with Different Classes of Customers. JACM, 22(2),
        1975. doi:10.1145/321879.321887 (insensitivity).
    Yashkov S.F. Processor-Sharing Queues: Some Progress in Analysis. Queueing
        Systems, 2, 1987. doi:10.1007/bf01182931 (higher moments, not implemented).
"""

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MG1PSCalc(BaseQueue):
    """
    M/G/1 Processor Sharing: the server is shared equally by all jobs present
    (each of k jobs is served at rate 1/k). No queue and no waiting in the
    classic sense; "waiting" below is the delay E[V] - b1 caused by sharing.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)

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
        :param b: raw moments of service time distribution (only b[0] is used —
            PS characteristics are insensitive to the rest)
        """
        if not b or b[0] <= 0:
            raise ValueError("Service time moments must be non-empty with positive mean")
        self.b = list(b)
        self.is_servers_set = True

    def _utilization(self) -> float:
        self._check_if_servers_and_sources_set()
        ro = self.l * self.b[0]
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")
        return ro

    def get_p(self) -> list[float]:
        """
        Get probabilities of states: geometric, p[k] = (1 - rho) * rho^k
        (insensitive to the service distribution).
        """
        ro = self._utilization()
        num_probs = self.calc_params.p_num
        self.p = [(1.0 - ro) * ro**k for k in range(num_probs)]
        return self.p

    def get_conditional_sojourn_mean(self, x: float) -> float:
        """
        Exact conditional mean sojourn time of a job of size x: x / (1 - rho).
        """
        return x / (1.0 - self._utilization())

    def get_mean_slowdown(self) -> float:
        """
        Mean slowdown (sojourn / size), uniform over sizes: 1 / (1 - rho).
        """
        return 1.0 / (1.0 - self._utilization())

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sojourn time (first moment only): b1 / (1 - rho).
        """
        self.v = [self.b[0] / (1.0 - self._utilization())]
        return self.v

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sharing delay (first moment only): E[V] - b1 = rho * b1 / (1 - rho).
        """
        ro = self._utilization()
        self.w = [ro * self.b[0] / (1.0 - ro)]
        return self.w

    def run(self, num_of_moments: int = 1) -> QueueResults:
        """
        Run calculation. Only first moments of v and w are produced.
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
