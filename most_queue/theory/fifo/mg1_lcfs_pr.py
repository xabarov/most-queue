"""
M/G/1 with preemptive-resume LCFS (LCFS-PR).

A newly arriving job preempts the job in service; preempted work is resumed
later from the interruption point. The stationary number of jobs is the same
geometric law as under PS (BCMP insensitivity), and the sojourn time of a job
is distributed exactly as an ordinary M/G/1 busy period initiated by its own
service time — so all sojourn moments follow from the Takacs busy-period
recursions already available in the library.

References:
    Baskett F., Chandy K.M., Muntz R.R., Palacios F.G. Open, Closed, and Mixed
        Networks of Queues with Different Classes of Customers. JACM, 22(2),
        1975. doi:10.1145/321879.321887 (LCFS-PR as a BCMP node, insensitivity).
    Takacs L. Introduction to the Theory of Queues. Oxford University Press,
        1962 (busy period moments).
"""

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.busy_periods import busy_calc


class MG1LcfsPrCalc(BaseQueue):
    """
    M/G/1 LCFS-PR: sojourn time moments are the M/G/1 busy period moments;
    state probabilities are geometric. Waiting (time not in service) is
    reported by its mean only: E[W] = E[V] - b1.
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
        :param b: raw moments of service time distribution
            (num sojourn moments require num service moments)
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

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: M/G/1 busy period moments (Takacs).
        """
        self._utilization()
        self.v = busy_calc(self.l, self.b, num=num)
        return self.v

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean time not in service (first moment only): E[V] - b1.
        """
        v = self.v if self.v is not None else self.get_v()
        self.w = [v[0] - self.b[0]]
        return self.w

    def run(self, num_of_moments: int = 3) -> QueueResults:
        """
        Run calculation of the queue system.
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p()
            v = self.get_v(num_of_moments)
            w = self.get_w()
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
