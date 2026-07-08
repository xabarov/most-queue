"""
Erlang-A: M/M/n+M queue — multi-server system with exponential impatience
(abandonment). A waiting job leaves the queue after an exponential patience
time with rate theta.

Birth-death chain with death rates min(k, n)*mu + max(0, k - n)*theta; always
stable for theta > 0. The workhorse of call-center staffing.

References:
    Palm C. Methods of judging the annoyance caused by congestion.
        Tele (Televerket, Stockholm), 4, 1953 (original Erlang-A).
    Garnett O., Mandelbaum A., Reiman M. Designing a Call Center with
        Impatient Customers. Manufacturing & Service Operations Management,
        4(3), 2002. doi:10.1287/msom.4.3.208.7753.
    Mandelbaum A., Zeltyn S. Service Engineering in Action: The Palm/Erlang-A
        Queue. In: Advances in Services Innovations, Springer, 2007.
        doi:10.1007/978-3-540-29860-1_2.
"""

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MMnImpatienceCalc(BaseQueue):
    """
    Erlang-A (M/M/n+M): n servers, Poisson arrivals, exponential service and
    exponential patience. Multi-server generalization of MM1Impatience.
    """

    def __init__(self, n: int, theta: float, calc_params: CalcParams | None = None):
        """
        :param n: number of servers
        :param theta: impatience (abandonment) rate of a waiting job
        """
        super().__init__(n=n, calc_params=calc_params)
        if theta <= 0:
            raise ValueError(f"Impatience rate theta must be positive, got {theta}")
        self.theta = theta
        self.l = None
        self.mu = None

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
        :param mu: service rate of each server
        """
        if mu <= 0:
            raise ValueError(f"Service rate must be positive, got {mu}")
        self.mu = mu
        self.is_servers_set = True

    def get_p(self) -> list[float]:
        """
        Get probabilities of states (birth-death chain, truncated when the
        tail becomes negligible).
        """
        self._check_if_servers_and_sources_set()
        if self.p is not None:
            return self.p

        tol = self.calc_params.tolerance
        max_num = max(self.calc_params.p_num, self.n + 10)
        unnorm = [1.0]
        for k in range(1, max_num):
            death = self.mu * min(k, self.n) + self.theta * max(0, k - self.n)
            unnorm.append(unnorm[-1] * self.l / death)
            if k > self.n and unnorm[-1] < tol * unnorm[0]:
                break
        total = sum(unnorm)
        self.p = [u / total for u in unnorm]
        return self.p

    def get_mean_jobs(self) -> float:
        """Mean number of jobs in the system."""
        p = self.get_p()
        return sum(k * pk for k, pk in enumerate(p))

    def get_mean_queue(self) -> float:
        """Mean number of jobs waiting in the queue."""
        p = self.get_p()
        return sum((k - self.n) * pk for k, pk in enumerate(p) if k > self.n)

    def get_waiting_probability(self) -> float:
        """Probability that an arriving job finds all servers busy (PASTA)."""
        p = self.get_p()
        return sum(p[self.n :])

    def get_abandonment_probability(self) -> float:
        """
        Probability that an arriving job abandons before reaching a server:
        rate balance gives P_ab = theta * E[Q] / lambda.
        """
        return self.theta * self.get_mean_queue() / self.l

    def get_w1(self) -> float:
        """
        Mean time in queue (until service start OR abandonment), by Little's
        law applied to the queue: E[W] = E[Q] / lambda.
        """
        return self.get_mean_queue() / self.l

    def get_v1(self) -> float:
        """Mean time in system (served and abandoned jobs): E[N] / lambda."""
        return self.get_mean_jobs() / self.l

    def find_min_servers(self, target_abandonment: float, max_n: int = 10_000) -> int:
        """
        Staffing helper: the minimal number of servers n such that the
        abandonment probability does not exceed the target.
        """
        self._check_if_servers_and_sources_set()
        for n in range(1, max_n + 1):
            calc = MMnImpatienceCalc(n=n, theta=self.theta, calc_params=self.calc_params)
            calc.set_sources(self.l)
            calc.set_servers(self.mu)
            if calc.get_abandonment_probability() <= target_abandonment:
                return n
        raise ValueError(f"No n <= {max_n} achieves abandonment <= {target_abandonment}")

    def run(self) -> QueueResults:
        """
        Run calculation of the queue system. Only first moments of w and v
        are produced (means via Little's law).
        """
        start = self._measure_time()
        with self._validate_state():
            p = self.get_p()
            w1 = self.get_w1()
            v1 = self.get_v1()
            # served load per server: (lambda * (1 - P_ab)) / (n * mu)
            utilization = self.l * (1.0 - self.get_abandonment_probability()) / (self.n * self.mu)
            self.ro = utilization

        result = QueueResults(p=p, w=[w1, 0, 0], v=[v1, 0, 0], utilization=utilization)
        self._set_duration(result, start)
        return result
