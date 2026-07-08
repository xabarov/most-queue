"""
M/G/1 retrial queue with the classical (linear) retrial policy.

A job that finds the server busy joins the orbit and retries after an
exponential time with rate gamma (total retrial rate j*gamma with j jobs in
orbit). Mean characteristics are in closed form (Falin & Templeton); the mean
orbit size decomposes into the M/G/1 queue-length part plus a retrial term:

    E[N_o] = lam^2 b2 / (2 (1 - rho)) + lam rho / (gamma (1 - rho))

which reduces to the ordinary M/G/1 mean queue length as gamma -> infinity.
The formula was additionally verified in this library against the exact
level-dependent solution of the M/M/1 retrial queue and against simulation.

References:
    Falin G.I., Templeton J.G.C. Retrial Queues. Chapman & Hall, 1997.
    Artalejo J.R., Falin G.I. Standard and retrial queueing systems:
        a comparative analysis. Revista Matematica Complutense, 15(1), 2002.
        doi:10.5209/rev_rema.2002.v15.n1.16950.
"""

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MG1RetrialCalc(BaseQueue):
    """
    M/G/1 retrial queue (classical linear retrial policy): mean orbit size,
    mean orbit waiting time and mean sojourn time in closed form.
    """

    def __init__(self, gamma: float, calc_params: CalcParams | None = None):
        """
        :param gamma: retrial rate of each orbiting job
        """
        super().__init__(n=1, calc_params=calc_params)
        if gamma <= 0:
            raise ValueError(f"Retrial rate gamma must be positive, got {gamma}")
        self.gamma = gamma
        self.l = None
        self.b = None

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
        :param b: raw moments of service time distribution (b1, b2 required)
        """
        if not b or len(b) < 2 or b[0] <= 0:
            raise ValueError("Service time moments must contain at least two moments with positive mean")
        self.b = list(b)
        self.is_servers_set = True

    def _utilization(self) -> float:
        self._check_if_servers_and_sources_set()
        ro = self.l * self.b[0]
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")
        return ro

    def get_orbit_mean(self) -> float:
        """
        Mean number of jobs in the orbit (Falin-Templeton):
        lam^2 b2 / (2(1-rho)) + lam rho / (gamma (1-rho)).
        """
        ro = self._utilization()
        return self.l**2 * self.b[1] / (2.0 * (1.0 - ro)) + self.l * ro / (self.gamma * (1.0 - ro))

    def get_w1(self) -> float:
        """Mean time in orbit (Little's law on the orbit): E[N_o] / lambda."""
        return self.get_orbit_mean() / self.l

    def get_v1(self) -> float:
        """Mean sojourn time: orbit time + own service time."""
        return self.get_w1() + self.b[0]

    def run(self) -> QueueResults:
        """
        Run calculation of the queue system. Only first moments of w and v
        are produced; state probabilities are not computed (use MM1RetrialCalc
        for the exponential case if the distribution is needed).
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            w1 = self.get_w1()
            v1 = self.get_v1()
            self.ro = utilization

        result = QueueResults(w=[w1, 0, 0], v=[v1, 0, 0], utilization=utilization)
        self._set_duration(result, start)
        return result
