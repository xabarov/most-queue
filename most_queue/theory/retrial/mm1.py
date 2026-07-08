"""
M/M/1 retrial queue with the classical (linear) retrial policy.

A job that finds the server busy joins the ORBIT and retries after an
exponential time with rate gamma (each orbiting job independently, so the
total retrial rate with j jobs in orbit is j*gamma). There is no queue.

The two-dimensional chain (server state, orbit size) is level-dependent, so
the stationary distribution is computed exactly by solving the truncated
balance equations (the orbit tail decays geometrically; the truncation level
is grown until the tail mass is negligible).

References:
    Falin G.I., Templeton J.G.C. Retrial Queues. Chapman & Hall, 1997.
    Artalejo J.R., Gomez-Corral A. Retrial Queueing Systems. Springer, 2008.
"""

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MM1RetrialCalc(BaseQueue):
    """
    M/M/1 retrial queue (classical linear retrial policy), solved numerically
    to arbitrary precision via truncation of the orbit dimension.
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
        self.mu = None
        self._idle = None  # p(server idle, orbit=j)
        self._busy = None  # p(server busy, orbit=j)

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

    def _solve(self, orbit_cap: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the truncated balance equations for states (c, j), c in {0, 1},
        j = 0..orbit_cap. Returns (idle_probs, busy_probs).
        """
        lam, mu, gamma = self.l, self.mu, self.gamma
        size = 2 * (orbit_cap + 1)  # index: idle j -> j, busy j -> (orbit_cap+1) + j
        q = np.zeros((size, size))
        off = orbit_cap + 1

        for j in range(orbit_cap + 1):
            # idle state (0, j): arrival -> busy(j); retry -> busy(j-1)
            q[j, off + j] += lam
            if j > 0:
                q[j, off + j - 1] += j * gamma
            # busy state (1, j): completion -> idle(j); arrival -> busy(j+1)
            q[off + j, j] += mu
            if j < orbit_cap:
                q[off + j, off + j + 1] += lam
        np.fill_diagonal(q, 0.0)
        np.fill_diagonal(q, -q.sum(axis=1))

        a = q.T.copy()
        a[-1, :] = 1.0
        b = np.zeros(size)
        b[-1] = 1.0
        x = np.linalg.solve(a, b)
        return x[:off], x[off:]

    def _ensure_solved(self) -> None:
        if self._idle is not None:
            return
        self._check_if_servers_and_sources_set()
        ro = self.l / self.mu
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")
        tol = self.calc_params.tolerance
        cap = 32
        while True:
            idle, busy = self._solve(cap)
            if idle[-1] + busy[-1] < tol or cap > 65536:
                break
            cap *= 2
        self._idle, self._busy = idle, busy

    def get_orbit_mean(self) -> float:
        """Mean number of jobs in the orbit."""
        self._ensure_solved()
        j = np.arange(len(self._idle))
        return float(j @ self._idle + j @ self._busy)

    def get_busy_probability(self) -> float:
        """Probability that the server is busy (equals rho = l/mu)."""
        self._ensure_solved()
        return float(np.sum(self._busy))

    def get_p(self) -> list[float]:
        """
        Get probabilities of the number of jobs in the SYSTEM
        (orbit + the job in service, if any).
        """
        self._ensure_solved()
        num = self.calc_params.p_num
        p = np.zeros(num)
        for j, prob in enumerate(self._idle):
            if j < num:
                p[j] += prob
        for j, prob in enumerate(self._busy):
            if j + 1 < num:
                p[j + 1] += prob
        self.p = p.tolist()
        return self.p

    def get_w1(self) -> float:
        """Mean time in orbit (Little's law on the orbit): E[N_o] / lambda."""
        return self.get_orbit_mean() / self.l

    def get_v1(self) -> float:
        """Mean sojourn time: orbit time + service time."""
        return self.get_w1() + 1.0 / self.mu

    def run(self) -> QueueResults:
        """
        Run calculation of the queue system. Only first moments of w and v
        are produced.
        """
        start = self._measure_time()
        with self._validate_state():
            p = self.get_p()
            w1 = self.get_w1()
            v1 = self.get_v1()
            utilization = self.l / self.mu
            self.ro = utilization

        result = QueueResults(p=p, w=[w1, 0, 0], v=[v1, 0, 0], utilization=utilization)
        self._set_duration(result, start)
        return result
