"""
MAP/M/c queue (correlated arrivals, c exponential servers) via a QBD process.

Level k = number of jobs in the system; the phase is the MAP phase only
(exponential service needs no service phase — the level already encodes how
many servers are busy). The chain is a level-dependent QBD at the boundary
(levels 0..c-1, where the departure rate is k*mu) and a homogeneous QBD from
level c on (all c servers busy, departure rate c*mu).

MAP/M/1 (c = 1) coincides with the single-server QBD result; a Poisson MAP
reduces the model to Erlang C.

References:
    Latouche G., Ramaswami V. Introduction to Matrix Analytic Methods in
        Stochastic Modeling. SIAM, 1999. doi:10.1137/1.9780898719734.
    Neuts M.F. Matrix-Geometric Solutions in Stochastic Models. Johns Hopkins
        University Press, 1981.
"""

import numpy as np

from most_queue.random.map_ph import MAP, MAPParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import logarithmic_reduction_g


class MapMMcCalc(BaseQueue):
    """
    MAP/M/c queue: Markovian (correlated) arrivals and c identical exponential
    servers, FCFS, infinite buffer. Produces state probabilities and
    Little-law means of the number in system / queue / waiting / sojourn.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        :param n: number of servers (c)
        """
        super().__init__(n=n, calc_params=calc_params)
        self.map_params: MAPParams | None = None
        self.mu: float | None = None
        self._r = None
        self._pi = None  # boundary phase vectors pi_0 .. pi_c

    def set_sources(self, map_params: MAPParams):  # pylint: disable=arguments-differ
        """
        Set sources
        :param map_params: MAPParams (D0, D1) of the arrival process
        """
        self.map_params = map_params
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

    def _solve(self) -> None:
        if self._pi is not None:
            return
        self._check_if_servers_and_sources_set()
        d0 = np.asarray(self.map_params.D0, dtype=float)
        d1 = np.asarray(self.map_params.D1, dtype=float)
        m = d0.shape[0]
        c = self.n
        mu = self.mu
        eye = np.eye(m)

        lam = MAP.arrival_rate(self.map_params)
        ro = lam / (c * mu)
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")

        # repeating blocks for levels >= c: up=D1, local=D0 - c*mu*I, down=c*mu*I
        a0 = d1
        a1 = d0 - c * mu * eye
        a2 = c * mu * eye
        g = logarithmic_reduction_g(a0, a1, a2)
        r = a0 @ np.linalg.inv(-(a1 + a0 @ g))
        self._r = r

        # boundary balance for pi_0 .. pi_c (each length m), block-tridiagonal.
        # level k local block: D0 - min(k,c)*mu*I ; up block: D1 ; down from k: min(k,c)*mu*I
        # level c "local" folds the matrix-geometric tail: A1 + R A2
        size = (c + 1) * m
        gen = np.zeros((size, size))

        def block(i, j):
            return slice(i * m, (i + 1) * m), slice(j * m, (j + 1) * m)

        for k in range(c + 1):
            busy = min(k, c)
            if k < c:
                local = d0 - busy * mu * eye
            else:
                local = a1 + r @ a2  # tail closure at level c
            gen[block(k, k)] = local
            if k < c:
                gen[block(k, k + 1)] = d1  # up
            if k >= 1:
                gen[block(k, k - 1)] = min(k, c) * mu * eye  # down

        # stationary: x @ gen = 0, with normalization
        # sum_{k<c} pi_k 1 + pi_c (I-R)^{-1} 1 = 1
        a_full = gen.T.copy()
        norm_row = np.zeros(size)
        for k in range(c):
            norm_row[k * m : (k + 1) * m] = 1.0
        norm_row[c * m :] = np.linalg.inv(eye - r) @ np.ones(m)
        a_full[-1, :] = norm_row
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        x = np.linalg.solve(a_full, rhs)
        self._pi = [x[k * m : (k + 1) * m] for k in range(c + 1)]

    def get_p(self) -> list[float]:
        """
        Get probabilities of the number of jobs in the system (levels).
        """
        self._solve()
        c, r = self.n, self._r
        num = self.calc_params.p_num
        p = [0.0] * num
        for k in range(min(c, num)):
            p[k] = float(np.sum(self._pi[k]))
        # tail k >= c: pi_{c+j} = pi_c R^j
        vec = self._pi[c].copy()
        j = 0
        while c + j < num:
            p[c + j] = float(np.sum(vec))
            vec = vec @ r
            j += 1
        self.p = p
        return p

    def _mean_in_system(self) -> tuple[float, float]:
        """Return (E[L], E[Nq]) — mean number in system and in queue."""
        self._solve()
        c, r = self.n, self._r
        ones = np.ones(r.shape[0])
        pic = self._pi[c]
        inv = np.linalg.inv(np.eye(r.shape[0]) - r)
        # boundary part k < c
        el = sum(k * float(np.sum(self._pi[k])) for k in range(c))
        # tail sum_{j>=0} (c+j) pi_c R^j 1 = c*pi_c inv 1 + pi_c R inv^2 1
        el += c * float(pic @ inv @ ones) + float(pic @ r @ inv @ inv @ ones)
        # queue: sum_{j>=1} j pi_c R^j 1 = pi_c R inv^2 1
        eq = float(pic @ r @ inv @ inv @ ones)
        return el, eq

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean waiting time (first moment only): E[W] = E[Nq] / lambda.
        """
        _, eq = self._mean_in_system()
        lam = MAP.arrival_rate(self.map_params)
        self.w = [eq / lam]
        return self.w

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sojourn time (first moment only): E[V] = E[L] / lambda.
        """
        el, _ = self._mean_in_system()
        lam = MAP.arrival_rate(self.map_params)
        self.v = [el / lam]
        return self.v

    def run(self, num_of_moments: int = 1) -> QueueResults:  # pylint: disable=unused-argument
        """
        Run calculation. Only first moments of w and v are produced.
        """
        start = self._measure_time()
        with self._validate_state():
            p = self.get_p()
            w = self.get_w()
            v = self.get_v()
            lam = MAP.arrival_rate(self.map_params)
            utilization = lam / (self.n * self.mu)
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
