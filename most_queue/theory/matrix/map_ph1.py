"""
MAP/PH/1 queue (FCFS, infinite buffer) solved as a QBD process,
with M/PH/1 and PH/PH/1 as special cases.

Level k = number of jobs in the system. Level 0 phases are the MAP phases
(server idle); level k >= 1 phases are (MAP phase) x (service PH phase).
The stationary distribution is matrix-geometric (QBD); waiting time moments
are obtained by numerically differentiating the scalar waiting-time LST of an
arriving job (remaining service of the job in service plus full services of
the queued jobs ahead).

References:
    Neuts M.F. Matrix-Geometric Solutions in Stochastic Models. Johns Hopkins
        University Press, 1981.
    Latouche G., Ramaswami V. Introduction to Matrix Analytic Methods in
        Stochastic Modeling. SIAM, 1999. doi:10.1137/1.9780898719734.
"""

import numpy as np

from most_queue.random.map_ph import MAP, MAPParams, PHDistribution, PHParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import QBDSolver
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.derivative import derivative


class MapPh1Calc(BaseQueue):
    """
    MAP/PH/1 queue: correlated arrivals (MAP) and phase-type service.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)

        self.map_params: MAPParams | None = None
        self.ph_params: PHParams | None = None
        self._solver: QBDSolver | None = None

    def set_sources(self, map_params: MAPParams):  # pylint: disable=arguments-differ
        """
        Set sources
        :param map_params: MAPParams (D0, D1) of the arrival process
        """
        self.map_params = map_params
        self.is_sources_set = True

    def set_servers(self, ph_params: PHParams):  # pylint: disable=arguments-differ
        """
        Set servers
        :param ph_params: PHParams (alpha, T) of the service time distribution
        """
        self.ph_params = ph_params
        self.is_servers_set = True

    # ------------------------------------------------------------ internals
    def _build_solver(self) -> QBDSolver:
        if self._solver is not None:
            return self._solver
        d0 = np.asarray(self.map_params.D0, dtype=float)
        d1 = np.asarray(self.map_params.D1, dtype=float)
        beta = np.asarray(self.ph_params.alpha, dtype=float)
        s = np.asarray(self.ph_params.T, dtype=float)
        s0 = -s @ np.ones(s.shape[0])
        i_a = np.eye(d0.shape[0])
        i_s = np.eye(s.shape[0])

        a0 = np.kron(d1, i_s)  # arrival at level >= 1
        a1 = np.kron(d0, i_s) + np.kron(i_a, s)  # phase changes without level change
        a2 = np.kron(i_a, np.outer(s0, beta))  # service completion, next job starts
        b00 = d0  # level 0: idle server, MAP phases only
        b01 = np.kron(d1, beta.reshape(1, -1))  # first arrival starts service
        b10 = np.kron(i_a, s0.reshape(-1, 1))  # last job leaves

        self._solver = QBDSolver(a0, a1, a2, b00, b01, b10)
        self._solver.solve()
        return self._solver

    def _utilization(self) -> float:
        lam = MAP.arrival_rate(self.map_params)
        b1 = PHDistribution.calc_theory_moments(self.ph_params, 1)[0]
        ro = lam * b1
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")
        return ro

    def _w_lst(self, sv: float) -> float:
        """
        Waiting time LST of an arriving job, w*(s) = E[e^{-s W}]:
        conditioning on the state seen at an arrival epoch (MAP-weighted).
        """
        solver = self._build_solver()
        d1 = np.asarray(self.map_params.D1, dtype=float)
        s = np.asarray(self.ph_params.T, dtype=float)
        beta = np.asarray(self.ph_params.alpha, dtype=float)
        s0 = -s @ np.ones(s.shape[0])
        lam = MAP.arrival_rate(self.map_params)

        m_s = np.linalg.inv(sv * np.eye(s.shape[0]) - s)
        r_vec = m_s @ s0  # r_j(s): remaining service LST from phase j
        b_scalar = float(beta @ r_vec)  # full service LST
        d1_row = d1 @ np.ones(d1.shape[0])  # arrival intensity by MAP phase

        # level 0: arriving job starts service immediately, W = 0
        total = float(solver.pi0 @ d1_row)
        # levels >= 1: pi_1 (I - b(s) R)^{-1} u(s), u = kron(d1_row, r_vec)
        u = np.kron(d1_row, r_vec)
        m = solver.r.shape[0]
        geom = np.linalg.inv(np.eye(m) - b_scalar * solver.r)
        total += float(solver.pi1 @ geom @ u)
        return total / lam

    # ------------------------------------------------------------- results
    def get_p(self) -> list[float]:
        """
        Get probabilities of the number of jobs in the system (levels).
        """
        self._check_if_servers_and_sources_set()
        solver = self._build_solver()
        self.p = solver.marginal_level_probs(self.calc_params.p_num)
        return self.p

    def get_w(self, num: int = 3) -> list[float]:
        """
        Raw moments of waiting time via numerical differentiation of the LST.
        """
        self._check_if_servers_and_sources_set()
        b1 = PHDistribution.calc_theory_moments(self.ph_params, 1)[0]
        w = [0.0] * num
        for i in range(num):
            w[i] = derivative(self._w_lst, 0, dx=1e-3 / b1, n=i + 1, order=9)
            if (i + 1) % 2 != 0:
                w[i] = -w[i]
        self.w = [float(np.real(x)) for x in w]
        return self.w

    def get_v(self, num: int = 3) -> list[float]:
        """
        Raw moments of sojourn time: waiting + independent own service.
        """
        w = self.w if self.w is not None else self.get_w(num)
        b = PHDistribution.calc_theory_moments(self.ph_params, num)
        self.v = conv_moments(w, b, num=num)
        return self.v

    def run(self, num_of_moments: int = 3) -> QueueResults:
        """
        Run calculation of the queue system.
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


class MPh1Calc(MapPh1Calc):
    """
    M/PH/1: Poisson arrivals, phase-type service (special case of MAP/PH/1).
    """

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        super().set_sources(MAP.poisson(l))


class PhPh1Calc(MapPh1Calc):
    """
    PH/PH/1: renewal PH arrivals, phase-type service (special case of MAP/PH/1).
    """

    def set_sources(self, arrival_ph: PHParams):  # pylint: disable=arguments-differ
        """
        Set sources
        :param arrival_ph: PHParams of the interarrival time distribution
        """
        super().set_sources(MAP.from_ph_renewal(arrival_ph))
