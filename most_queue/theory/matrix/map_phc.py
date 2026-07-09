"""
MAP/PH/c queue (correlated arrivals, phase-type service, c servers) via a QBD.

The phase of level k is (MAP phase) x (configuration of the busy servers'
service phases). Because the c servers are identical, the service
configuration is a multiset — a count vector (n_1, ..., n_ms) telling how many
of the min(k, c) busy servers sit in each of the ms service phases. The QBD is
level-dependent at the boundary (levels 0..c-1 have a growing service-phase
space) and homogeneous from level c on.

Special cases (used as validation): a one-phase PH service reduces this to
MAP/M/c; c = 1 reduces it to MAP/PH/1; Poisson arrivals with H2 service
reproduce the Takahashi-Takami M/H2/c result.

References:
    Neuts M.F. Matrix-Geometric Solutions in Stochastic Models. Johns Hopkins
        University Press, 1981.
    Latouche G., Ramaswami V. Introduction to Matrix Analytic Methods in
        Stochastic Modeling. SIAM, 1999. doi:10.1137/1.9780898719734.
"""

from itertools import combinations_with_replacement

import numpy as np

from most_queue.random.map_ph import MAP, MAPParams, PHParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.matrix.qbd import logarithmic_reduction_g


def _service_states(ms: int, s: int) -> list[tuple[int, ...]]:
    """All count vectors (n_0..n_{ms-1}) with sum s (multisets of busy phases)."""
    states = []
    for combo in combinations_with_replacement(range(ms), s):
        cnt = [0] * ms
        for j in combo:
            cnt[j] += 1
        states.append(tuple(cnt))
    return states


class MapPhCCalc(BaseQueue):
    """
    MAP/PH/c queue: Markovian (correlated) arrivals, phase-type service, c
    identical servers, FCFS, infinite buffer. Produces state probabilities and
    Little-law means. Phase space grows combinatorially, so keep the PH order
    and c modest (e.g. ms <= 3, c <= 6).
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        :param n: number of servers (c)
        """
        super().__init__(n=n, calc_params=calc_params)
        self.map_params: MAPParams | None = None
        self.ph_params: PHParams | None = None
        self._r = None
        self._pi = None  # boundary phase vectors pi_0 .. pi_c
        self._svc = None  # service-state lists per busy count

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

    # ------------------------------------------------------- service blocks
    def _build_service_blocks(
        self, beta, s_mat, s0, c, ms
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Return per-busy-count service states and the transfer matrices."""
        svc = [_service_states(ms, s) for s in range(c + 1)]
        idx = [{st: i for i, st in enumerate(lst)} for lst in svc]

        def refill(s):  # svc[s] -> svc[s+1]: a new server starts in phase j w.p. beta[j]
            mat = np.zeros((len(svc[s]), len(svc[s + 1])))
            for a, st in enumerate(svc[s]):
                for j in range(ms):
                    nxt = list(st)
                    nxt[j] += 1
                    mat[a, idx[s + 1][tuple(nxt)]] += beta[j]
            return mat

        def comp(s):  # svc[s] -> svc[s-1]: a server in phase i completes (no refill)
            mat = np.zeros((len(svc[s]), len(svc[s - 1])))
            for a, st in enumerate(svc[s]):
                for i in range(ms):
                    if st[i] > 0:
                        nxt = list(st)
                        nxt[i] -= 1
                        mat[a, idx[s - 1][tuple(nxt)]] += st[i] * s0[i]
            return mat

        def comp_refill():  # svc[c] -> svc[c]: completion + a waiting job starts (phase j)
            mat = np.zeros((len(svc[c]), len(svc[c])))
            for a, st in enumerate(svc[c]):
                for i in range(ms):
                    if st[i] > 0:
                        for j in range(ms):
                            nxt = list(st)
                            nxt[i] -= 1
                            nxt[j] += 1
                            mat[a, idx[c][tuple(nxt)]] += st[i] * s0[i] * beta[j]
            return mat

        def local(s):  # svc[s] -> svc[s]: phase transitions; diagonal absorbs completion out
            mat = np.zeros((len(svc[s]), len(svc[s])))
            for a, st in enumerate(svc[s]):
                for i in range(ms):
                    if st[i] == 0:
                        continue
                    for j in range(ms):
                        if j != i:
                            nxt = list(st)
                            nxt[i] -= 1
                            nxt[j] += 1
                            mat[a, idx[s][tuple(nxt)]] += st[i] * s_mat[i, j]
                off = mat[a].sum()
                completion = sum(st[i] * s0[i] for i in range(ms))
                mat[a, a] = -off - completion
            return mat

        return svc, refill, comp, comp_refill, local

    def _solve(self) -> None:
        if self._pi is not None:
            return
        self._check_if_servers_and_sources_set()
        d0 = np.asarray(self.map_params.D0, dtype=float)
        d1 = np.asarray(self.map_params.D1, dtype=float)
        beta = np.asarray(self.ph_params.alpha, dtype=float)
        s_mat = np.asarray(self.ph_params.T, dtype=float)
        s0 = -s_mat @ np.ones(s_mat.shape[0])
        ma, ms, c = d0.shape[0], s_mat.shape[0], self.n
        eye_a = np.eye(ma)

        lam = MAP.arrival_rate(self.map_params)
        b1 = float(beta @ np.linalg.inv(-s_mat) @ np.ones(ms))
        ro = lam * b1 / c
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")

        svc, refill, comp, comp_refill, local = self._build_service_blocks(beta, s_mat, s0, c, ms)
        self._svc = svc

        # repeating blocks (levels >= c): all c servers busy
        a0 = np.kron(d1, np.eye(len(svc[c])))  # arrival -> job waits
        a1 = np.kron(d0, np.eye(len(svc[c]))) + np.kron(eye_a, local(c))
        a2 = np.kron(eye_a, comp_refill())  # completion + refill
        g = logarithmic_reduction_g(a0, a1, a2)
        r = a0 @ np.linalg.inv(-(a1 + a0 @ g))
        self._r = r

        # per-level up/down/local blocks for the boundary 0..c
        dims = [ma * len(svc[s]) for s in range(c + 1)]
        up = []  # up[k]: level k -> k+1
        down = []  # down[k]: level k -> k-1 (down[0] unused)
        loc = []  # loc[k]: local at level k
        for k in range(c + 1):
            if k == 0:
                loc.append(d0.copy())
                up.append(np.kron(d1, refill(0)))
                down.append(None)
            elif k < c:
                loc.append(np.kron(d0, np.eye(len(svc[k]))) + np.kron(eye_a, local(k)))
                up.append(np.kron(d1, refill(k)))
                down.append(np.kron(eye_a, comp(k)))
            else:  # k == c: local folds the matrix-geometric tail
                loc.append(a1 + r @ a2)
                up.append(None)
                down.append(np.kron(eye_a, comp(c)))

        # assemble the boundary generator over levels 0..c and solve x @ Q = 0
        offs = np.concatenate([[0], np.cumsum(dims)])
        size = int(offs[-1])
        gen = np.zeros((size, size))
        for k in range(c + 1):
            r0, r1 = offs[k], offs[k + 1]
            gen[r0:r1, r0:r1] = loc[k]
            if k < c:
                c0, c1 = offs[k + 1], offs[k + 2]
                gen[r0:r1, c0:c1] = up[k]
            if k >= 1:
                c0, c1 = offs[k - 1], offs[k]
                gen[r0:r1, c0:c1] = down[k]

        a_full = gen.T.copy()
        norm_row = np.ones(size)
        # tail closure: level c mass sum -> pi_c (I-R)^{-1} 1
        inv = np.linalg.inv(np.eye(len(svc[c]) * ma) - r)
        norm_row[offs[c] : offs[c + 1]] = inv @ np.ones(dims[c])
        a_full[-1, :] = norm_row
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        x = np.linalg.solve(a_full, rhs)
        self._pi = [x[offs[k] : offs[k + 1]] for k in range(c + 1)]

    # ------------------------------------------------------------- results
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
        vec = self._pi[c].copy()
        j = 0
        while c + j < num:
            p[c + j] = float(np.sum(vec))
            vec = vec @ r
            j += 1
        self.p = p
        return p

    def _mean_in_system(self) -> tuple[float, float]:
        self._solve()
        c, r = self.n, self._r
        m = r.shape[0]
        ones = np.ones(m)
        pic = self._pi[c]
        inv = np.linalg.inv(np.eye(m) - r)
        el = sum(k * float(np.sum(self._pi[k])) for k in range(c))
        el += c * float(pic @ inv @ ones) + float(pic @ r @ inv @ inv @ ones)
        eq = float(pic @ r @ inv @ inv @ ones)
        return el, eq

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean waiting time (first moment only): E[W] = E[Nq] / lambda.
        """
        _, eq = self._mean_in_system()
        self.w = [eq / MAP.arrival_rate(self.map_params)]
        return self.w

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sojourn time (first moment only): E[V] = E[L] / lambda.
        """
        el, _ = self._mean_in_system()
        self.v = [el / MAP.arrival_rate(self.map_params)]
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
            b1 = float(
                np.asarray(self.ph_params.alpha)
                @ np.linalg.inv(-np.asarray(self.ph_params.T))
                @ np.ones(np.asarray(self.ph_params.T).shape[0])
            )
            utilization = lam * b1 / self.n
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
