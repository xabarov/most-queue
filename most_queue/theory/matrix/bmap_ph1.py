"""
BMAP/PH/1 queue: batch Markovian arrivals, phase-type service, one server.

This is the general-service member of the BMAP family. Rather than the
Lucantoni M/G/1-type G-matrix machinery, it is solved by the same robust
level-truncation used for BMAP/M/1 — here the state also carries the service
PH phase: (level, BMAP phase, service phase). The CTMC is solved on levels
0..N with N grown until the top-level probability is negligible.

A general (non-phase-type) service time is handled by first fitting a PH
distribution to its moments (e.g. ``fit_h2`` / ``fit_cox`` in
``most_queue.random.utils.fit``) and passing the resulting PHParams.

Reduces exactly to BMAP/M/1 (one-phase PH service) and to MAP/PH/1 (all
batches of size one).

References:
    Lucantoni D. New results on the single server queue with a batch Markovian
        arrival process. Comm. Statist. Stochastic Models, 7(1), 1991.
        doi:10.1080/15326349108807174.
"""

import numpy as np

from most_queue.random.map_ph import BMAPParams, PHParams, bmap_arrival_rate
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class BmapPh1Calc(BaseQueue):
    """
    BMAP/PH/1 queue solved by level truncation over (level, BMAP phase,
    service phase). Produces state probabilities and Little-law means.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)
        self.bmap: BMAPParams | None = None
        self.ph: PHParams | None = None
        self._levels = None  # marginal level probabilities (truncated)

    def set_sources(self, bmap: BMAPParams):  # pylint: disable=arguments-differ
        """
        Set sources
        :param bmap: BMAPParams — the batch Markovian arrival process
        """
        self.bmap = bmap
        self.is_sources_set = True

    def set_servers(self, ph: PHParams):  # pylint: disable=arguments-differ
        """
        Set servers
        :param ph: PHParams (alpha, T) of the service time distribution
        """
        self.ph = ph
        self.is_servers_set = True

    def _solve(self) -> None:
        if self._levels is not None:
            return
        self._check_if_servers_and_sources_set()
        d = [np.asarray(x, dtype=float) for x in self.bmap.D]
        alpha = np.asarray(self.ph.alpha, dtype=float)
        s_mat = np.asarray(self.ph.T, dtype=float)
        s0 = -s_mat @ np.ones(s_mat.shape[0])
        ma, mh, kmax = d[0].shape[0], s_mat.shape[0], len(d) - 1
        eye_a, eye_h = np.eye(ma), np.eye(mh)

        lam = bmap_arrival_rate(self.bmap)
        b1 = float(alpha @ np.linalg.inv(-s_mat) @ np.ones(mh))
        ro = lam * b1
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")

        tol = self.calc_params.tolerance
        cap = max(64, 4 * self.calc_params.p_num)
        while True:
            level_probs = self._solve_truncated(d, alpha, s_mat, s0, ma, mh, kmax, eye_a, eye_h, cap)
            if float(np.sum(level_probs[-1])) < tol or cap > 100_000:
                break
            cap *= 2
        self._levels = [float(np.sum(v)) for v in level_probs]

    def _solve_truncated(
        self, d, alpha, s_mat, s0, ma, mh, kmax, eye_a, eye_h, cap
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        """
        Solve the CTMC on levels 0..cap. Level 0 has dimension ma (idle server);
        levels >= 1 have dimension ma*mh (BMAP phase x service phase).
        """
        d0 = d[0]
        dims = [ma] + [ma * mh] * cap
        offs = np.concatenate([[0], np.cumsum(dims)])
        size = int(offs[-1])
        gen = np.zeros((size, size))

        def rows(lvl):
            return slice(offs[lvl], offs[lvl + 1])

        outer_s0_alpha = np.outer(s0, alpha)  # completion + next job starts (mh x mh)

        for lvl in range(cap + 1):
            r = rows(lvl)
            if lvl == 0:
                gen[r, r] += d0  # local: BMAP phase change, idle
                for k in range(1, kmax + 1):
                    tgt = min(k, cap)
                    # first job of the batch enters service in phase alpha
                    gen[r, rows(tgt)] += np.kron(d[k], alpha.reshape(1, -1))
            else:
                gen[r, r] += np.kron(d0, eye_h) + np.kron(eye_a, s_mat)  # local
                for k in range(1, kmax + 1):
                    tgt = min(lvl + k, cap)
                    gen[r, rows(tgt)] += np.kron(d[k], eye_h)  # waiting batch
                # service completion: level lvl -> lvl-1
                if lvl == 1:
                    gen[r, rows(0)] += np.kron(eye_a, s0.reshape(-1, 1))  # -> idle
                else:
                    gen[r, rows(lvl - 1)] += np.kron(eye_a, outer_s0_alpha)  # next job starts

        a_full = gen.T.copy()
        a_full[-1, :] = 1.0
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        x = np.linalg.solve(a_full, rhs)
        return [x[offs[lvl] : offs[lvl + 1]] for lvl in range(cap + 1)]

    def get_p(self) -> list[float]:
        """
        Get probabilities of the number of jobs in the system (levels).
        """
        self._solve()
        num = self.calc_params.p_num
        p = [0.0] * num
        for k in range(min(num, len(self._levels))):
            p[k] = self._levels[k]
        self.p = p
        return p

    def _means(self) -> tuple[float, float]:
        self._solve()
        el = sum(k * pk for k, pk in enumerate(self._levels))
        eq = sum((k - 1) * pk for k, pk in enumerate(self._levels) if k >= 1)
        return el, eq

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean waiting time (first moment only): E[W] = E[Nq] / lambda.
        """
        _, eq = self._means()
        self.w = [eq / bmap_arrival_rate(self.bmap)]
        return self.w

    def get_v(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sojourn time (first moment only): E[V] = E[L] / lambda.
        """
        el, _ = self._means()
        self.v = [el / bmap_arrival_rate(self.bmap)]
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
            alpha = np.asarray(self.ph.alpha, dtype=float)
            s_mat = np.asarray(self.ph.T, dtype=float)
            b1 = float(alpha @ np.linalg.inv(-s_mat) @ np.ones(s_mat.shape[0]))
            utilization = bmap_arrival_rate(self.bmap) * b1
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
