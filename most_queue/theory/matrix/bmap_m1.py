"""
BMAP/M/1 queue: batch Markovian arrivals, single exponential server.

A batch of k jobs raises the level by k, so the level process is not a QBD
(up-jumps are unbounded) but an M/G/1-type Markov chain. The stationary
distribution here is obtained by a robust level-truncation: the CTMC is solved
on levels 0..N with N grown until the top-level probability is negligible.
This sidesteps the fragile M/G/1-type G-matrix / Ramaswami machinery and is
easy to validate — it reduces exactly to M^[X]/M/1 (Poisson batch arrivals)
and to MAP/M/1 (all batches of size one).

References:
    Lucantoni D. New results on the single server queue with a batch Markovian
        arrival process. Comm. Statist. Stochastic Models, 7(1), 1991.
        doi:10.1080/15326349108807174.
    Neuts M.F. Matrix-Geometric Solutions in Stochastic Models. Johns Hopkins
        University Press, 1981.
"""

import numpy as np

from most_queue.random.map_ph import BMAPParams, bmap_arrival_rate
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class BmapM1Calc(BaseQueue):
    """
    BMAP/M/1 queue solved by level truncation. Produces state probabilities and
    Little-law means of the number in system / queue / waiting / sojourn.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)
        self.bmap: BMAPParams | None = None
        self.mu: float | None = None
        self._levels = None  # marginal level probabilities (long, truncated)

    def set_sources(self, bmap: BMAPParams):  # pylint: disable=arguments-differ
        """
        Set sources
        :param bmap: BMAPParams — the batch Markovian arrival process
        """
        self.bmap = bmap
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

    def _solve(self) -> None:
        if self._levels is not None:
            return
        self._check_if_servers_and_sources_set()
        d = [np.asarray(x, dtype=float) for x in self.bmap.D]
        m = d[0].shape[0]
        mu = self.mu
        kmax = len(d) - 1  # maximum batch size

        lam = bmap_arrival_rate(self.bmap)
        ro = lam / mu
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")

        tol = self.calc_params.tolerance
        cap = max(64, 4 * self.calc_params.p_num)
        while True:
            level_probs = self._solve_truncated(d, mu, m, kmax, cap)
            top = float(np.sum(level_probs[-1]))
            if top < tol or cap > 200_000:
                break
            cap *= 2
        # aggregate to marginal level probabilities
        self._levels = [float(np.sum(v)) for v in level_probs]

    def _solve_truncated(
        self, d, mu, m, kmax, cap
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Solve the CTMC on levels 0..cap; return per-level phase vectors."""
        n_levels = cap + 1
        size = n_levels * m
        gen = np.zeros((size, size))

        def blk(i, j):
            return slice(i * m, (i + 1) * m), slice(j * m, (j + 1) * m)

        for lvl in range(n_levels):
            # local: D0, minus service out-rate for lvl >= 1
            local = d[0].copy()
            if lvl >= 1:
                local = local - mu * np.eye(m)
                gen[blk(lvl, lvl - 1)] += mu * np.eye(m)  # service completion
            gen[blk(lvl, lvl)] += local
            # batch arrivals: lvl -> lvl + k (capped at the top level)
            for k in range(1, kmax + 1):
                tgt = min(lvl + k, n_levels - 1)
                gen[blk(lvl, tgt)] += d[k]

        # solve x @ gen = 0 with normalization
        a_full = gen.T.copy()
        a_full[-1, :] = 1.0
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        x = np.linalg.solve(a_full, rhs)
        return [x[lvl * m : (lvl + 1) * m] for lvl in range(n_levels)]

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
        """Return (E[L], E[Nq]) — mean number in system and in queue."""
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
            utilization = bmap_arrival_rate(self.bmap) / self.mu
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
