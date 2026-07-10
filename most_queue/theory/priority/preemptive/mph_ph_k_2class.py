"""
Exact solver for M/PH/PH/k with two preemptive-resume priority classes, both
with phase-type service. This is the §2.3 base case of the RDR paper: the low
(target) class keeps a phase-type service, so its response-time depends on the
service *variance*, not only the mean.

State = (n_high, n_low, hp, lp) where
  * n_high, n_low  -- number of high / low jobs in system (truncated),
  * hp             -- sorted tuple of the PH phases of the in-service high jobs
                      (high is never preempted, so all min(n_high, k) are active),
  * lp             -- age-ordered tuple (oldest first) of the PH phases of the
                      low jobs that have *started* service. The first
                      min(len(lp), k - min(n_high, k)) of them are in service and
                      advance their phase; the rest are frozen (preempted, resume
                      later at the same phase). Keeping the tuple age-ordered makes
                      FCFS-resume exact.

Only <= k in-service and <= k frozen jobs ever carry a live phase, so the state
space is finite. The chain is built by breadth-first reachability from the empty
state and solved for its stationary distribution; per-class means follow by
Little's law.
"""

import time

import numpy as np
import scipy.sparse as sp

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue


class PhaseType:
    """Minimal phase-type service description (alpha, sub-generator T)."""

    def __init__(self, alpha, T):
        self.alpha = np.asarray(alpha, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.p = len(self.alpha)
        self.exit = -self.T.sum(axis=1)  # completion rate from each phase

    @classmethod
    def exponential(cls, mu):
        return cls([1.0], [[-mu]])

    @classmethod
    def coxian2(cls, p1, mu1, mu2):
        # phase 0 -> phase 1 with prob p1, else exit; phase 1 -> exit
        return cls([1.0, 0.0], [[-mu1, mu1 * p1], [0.0, -mu2]])

    @classmethod
    def from_moments(cls, moments):
        """Fit a 2-phase Coxian to three raw moments (falls back to exponential)."""
        from most_queue.random.distributions import CoxDistribution

        if moments[0] <= 0:
            raise ValueError("mean must be positive")
        cv2 = (moments[1] - moments[0] ** 2) / moments[0] ** 2
        if abs(cv2 - 1.0) < 1e-6:
            return cls.exponential(1.0 / moments[0])
        prm = CoxDistribution.get_params(moments)
        p1 = float(np.real(prm.p1))
        mu1 = float(np.real(prm.mu1))
        mu2 = float(np.real(prm.mu2))
        return cls.coxian2(p1, mu1, mu2)


class MPhPhK2Class(BaseQueue):
    """
    Exact M/PH/PH/k with two preemptive-resume priority classes (index 0 = high).
    Returns per-class mean sojourn/waiting time.
    """

    def __init__(self, n: int, truncation: int = 40, tol: float = 1e-12, max_iter: int = 300_000):
        super().__init__(n=n)
        self.n = n
        self.truncation = truncation
        self.tol = tol
        self.max_iter = max_iter
        self.l_high = None
        self.l_low = None
        self.ph_high: PhaseType | None = None
        self.ph_low: PhaseType | None = None
        self.n_iter_ = 0
        self.num_states_ = 0

    def set_sources(self, l_high: float, l_low: float):  # pylint: disable=arguments-differ
        self.l_high = l_high
        self.l_low = l_low
        self.is_sources_set = True

    def set_servers(self, ph_high: PhaseType, ph_low: PhaseType):  # pylint: disable=arguments-differ
        self.ph_high = ph_high
        self.ph_low = ph_low
        self.is_servers_set = True

    def _hstart(self):
        """Entry phase of a fresh job (assumes a point-mass alpha, true for exp/Coxian)."""
        return int(np.argmax(self.ph_high.alpha)), int(np.argmax(self.ph_low.alpha))

    def _saturate(self, nH, nL, hp, lp):
        """
        Normalize a state by instantaneously starting queued jobs into free
        servers. High always fills first; the low jobs that have started are kept
        in `lp` (age-ordered); any queued low job starts (fresh phase) while a
        server is free. Freezing/resuming needs no change here (it is implicit in
        how many of `lp` are in service, = min(len(lp), k - min(nH, k))).
        """
        _, l0 = self._hstart()
        k = self.n
        free_l = k - min(nH, k)
        lp = tuple(lp)
        # start queued low jobs while a server is free for low and one is waiting
        while min(len(lp), free_l) < free_l and nL > len(lp):
            lp = lp + (l0,)
        return (nH, nL, tuple(sorted(hp)), lp)

    def _transitions(self, state):
        """Yield (target_state, rate) out of `state` (targets already saturated)."""
        nH, nL, hp, lp = state
        k = self.n
        H, L = self.ph_high, self.ph_low
        NH = NL = self.truncation
        sh = min(nH, k)  # high in service (= len(hp))
        free_l = k - sh
        in_serv_l = min(len(lp), free_l)
        h0, _ = self._hstart()

        out = []

        # high arrival
        if nH < NH:
            hp2 = tuple(sorted(hp + (h0,))) if min(nH + 1, k) > sh else hp  # enters service or queues
            out.append((self._saturate(nH + 1, nL, hp2, lp), self.l_high))

        # low arrival (saturate starts it if a server is free)
        if nL < NL:
            out.append((self._saturate(nH, nL + 1, hp, lp), self.l_low))

        # high in-service phase transitions and completions
        for i, ph in enumerate(hp):
            for ph2 in range(H.p):
                if ph2 != ph and H.T[ph, ph2] > 0:
                    out.append((self._saturate(nH, nL, hp[:i] + (ph2,) + hp[i + 1 :], lp), H.T[ph, ph2]))
            crate = H.exit[ph]
            if crate > 0:
                rem = hp[:i] + hp[i + 1 :]
                if nH - 1 >= k:  # a queued high starts fresh
                    rem = rem + (h0,)
                out.append((self._saturate(nH - 1, nL, rem, lp), crate))

        # low in-service phase transitions and completions (front in_serv_l jobs)
        for i in range(in_serv_l):
            ph = lp[i]
            for ph2 in range(L.p):
                if ph2 != ph and L.T[ph, ph2] > 0:
                    out.append((self._saturate(nH, nL, hp, lp[:i] + (ph2,) + lp[i + 1 :]), L.T[ph, ph2]))
            crate = L.exit[ph]
            if crate > 0:
                out.append((self._saturate(nH, nL - 1, hp, lp[:i] + lp[i + 1 :]), crate))

        return out

    def run(self) -> PriorityResults:
        self._check_if_servers_and_sources_set()
        start = time.process_time()

        # breadth-first reachable-state enumeration
        s0 = self._saturate(0, 0, (), ())
        index = {s0: 0}
        order = [s0]
        edges = []  # (src_idx, dst_idx, rate)
        head = 0
        while head < len(order):
            s = order[head]
            si = head
            head += 1
            for t, rate in self._transitions(s):
                ti = index.get(t)
                if ti is None:
                    ti = len(order)
                    index[t] = ti
                    order.append(t)
                edges.append((si, ti, rate))
        total = len(order)
        self.num_states_ = total

        rows = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        cols = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        vals = np.fromiter((e[2] for e in edges), dtype=float, count=len(edges))
        R = sp.coo_matrix((vals, (rows, cols)), shape=(total, total)).tocsr()
        out_rate = np.asarray(R.sum(axis=1)).ravel()

        # uniformized power iteration for the stationary distribution
        lam_unif = out_rate.max() * 1.0000001
        Pt = R.transpose().tocsr() / lam_unif
        stay = 1.0 - out_rate / lam_unif
        pi = np.full(total, 1.0 / total)
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            nxt = Pt.dot(pi) + stay * pi
            nxt /= nxt.sum()
            self.n_iter_ += 1
            if np.abs(nxt - pi).sum() < self.tol:
                pi = nxt
                break
            pi = nxt
        pi = np.maximum(pi, 0.0)
        pi /= pi.sum()

        nH_arr = np.fromiter((s[0] for s in order), dtype=float, count=total)
        nL_arr = np.fromiter((s[1] for s in order), dtype=float, count=total)
        e_nh = float((pi * nH_arr).sum())
        e_nl = float((pi * nL_arr).sum())
        mean_h = self.ph_high.alpha @ np.linalg.solve(-self.ph_high.T, np.ones(self.ph_high.p))
        mean_l = self.ph_low.alpha @ np.linalg.solve(-self.ph_low.T, np.ones(self.ph_low.p))
        v_high = e_nh / self.l_high
        v_low = e_nl / self.l_low

        self.boundary_mass = float(pi[(nH_arr == self.truncation) | (nL_arr == self.truncation)].sum())

        v = [[v_high, 0, 0, 0], [v_low, 0, 0, 0]]
        w = [[v_high - mean_h, 0, 0, 0], [v_low - mean_l, 0, 0, 0]]
        util = (self.l_high * mean_h + self.l_low * mean_l) / self.n
        results = PriorityResults(v=v, w=w, p=[], utilization=util)
        results.duration = time.process_time() - start
        return results
