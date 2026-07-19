"""
MMAP[2]/PH[2]/1 priority queue: marked MAP arrivals (two classes sharing one
modulating phase process), phase-type service per class, non-preemptive or
preemptive-resume priority.

Literature: Takine et al. (JORSJ, 1996) — nonpreemptive MAP/G/1 with two
classes; Horvath et al. (Performance Evaluation, 2012) — MMAP/MAP/1
preemptive priority; Klimenok & Dudin (Mathematics, 2020) — correlated
arrivals with priorities.

Exact solution of the truncated CTMC on states
(q0, q1, arrival phase, server state[, frozen low phase for PR]):

- NP: an arriving high-priority job never interrupts service;
- PR: a high-priority arrival seizes the server; the interrupted low job is
  frozen mid-phase at the head of the low queue and resumes from the same PH
  phase once no high-priority work remains (preemptive resume).

Special cases used in tests: a one-phase MMAP is a pair of Poisson flows and
exponential PH is M — the model reduces to the classic M/M/1 two-class
Cobham (NP) and preemptive-resume formulas.
"""

import time

import numpy as np

from most_queue.structs import MulticlassResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary

IDLE = 0


class MapPh1PriorityCalc(BaseQueue):
    """
    Two-class priority MMAP/PH/1 queue (NP or PR).

    :param discipline: "NP" (non-preemptive), "PR" (preemptive resume) or
        "RS" (preemptive repeat with resampling: the interrupted low job goes
        back to the queue and later restarts with a fresh PH draw).
    """

    def __init__(self, discipline: str = "NP"):
        super().__init__(n=1)
        self.discipline = discipline.upper()
        if self.discipline not in ("NP", "PR", "RS"):
            raise ValueError("discipline must be 'NP', 'PR' or 'RS'")
        self.D0 = None
        self.D1 = None  # per class
        self.alpha = None  # per class PH initial vectors
        self.T = None  # per class PH sub-generators
        self.t0 = None  # per class absorption rates
        self.arrival_rates = None
        self.mean_jobs = None

    def set_sources(self, D0, D1_high, D1_low):  # pylint: disable=arguments-differ
        """
        :param D0: MMAP phase transitions without arrivals (m_a x m_a).
        :param D1_high: transition rates with a class-0 (priority) arrival.
        :param D1_low: transition rates with a class-1 arrival.
        """
        self.D0 = np.asarray(D0, dtype=float)
        self.D1 = [np.asarray(D1_high, dtype=float), np.asarray(D1_low, dtype=float)]
        gen_rows = (self.D0 + self.D1[0] + self.D1[1]).sum(axis=1)
        if not np.allclose(gen_rows, 0.0, atol=1e-9):
            raise ValueError("D0 + D1_high + D1_low must be a generator (zero row sums)")
        self.is_sources_set = True

    def set_servers(self, ph_high, ph_low):  # pylint: disable=arguments-differ
        """
        :param ph_high: PH of class-0 service as a pair (alpha, T).
        :param ph_low: PH of class-1 service as a pair (alpha, T).
        """
        self.alpha, self.T, self.t0 = [], [], []
        for alpha, t_matrix in (ph_high, ph_low):
            alpha = np.asarray(alpha, dtype=float)
            t_matrix = np.asarray(t_matrix, dtype=float)
            self.alpha.append(alpha)
            self.T.append(t_matrix)
            self.t0.append(-t_matrix @ np.ones(t_matrix.shape[0]))
        self.is_servers_set = True

    def _fundamental_rates(self) -> list[float]:
        d_gen = self.D0 + self.D1[0] + self.D1[1]
        m_a = d_gen.shape[0]
        a_sys = np.vstack([d_gen.T, np.ones(m_a)])
        rhs = np.zeros(m_a + 1)
        rhs[-1] = 1.0
        theta, *_ = np.linalg.lstsq(a_sys, rhs, rcond=None)
        return [float(theta @ self.D1[k] @ np.ones(m_a)) for k in range(2)]

    def run(self, tail_tol: float = 1e-9, q_start: int = 40, q_max: int = 160) -> MulticlassResults:
        """
        Solve the truncated CTMC (queue caps grown until the boundary mass is
        below `tail_tol`, hard-capped at `q_max` per queue — at high loads
        with bursty MMAPs raise `q_max` consciously: the state count grows as
        q_max^2 * m_a * (1 + m0 + m1) [* (m1+1) for PR]).
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        m_a = self.D0.shape[0]
        m0, m1 = self.T[0].shape[0], self.T[1].shape[0]
        n_srv = 1 + m0 + m1  # idle | class-0 phase | class-1 phase
        f_dim = (m1 + 1) if self.discipline == "PR" else 1
        lam = self._fundamental_rates()

        def srv_high(s):
            return 1 + s

        def srv_low(s):
            return 1 + m0 + s

        q_cap = q_start
        while True:
            m_q = q_cap + 1

            def idx(q0, q1, a, srv, f=0):
                return (((q0 * m_q + q1) * m_a + a) * n_srv + srv) * f_dim + f

            n_states = m_q * m_q * m_a * n_srv * f_dim
            trans = []

            def add(src, dst, rate):
                if rate > 0:
                    trans.append((src, dst, rate))

            for q0 in range(m_q):
                for q1 in range(m_q):
                    for a in range(m_a):
                        for srv in range(n_srv):
                            for f in range(f_dim):
                                state = idx(q0, q1, a, srv, f)
                                # invalid states drain (unreachable, zero mass)
                                invalid = (srv == IDLE and (q0 or q1 or f)) or (f > 0 and not 1 <= srv <= m0)
                                if invalid:
                                    add(state, idx(0, 0, a, IDLE, 0), 1.0)
                                    continue

                                # MMAP phase changes without arrivals
                                for a2 in range(m_a):
                                    if a2 != a:
                                        add(state, idx(q0, q1, a2, srv, f), self.D0[a, a2])

                                # class-0 (priority) arrivals
                                for a2 in range(m_a):
                                    rate = self.D1[0][a, a2]
                                    if rate <= 0:
                                        continue
                                    if srv == IDLE:
                                        for i in range(m0):
                                            add(state, idx(0, 0, a2, srv_high(i), 0), rate * self.alpha[0][i])
                                    elif 1 <= srv <= m0 or self.discipline == "NP":
                                        if q0 < q_cap:
                                            add(state, idx(q0 + 1, q1, a2, srv, f), rate)
                                    elif self.discipline == "RS":
                                        # repeat-resampling: the low job returns to the
                                        # queue and will restart with a fresh PH draw
                                        q1_new = min(q1 + 1, q_cap)
                                        for i in range(m0):
                                            add(state, idx(q0, q1_new, a2, srv_high(i), 0), rate * self.alpha[0][i])
                                    else:  # PR: preempt the low job, freeze its phase
                                        s_low = srv - 1 - m0
                                        for i in range(m0):
                                            add(
                                                state,
                                                idx(q0, q1, a2, srv_high(i), s_low + 1),
                                                rate * self.alpha[0][i],
                                            )

                                # class-1 arrivals
                                for a2 in range(m_a):
                                    rate = self.D1[1][a, a2]
                                    if rate <= 0:
                                        continue
                                    if srv == IDLE:
                                        for j in range(m1):
                                            add(state, idx(0, 0, a2, srv_low(j), 0), rate * self.alpha[1][j])
                                    elif q1 < q_cap:
                                        add(state, idx(q0, q1 + 1, a2, srv, f), rate)

                                # service phase evolution and completions
                                if srv == IDLE:
                                    continue
                                cls, s = (0, srv - 1) if srv <= m0 else (1, srv - 1 - m0)
                                for s2 in range(self.T[cls].shape[0]):
                                    if s2 != s:
                                        add(state, idx(q0, q1, a, srv - s + s2, f), self.T[cls][s, s2])
                                comp = self.t0[cls][s]
                                if comp <= 0:
                                    continue
                                if q0 > 0:
                                    for i in range(m0):
                                        add(state, idx(q0 - 1, q1, a, srv_high(i), f), comp * self.alpha[0][i])
                                elif f > 0:  # resume the frozen low job mid-phase
                                    add(state, idx(q0, q1, a, srv_low(f - 1), 0), comp)
                                elif q1 > 0:
                                    for j in range(m1):
                                        add(state, idx(q0, q1 - 1, a, srv_low(j), 0), comp * self.alpha[1][j])
                                else:
                                    add(state, idx(0, 0, a, IDLE, 0), comp)

            pi = ctmc_stationary(trans, n_states)
            shape = (m_q, m_q, m_a, n_srv, f_dim)
            pi_nd = pi.reshape(shape)
            boundary = pi_nd[q_cap].sum() + pi_nd[:, q_cap].sum()
            if boundary < tail_tol or q_cap >= q_max:
                break
            q_cap *= 2

        qs = np.arange(m_q)
        q0_marg = pi_nd.sum(axis=(1, 2, 3, 4))
        q1_marg = pi_nd.sum(axis=(0, 2, 3, 4))
        p_serv0 = pi_nd[:, :, :, 1 : 1 + m0, :].sum()
        p_serv1 = pi_nd[:, :, :, 1 + m0 :, :].sum()
        p_frozen = pi_nd[:, :, :, :, 1:].sum() if f_dim > 1 else 0.0

        n0 = float(qs @ q0_marg + p_serv0)
        n1 = float(qs @ q1_marg + p_serv1 + p_frozen)
        self.mean_jobs = [n0, n1]
        self.arrival_rates = lam

        v = [[n0 / lam[0]], [n1 / lam[1]]]
        w = [
            [float(qs @ q0_marg) / lam[0]],
            [float(qs @ q1_marg + p_frozen) / lam[1]],
        ]
        self.results = MulticlassResults(
            w=w,
            v=v,
            utilization=float(p_serv0 + p_serv1),
            duration=time.process_time() - start,
        )
        return self.results
