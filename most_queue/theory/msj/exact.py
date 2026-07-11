"""
Exact analysis of the FCFS multiserver-job (MSJ) model for small systems.

Each job class has a server need (servers held simultaneously) and an exponential
service rate; there are k servers and FCFS head-of-line blocking. The exact state
is the arrival-ordered sequence of job classes present (the set in service is the
maximal fitting prefix), which makes the reachable state space grow with the
truncation length — the solver is therefore for small k / few classes / moderate
load, exactly the regime where response-time results are otherwise scarce.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.sim.msj import MsjClass
from most_queue.structs import QueueResults


class MsjExactCalc:
    """
    Exact mean response time of the FCFS MSJ model via a truncated CTMC on the
    arrival-ordered sequence of job classes.

    :param k: number of servers.
    :param classes: list of MsjClass (arrival_rate, servers, mu).
    :param truncation: maximum number of jobs in system (sequence length).
    """

    # cap on the reachable sequence-state space (it grows like m^N)
    MAX_STATES = 500_000

    def __init__(self, k: int, classes: list[MsjClass], truncation: int = 12):
        self.k = k
        self.classes = classes
        self.N = truncation
        self.boundary_mass = None

    def run(self) -> QueueResults:
        """Build and solve the CTMC; return overall and per-class mean sojourn."""
        m = len(self.classes)
        rates = [c.arrival_rate for c in self.classes]
        needs = [c.servers for c in self.classes]
        mus = [c.mu for c in self.classes]

        start = time.process_time()

        # BFS over reachable sequence-states
        s0 = ()
        index = {s0: 0}
        order = [s0]
        edges = []
        head = 0
        while head < len(order):
            seq = order[head]
            si = head
            head += 1

            # arrivals
            if len(seq) < self.N:
                for i in range(m):
                    t = seq + (i,)
                    ti = index.get(t)
                    if ti is None:
                        ti = len(order)
                        index[t] = ti
                        order.append(t)
                        if len(order) > self.MAX_STATES:
                            raise ValueError(
                                f"MSJ sequence state space exceeded {self.MAX_STATES}; "
                                f"reduce truncation ({self.N}) or the number of classes"
                            )
                    edges.append((si, ti, rates[i]))

            # completions of in-service jobs (greedy fitting prefix)
            free = self.k
            for pos, cls in enumerate(seq):
                if needs[cls] <= free:
                    free -= needs[cls]
                    t = seq[:pos] + seq[pos + 1 :]
                    ti = index.get(t)
                    if ti is None:
                        ti = len(order)
                        index[t] = ti
                        order.append(t)
                    edges.append((si, ti, mus[cls]))
                else:
                    break  # FCFS head-of-line blocking

        n_states = len(order)
        rows = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        cols = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        vals = np.fromiter((e[2] for e in edges), dtype=float, count=len(edges))
        Q = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
        out = np.asarray(Q.sum(axis=1)).ravel()
        Q = Q - sp.diags(out)

        A = Q.transpose().tolil()
        A[0, :] = 1.0
        rhs = np.zeros(n_states)
        rhs[0] = 1.0
        pi = spla.spsolve(A.tocsr(), rhs)
        pi = np.maximum(np.real(pi), 0.0)
        pi = pi / pi.sum()

        # per-class mean number in system -> Little's law
        counts = np.zeros((n_states, m))
        for idx, seq in enumerate(order):
            for cls in seq:
                counts[idx, cls] += 1
        e_n = pi @ counts
        v_per_class = [float(e_n[i] / rates[i]) for i in range(m)]
        lam_total = sum(rates)
        v_overall = float(sum(e_n) / lam_total)

        self.boundary_mass = float(pi[np.array([len(s) for s in order]) == self.N].sum())

        res = QueueResults(v=[v_overall, 0, 0, 0])
        res.v_per_class = v_per_class  # type: ignore[attr-defined]
        res.duration = time.process_time() - start
        return res
