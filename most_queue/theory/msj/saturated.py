"""
Saturated multiserver-job (MSJ) system: exact throughput and stability threshold.

Following Grosof, Harchol-Balter & Scheller-Wolf, the stability region of an FCFS
MSJ system is characterised by its *saturated* version — an always-backlogged copy
fed an i.i.d. stream of job classes with the arrival-mix probabilities p_i =
lambda_i / Lambda. The saturated system's long-run completion rate X_sat (its
throughput) is the maximum total arrival rate the real system can sustain: the
open system is stable iff Lambda < X_sat.

The saturated CTMC state is (multiset of in-service jobs, class of the blocked
head-of-line job). On each completion the freed servers admit the blocked job if
it now fits and then keep drawing-and-admitting fresh jobs until one does not fit,
which becomes the new blocked head. The state space is small when the server needs
and k are small.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.sim.msj import MsjClass


class MsjSaturatedCalc:
    """
    Exact saturated-throughput and stability threshold of the FCFS MSJ model.

    :param k: number of servers.
    :param classes: list of MsjClass (only the arrival_rate ratios and the
        server needs / service rates matter; arrival_rate sets the class mix).
    """

    def __init__(self, k: int, classes: list[MsjClass]):
        self.k = k
        self.classes = classes
        self.throughput = None
        self.stability_threshold = None

    def _admit_and_draw(self, counts, free):
        """
        From a configuration with `free` servers available and no pending blocked
        job, repeatedly draw a fresh head-of-line job (class ~ p) and admit it
        while it fits; the first job that does not fit becomes the blocked head.
        Returns {(counts_tuple, blocked_class): probability}.
        """
        m = len(self.classes)
        needs = [c.servers for c in self.classes]
        p = self._mix()
        out: dict = {}

        def rec(cnts, fr, prob):
            for i in range(m):
                if needs[i] <= fr:
                    nc = list(cnts)
                    nc[i] += 1
                    rec(tuple(nc), fr - needs[i], prob * p[i])
                else:
                    key = (tuple(cnts), i)
                    out[key] = out.get(key, 0.0) + prob * p[i]

        rec(counts, free, 1.0)
        return out

    def _mix(self):
        rates = [c.arrival_rate for c in self.classes]
        tot = sum(rates)
        return [r / tot for r in rates]

    def _used(self, counts):
        return sum(counts[i] * self.classes[i].servers for i in range(len(self.classes)))

    def run(self):
        """Compute the saturated throughput and the stability threshold."""
        start = time.process_time()
        m = len(self.classes)
        needs = [c.servers for c in self.classes]
        mus = [c.mu for c in self.classes]

        # initial states: fill the empty system by drawing-and-admitting
        init = self._admit_and_draw(tuple([0] * m), self.k)
        index: dict = {}
        order = []
        for st in init:
            if st not in index:
                index[st] = len(order)
                order.append(st)

        edges = []
        head = 0
        while head < len(order):
            counts, blocked = order[head]
            si = head
            head += 1
            free = self.k - self._used(counts)
            for c in range(m):
                if counts[c] == 0:
                    continue
                rate = counts[c] * mus[c]
                # class-c job completes -> free its servers
                nc = list(counts)
                nc[c] -= 1
                free2 = free + needs[c]
                if needs[blocked] <= free2:
                    # blocked head fits now: admit it, then draw-and-admit more
                    nc[blocked] += 1
                    dist = self._admit_and_draw(tuple(nc), free2 - needs[blocked])
                else:
                    dist = {(tuple(nc), blocked): 1.0}
                for st, prob in dist.items():
                    if st not in index:
                        index[st] = len(order)
                        order.append(st)
                    edges.append((si, index[st], rate * prob))

        n = len(order)
        rows = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        cols = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        vals = np.fromiter((e[2] for e in edges), dtype=float, count=len(edges))
        Q = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        out = np.asarray(Q.sum(axis=1)).ravel()
        Q = Q - sp.diags(out)
        A = Q.transpose().tolil()
        A[0, :] = 1.0
        rhs = np.zeros(n)
        rhs[0] = 1.0
        pi = spla.spsolve(A.tocsr(), rhs)
        pi = np.maximum(np.real(pi), 0.0)
        pi = pi / pi.sum()

        # throughput = mean total completion rate
        comp_rate = np.array([sum(cnts[c] * mus[c] for c in range(m)) for (cnts, _b) in order])
        self.throughput = float(pi @ comp_rate)
        self.stability_threshold = self.throughput  # max total arrival rate Lambda
        self.duration = time.process_time() - start
        return self.throughput
