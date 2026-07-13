"""
BCMP networks (Baskett, Chandy, Muntz, Palacios — JACM 1975): multi-class
product-form networks with four station types:

- "fcfs"    — FCFS, exponential service, class-independent rate (BCMP type 1);
- "ps"      — processor sharing, general service, class-dependent means (type 2);
- "is"      — infinite server / delay, general service (type 3);
- "lcfs_pr" — LCFS preemptive-resume, general service (type 4).

PS / IS / LCFS-PR stations are insensitive: only mean service times enter the
formulas. Single-channel queueing stations (fcfs/ps/lcfs_pr) and delay
stations are supported.

`BCMPOpenNetworkCalc` — open network with Poisson external arrivals per class
and per-class Markovian routing; per-class means are exact:
L_ir = rho_ir / (1 - rho_i) for queueing stations, L_ir = rho_ir for IS.

`BCMPClosedNetworkCalc` — closed multi-chain network solved by exact
multi-chain MVA (Reiser & Lavenberg, JACM 1980): recursion over all
population vectors n <= N with W_ir(n) = s_ir (1 + L_i(n - e_r)).

Mixed networks are out of scope.
"""

import itertools
import time

import numpy as np

from most_queue.structs import BCMPNetworkResults
from most_queue.theory.networks.traffic import solve_traffic_equations

QUEUEING_TYPES = ("fcfs", "ps", "lcfs_pr")
STATION_TYPES = QUEUEING_TYPES + ("is",)


def _validate_station_types(station_types: list[str]) -> list[str]:
    types = [t.lower() for t in station_types]
    for t in types:
        if t not in STATION_TYPES:
            raise ValueError(f"Unknown station type '{t}'; supported: {STATION_TYPES}")
    return types


def _validate_fcfs_rates(types: list[str], s: np.ndarray) -> None:
    """BCMP type-1 condition: FCFS service means must be class-independent."""
    for i, t in enumerate(types):
        if t == "fcfs" and not np.allclose(s[i], s[i][0], rtol=1e-9):
            raise ValueError(
                f"Node {i}: BCMP FCFS stations require a class-independent " f"service rate, got means {s[i]}"
            )


class BCMPOpenNetworkCalc:
    """
    Open multi-class BCMP network (exact product form, mean values).
    """

    def __init__(self):
        self.arrival_rates = None
        self.R = None  # per-class routing matrices, (m+1 x m+1) each
        self.s = None  # mean service times, s[i][r]
        self.station_types = None
        self.is_sources_set = False
        self.is_nodes_set = False
        self.results = None

    def set_sources(self, arrival_rates: list[float], R: list):
        """
        :param arrival_rates: external arrival rate per class.
        :param R: routing matrix per class, each dim (m + 1 x m + 1) (same
            format as `OpenNetworkCalc`: row 0 — from the source, last
            column — out of the system).
        """
        self.arrival_rates = [float(x) for x in arrival_rates]
        self.R = [np.asarray(r, dtype=float) for r in R]
        if len(self.R) != len(self.arrival_rates):
            raise ValueError("Need one routing matrix per class")
        self.is_sources_set = True

    def set_nodes(self, s: list[list[float]], station_types: list[str]):
        """
        :param s: mean service times, s[i][r] — node i, class r.
        :param station_types: per node, one of "fcfs", "ps", "lcfs_pr", "is".
        """
        self.s = np.asarray(s, dtype=float)
        self.station_types = _validate_station_types(station_types)
        _validate_fcfs_rates(self.station_types, self.s)
        self.is_nodes_set = True

    def run(self) -> BCMPNetworkResults:
        """
        Run the exact product-form calculation.
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise ValueError("Sources and nodes must be set before run()")

        n_classes = len(self.arrival_rates)
        m = self.s.shape[0]

        lam = np.zeros((m, n_classes))
        for r in range(n_classes):
            lam[:, r] = solve_traffic_equations(self.arrival_rates[r], self.R[r])

        rho = lam * self.s  # rho[i, r]
        rho_total = rho.sum(axis=1)
        for i, t in enumerate(self.station_types):
            if t in QUEUEING_TYPES and rho_total[i] >= 1.0:
                raise ValueError(f"Node {i} is unstable: utilization {rho_total[i]:.3f} >= 1")

        mean_jobs = np.zeros((m, n_classes))
        for i, t in enumerate(self.station_types):
            if t == "is":
                mean_jobs[i] = rho[i]
            else:
                mean_jobs[i] = rho[i] / (1.0 - rho_total[i])

        with np.errstate(divide="ignore", invalid="ignore"):
            v_node = np.where(lam > 1e-12, mean_jobs / lam, 0.0)

        v = [[float(mean_jobs[:, r].sum() / self.arrival_rates[r])] for r in range(n_classes)]

        self.results = BCMPNetworkResults(
            v=v,
            intensities=[[float(lam[i, r]) for i in range(m)] for r in range(n_classes)],
            loads=[float(x) for x in rho_total],
            mean_jobs=[[float(mean_jobs[i, r]) for i in range(m)] for r in range(n_classes)],
            v_node=[[float(v_node[i, r]) for i in range(m)] for r in range(n_classes)],
            duration=time.process_time() - start,
        )
        return self.results


class BCMPClosedNetworkCalc:
    """
    Closed multi-chain BCMP network solved by exact multi-chain MVA.

    Populations are given per class; single-channel queueing stations and
    IS (delay) stations are supported.
    """

    def __init__(self):
        self.R = None  # per-class routing matrices, (m x m) each
        self.N = None  # population per class
        self.s = None
        self.station_types = None
        self.e = None  # visit ratios e[r][i]
        self.is_sources_set = False
        self.is_nodes_set = False
        self.results = None

    def set_sources(self, R: list, N: list[int]):
        """
        :param R: routing matrix per class, each dim (m x m), rows sum to 1.
        :param N: population of each class.
        """
        self.R = [np.asarray(r, dtype=float) for r in R]
        self.N = [int(x) for x in N]
        for k, r in enumerate(self.R):
            if not np.allclose(r.sum(axis=1), 1.0, atol=1e-8):
                raise ValueError(f"Class {k}: rows of a closed routing matrix must sum to 1")
        self.is_sources_set = True

    def set_nodes(self, s: list[list[float]], station_types: list[str]):
        """
        :param s: mean service times, s[i][r] — node i, class r.
        :param station_types: per node, one of "fcfs", "ps", "lcfs_pr", "is".
        """
        self.s = np.asarray(s, dtype=float)
        self.station_types = _validate_station_types(station_types)
        _validate_fcfs_rates(self.station_types, self.s)
        self.is_nodes_set = True

    def _visit_ratios(self) -> np.ndarray:
        n_classes = len(self.N)
        m = self.R[0].shape[0]
        e = np.zeros((n_classes, m))
        for r in range(n_classes):
            A = self.R[r].T - np.eye(m)
            A[0, :] = 0.0
            A[0, 0] = 1.0
            rhs = np.zeros(m)
            rhs[0] = 1.0
            e[r] = np.linalg.solve(A, rhs)
        return e

    def run(self) -> BCMPNetworkResults:
        """
        Exact multi-chain MVA over all population vectors n <= N.
        """
        start = time.process_time()
        if not (self.is_sources_set and self.is_nodes_set):
            raise ValueError("Sources and nodes must be set before run()")

        n_classes = len(self.N)
        m = self.s.shape[0]
        self.e = self._visit_ratios()

        # l_total[n] — total mean jobs per node for population vector n
        l_total = {tuple([0] * n_classes): np.zeros(m)}
        w = np.zeros((m, n_classes))
        x = np.zeros(n_classes)
        l_per_class = np.zeros((m, n_classes))

        ranges = [range(N_r + 1) for N_r in self.N]
        vectors = sorted(itertools.product(*ranges), key=sum)
        for n_vec in vectors:
            if sum(n_vec) == 0:
                continue
            w.fill(0.0)
            for r in range(n_classes):
                if n_vec[r] == 0:
                    continue
                prev = list(n_vec)
                prev[r] -= 1
                l_prev = l_total[tuple(prev)]
                for i, t in enumerate(self.station_types):
                    if t == "is":
                        w[i, r] = self.s[i, r]
                    else:
                        w[i, r] = self.s[i, r] * (1.0 + l_prev[i])

            x = np.array(
                [n_vec[r] / float(np.dot(self.e[r], w[:, r])) if n_vec[r] > 0 else 0.0 for r in range(n_classes)]
            )
            l_per_class = w * (self.e.T * x)  # L_ir = X_r e_ir W_ir
            l_total[n_vec] = l_per_class.sum(axis=1)

        loads = np.zeros(m)
        for i, t in enumerate(self.station_types):
            if t in QUEUEING_TYPES:
                loads[i] = float(sum(x[r] * self.e[r][i] * self.s[i, r] for r in range(n_classes)))

        v = [[float(self.N[r] / x[r])] if x[r] > 0 else [0.0] for r in range(n_classes)]

        self.results = BCMPNetworkResults(
            v=v,
            intensities=[[float(x[r] * self.e[r][i]) for i in range(m)] for r in range(n_classes)],
            loads=[float(l) for l in loads],
            mean_jobs=[[float(l_per_class[i, r]) for i in range(m)] for r in range(n_classes)],
            v_node=[[float(w[i, r]) for i in range(m)] for r in range(n_classes)],
            throughput=[float(x[r]) for r in range(n_classes)],
            duration=time.process_time() - start,
        )
        return self.results
