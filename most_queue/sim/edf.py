"""
Discrete-event simulator for EDF (Earliest Deadline First): the service
discipline itself is deadline-aware, unlike most_queue.theory.utils.sla
(EPIC-021), which only computes a deadline-violation probability on top of
an already-fixed discipline's waiting-time distribution without changing
service order.

K classes share one server. Each job's absolute deadline is fixed at
arrival: arrival_time + D, D drawn per-class from any of the library's
Kendall-notation distributions (deterministic "D" or random, e.g. "M" for
Exp(theta)). Whenever the server frees up, it starts serving the waiting job
with the smallest absolute deadline (ties broken by arrival order); a job
whose deadline is reached before service starts leaves unserved (a miss).

No closed-form/CTMC calculator is provided -- exact finite-state analysis of
EDF is a genuinely hard problem in general (confirmed both by the
heavy-traffic-only literature and by four "surprising reduction" hypotheses
tested and rejected here; see docs/research/edf-scheduling-2026.md). The
classical work-conservation law (sum_k rho_k*E[W_k] = rho*E0/(1-rho)) is
*not* exact here either once reneging is non-negligible -- it only holds in
the low-reneging limit (verified numerically), so it is used as a regression
test restricted to that regime, not a general cross-check. See
docs/roadmaps/edf_scheduling_roadmap.md.
"""

import heapq
import time

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.structs import EDFResults


class EDFQueueSim(BaseSimulationCore):
    """
    Single-server, K-class EDF queue: the waiting job with the smallest
    absolute deadline (arrival_time + D) is served next; a job whose
    deadline elapses before service starts leaves unserved (a miss).

    :param discipline: "edf" (default) picks the smallest absolute deadline;
        "fcfs" picks the earliest arrival instead, everything else (arrival
        process, service, reneging-on-deadline) identical -- an apples-to-
        apples baseline for comparing EDF's deadline-miss rate against FCFS
        under the exact same reneging mechanics (see
        docs/roadmaps/edf_scheduling_roadmap.md sec. 5).
    """

    def __init__(self, n_classes: int, discipline: str = "edf", seed: int | None = None):
        super().__init__(seed=seed)
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")
        if discipline not in ("edf", "fcfs"):
            raise ValueError(f"discipline must be 'edf' or 'fcfs', got {discipline!r}")
        self.n_classes = n_classes
        self.discipline = discipline
        self.sources = None
        self.servers = None
        self.deadline_dists = None
        self.is_sources_set = False
        self.is_servers_set = False
        self.is_deadlines_set = False

    def set_sources(self, l: list[dict]):
        """:param l: per-class arrival spec, [{"type": kendall, "params": ...}]."""
        if len(l) != self.n_classes:
            raise ValueError(f"expected {self.n_classes} sources, got {len(l)}")
        self.sources = [create_distribution(s["params"], s["type"], self.generator) for s in l]
        self.is_sources_set = True

    def set_servers(self, b: list[dict]):
        """:param b: per-class service spec, [{"type": kendall, "params": ...}] (one shared server)."""
        if len(b) != self.n_classes:
            raise ValueError(f"expected {self.n_classes} service specs, got {len(b)}")
        self.servers = [create_distribution(s["params"], s["type"], self.generator) for s in b]
        self.is_servers_set = True

    def set_deadlines(self, deadlines: list[dict]):
        """:param deadlines: per-class relative-deadline spec, [{"type": kendall, "params": ...}]."""
        if len(deadlines) != self.n_classes:
            raise ValueError(f"expected {self.n_classes} deadline specs, got {len(deadlines)}")
        self.deadline_dists = [create_distribution(s["params"], s["type"], self.generator) for s in deadlines]
        self.is_deadlines_set = True

    def run(self, total_served: int, warmup_fraction: float = 0.05) -> EDFResults:  # pylint: disable=too-many-locals
        """Run until `total_served` (non-warmup) jobs have been served."""
        start = time.process_time()
        if not (self.is_sources_set and self.is_servers_set and self.is_deadlines_set):
            raise RuntimeError("set_sources, set_servers and set_deadlines must all be called before run()")

        k = self.n_classes
        t = 0.0
        next_arrival = [self.sources[c].generate() for c in range(k)]
        deadline_heap: list[tuple[float, int]] = []  # (abs_deadline, job_id), lazily invalidated
        cust: dict[int, tuple[int, float, float]] = {}  # id -> (class, arrival_time, abs_deadline)
        uid = 0
        serving: tuple[int, int, float, float] | None = None  # (class, job_id, start_time, arrival_time)
        busy_until = None

        served = [0] * k
        missed = [0] * k
        w_moments = [[0.0, 0.0, 0.0, 0.0] for _ in range(k)]
        v_moments = [[0.0, 0.0, 0.0, 0.0] for _ in range(k)]
        warm = int(total_served * warmup_fraction)
        total_served_count = 0
        busy_time = 0.0
        t_measured_from = None  # set once warmup ends

        while total_served_count < total_served:
            while deadline_heap and deadline_heap[0][1] not in cust:
                heapq.heappop(deadline_heap)
            candidates = list(next_arrival)
            if serving is not None:
                candidates.append(busy_until)
            if deadline_heap:
                candidates.append(deadline_heap[0][0])
            t_prev = t
            t = min(candidates)
            if t_measured_from is not None and serving is not None:
                busy_time += t - t_prev

            if serving is not None and t == busy_until:
                cls, _cid, start_time, arrival_time = serving  # pylint: disable=unpacking-non-sequence
                serving = None
                served[cls] += 1
                if served[cls] > warm:
                    if t_measured_from is None:
                        t_measured_from = t
                    total_served_count += 1
                    w_moments[cls] = refresh_moments_stat(w_moments[cls], start_time - arrival_time, served[cls] - warm)
                    v_moments[cls] = refresh_moments_stat(v_moments[cls], t - arrival_time, served[cls] - warm)
            elif t in next_arrival:
                cls = next_arrival.index(t)
                d = self.deadline_dists[cls].generate()
                cust[uid] = (cls, t, t + d)
                heapq.heappush(deadline_heap, (t + d, uid))
                uid += 1
                next_arrival[cls] = t + self.sources[cls].generate()
            else:
                dl, cid = deadline_heap[0]
                if cid in cust and dl == t:
                    heapq.heappop(deadline_heap)
                    cls = cust[cid][0]
                    del cust[cid]
                    missed[cls] += 1

            if serving is None and cust:
                key_idx = 2 if self.discipline == "edf" else 1  # abs_deadline or arrival_time
                cid = min(cust, key=lambda i: cust[i][key_idx])
                cls, arrival_time, _ = cust.pop(cid)
                serving = (cls, cid, t, arrival_time)
                busy_until = t + self.servers[cls].generate()

        elapsed = t - t_measured_from if t_measured_from is not None else 1.0
        result = EDFResults(
            v=[list(m) for m in v_moments],
            w=[list(m) for m in w_moments],
            miss_prob=[missed[c] / (served[c] + missed[c]) if (served[c] + missed[c]) > 0 else 0.0 for c in range(k)],
            utilization=busy_time / elapsed,
        )
        result.duration = time.process_time() - start
        return result
