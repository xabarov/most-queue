"""
Simulation of a BMAP/PH/1 queue: batch Markovian arrivals, phase-type service,
single FCFS server. Used to validate the analytical BmapPh1Calc.
"""

import time

import numpy as np

from most_queue.random.map_ph import BMAPParams, PHDistribution, PHParams
from most_queue.structs import QueueResults

NUM_OF_MOMENTS = 4
P_NUM = 100


class BmapPh1Sim:
    """
    Discrete-event simulator for BMAP/PH/1. The BMAP phase process jumps between
    phases; a jump via D[k] (k >= 1) emits a batch of k jobs; jobs are served
    FCFS with phase-type service.
    """

    def __init__(self, bmap: BMAPParams, ph: PHParams, seed: int | None = None):
        self.generator = np.random.default_rng(seed)
        self.d = [np.asarray(x, dtype=float) for x in bmap.D]
        self.bmap = bmap
        self.ph = ph
        self.m = self.d[0].shape[0]
        self._build_jump_table()
        self.ttek = 0.0
        self.served = 0
        self.busy_time = 0.0
        self.v = [0.0] * NUM_OF_MOMENTS
        self.w = [0.0] * NUM_OF_MOMENTS
        self.p_time = [0.0] * P_NUM

    def _build_jump_table(self):
        """Per phase: list of (rate, target_phase, batch_size) and total out-rate.

        The total event rate out of phase i is -D0[i][i]: the D0 diagonal
        absorbs every outgoing rate, silent phase changes and batch arrivals
        alike.
        """
        self.out_rate = -np.diag(self.d[0])
        self.jumps = [[] for _ in range(self.m)]
        for i in range(self.m):
            for j in range(self.m):
                if j != i and self.d[0][i, j] > 0:
                    self.jumps[i].append((self.d[0][i, j], j, 0))  # silent D0 transition
            for k in range(1, len(self.d)):
                for j in range(self.m):
                    if self.d[k][i, j] > 0:
                        self.jumps[i].append((self.d[k][i, j], j, k))  # batch of k

    def _next_batch(self, phase):
        """Sample time to the next BMAP transition and its (target phase, batch size)."""
        rate = self.out_rate[phase]
        dt = self.generator.exponential(1.0 / rate)
        opts = self.jumps[phase]
        probs = np.array([o[0] for o in opts]) / rate
        idx = int(self.generator.choice(len(opts), p=probs))
        _, target, batch = opts[idx]
        return dt, target, batch

    def _advance(self, dt, in_sys):
        if in_sys < P_NUM:
            self.p_time[in_sys] += dt
        if in_sys > 0:
            self.busy_time += dt

    def run(self, total_served: int) -> QueueResults:
        """Run until total_served jobs have departed."""
        start_time = time.process_time()
        rng = self.generator
        phase = int(rng.choice(self.m))
        queue = []  # arrival times of waiting jobs
        service_end = None
        current_arr = None
        dt, next_phase, batch = self._next_batch(phase)
        next_bmap = self.ttek + dt

        while self.served < total_served:
            in_sys = len(queue) + (1 if service_end is not None else 0)
            if service_end is not None and service_end <= next_bmap:
                # service completion
                self._advance(service_end - self.ttek, in_sys)
                self.ttek = service_end
                self._register(self.ttek - current_arr)
                if queue:
                    current_arr = queue.pop(0)
                    service_end = self.ttek + PHDistribution(self.ph, generator=rng).generate()
                else:
                    service_end = None
                    current_arr = None
            else:
                # BMAP transition (possibly a batch)
                self._advance(next_bmap - self.ttek, in_sys)
                self.ttek = next_bmap
                for _ in range(batch):
                    if service_end is None:
                        current_arr = self.ttek
                        service_end = self.ttek + PHDistribution(self.ph, generator=rng).generate()
                    else:
                        queue.append(self.ttek)
                phase = next_phase
                dt, next_phase, batch = self._next_batch(phase)
                next_bmap = self.ttek + dt

        return self._results(start_time)

    def _register(self, sojourn):
        # service = sojourn - waiting; but we only kept arrival time, so recover
        # waiting from the job's own service is not tracked here — use sojourn and
        # a separate service sample is not needed: W = V - S is done at aggregate
        # level via Little in the calculator. Here we store sojourn; waiting is
        # sojourn minus the realised service, which we did not keep, so store
        # sojourn only and derive mean waiting as mean sojourn - mean service.
        self.served += 1
        for k in range(NUM_OF_MOMENTS):
            self.v[k] += sojourn ** (k + 1)

    def _results(self, start_time):
        v = [m / self.served for m in self.v]
        t_mat = np.asarray(self.ph.T)
        b1 = float(np.asarray(self.ph.alpha) @ np.linalg.inv(-t_mat) @ np.ones(t_mat.shape[0]))
        w = [v[0] - b1]  # mean waiting = mean sojourn - mean service
        p = [float(t / self.ttek) for t in self.p_time]
        utilization = float(self.busy_time / self.ttek)
        return QueueResults(v=v, w=w, p=p, utilization=utilization, duration=time.process_time() - start_time)
