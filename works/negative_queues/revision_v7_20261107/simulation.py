"""Independent event simulator: no phase-queue or production simulator code."""

import math
import time
from collections import deque

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv


def run(case, seed, sample=500000, warm=50000):
    begin = time.perf_counter()
    n, lam, delta, cv, mean = (case[k] for k in ["n", "lam", "delta", "cv", "mean"])
    action, mode = case["action"], case["mode"]
    rng = np.random.default_rng(seed)
    h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(mean, cv)) if cv > 1 else None
    if action == "fixed":
        threshold = min(h.mu1, h.mu2) if h else 1 / (mean * cv**2)
        if 5 * delta >= threshold:
            raise ValueError("Fifth-moment condition requires a separate heavy-tail protocol")

    def service():
        if h:
            return rng.exponential(1 / (h.mu1 if rng.random() < h.p1 else h.mu2))
        return rng.gamma(1 / cv**2, mean * cv**2)

    jobs = [None] * n
    completions = [math.inf] * n
    queue = deque()
    now = 0.0
    arrival = rng.exponential(1 / lam)
    reset = rng.exponential(1 / delta) if delta else math.inf
    departed = 0
    sums = np.zeros(5)
    halves = np.zeros((2, 3))
    occupancy = np.zeros(2)
    sample_time = 0.0
    max_queue = 0

    def start(slot, arr):
        b = service()
        jobs[slot] = (arr, now, b)
        completions[slot] = now + b

    def record(arr, first, success):
        nonlocal departed
        if warm <= departed < warm + sample:
            w = (now if first is None else first) - arr
            v = now - arr
            sums[:] += np.array([w, w * w, v, v * v, success])
            half = 0 if departed - warm < sample // 2 else 1
            halves[half] += [w, v, 1]
        departed += 1

    while departed < warm + sample:
        slot = min(range(n), key=completions.__getitem__)
        clock = min(arrival, reset, completions[slot])
        if departed >= warm:
            dt = clock - now
            busy = sum(j is not None for j in jobs)
            occupancy += [dt * len(queue), dt * (len(queue) + busy)]
            sample_time += dt
        now = clock
        if clock == arrival:
            arrival += rng.exponential(1 / lam)
            free = next((k for k, j in enumerate(jobs) if j is None), None)
            if free is None:
                queue.append(now)
            else:
                start(free, now)
            if departed >= warm:
                max_queue = max(max_queue, len(queue))
        elif clock == reset:
            reset += rng.exponential(1 / delta)
            busy = [k for k, j in enumerate(jobs) if j is not None]
            selected = [busy[int(rng.integers(len(busy)))]] if mode == "rcs" and busy else busy
            if action == "remove":
                if mode == "disaster":
                    for k in selected:
                        arr, first, _ = jobs[k]
                        record(arr, first, False)
                        jobs[k] = None
                        completions[k] = math.inf
                    while queue:
                        record(queue.popleft(), None, False)
                elif selected:
                    k = selected[0]
                    arr, first, _ = jobs[k]
                    record(arr, first, False)
                    jobs[k] = None
                    completions[k] = math.inf
                    if queue:
                        start(k, queue.popleft())
            else:
                for k in selected:
                    completions[k] = now + (jobs[k][2] if action == "fixed" else service())
        else:
            arr, first, _ = jobs[slot]
            record(arr, first, True)
            jobs[slot] = None
            completions[slot] = math.inf
            if queue:
                start(slot, queue.popleft())
    values = sums / sample
    return dict(
        case=case,
        seed=seed,
        sample=sample,
        warmup=warm,
        w=float(values[0]),
        v=float(values[2]),
        sw=float(np.sqrt(values[1] - values[0] ** 2)),
        sv=float(np.sqrt(values[3] - values[2] ** 2)),
        q=float(values[4]),
        little_w=float(occupancy[0] / sample_time / lam),
        little_v=float(occupancy[1] / sample_time / lam),
        half_means=(halves[:, :2] / halves[:, 2:]).tolist(),
        max_queue=max_queue,
        seconds=time.perf_counter() - begin,
    )
