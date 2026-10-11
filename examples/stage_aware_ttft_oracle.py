"""Independent finite-loss token replay: no analytical transitions or rewards.

Compared with the solver's admitted-arrival rewards. Loss is decided at actual
arrival instants, including mid-iteration; outputs are timestamped explicitly.
"""

from collections import deque

import numpy as np

# Keep independent physical controls explicit in this test-only oracle.
# pylint: disable=too-many-arguments,too-many-positional-arguments


def replay_loss(arrivals, marks, batch_cap, token_budget, queue_capacity, costs, warmup=0):
    """Return per-arrival wait/TTFT/V (NaN for rejected requests), then drain."""
    arrivals, marks = np.asarray(arrivals), np.asarray(marks)
    n = len(arrivals)
    admitted = np.zeros(n, dtype=bool)
    starts, first, ends = [np.full(n, np.nan) for _ in range(3)]
    attained = np.zeros(n, dtype=int)
    active, waiting = [], deque()
    cursor, now = 0, 0.0
    alpha, beta, gamma = costs
    while cursor < n or waiting or active:
        if not active and not waiting:
            now = arrivals[cursor]
            # The first arrival initiates a cycle immediately; subsequent ties
            # are arrivals during its first iteration, not simultaneous admission.
            waiting.append(cursor)
            admitted[cursor] = True
            cursor += 1
        while waiting and len(active) < batch_cap:
            job = waiting.popleft()
            active.append(job)
            starts[job] = now
        decoding = [job for job in active if attained[job] >= marks[job, 0]]
        prefilling = [job for job in active if job not in decoding]
        steps = {job: 1 for job in decoding}
        remaining = sum(marks[job, 0] for job in prefilling) if token_budget is None else token_budget - len(decoding)
        for job in prefilling:
            amount = min(remaining, marks[job, 0] - attained[job])
            if amount:
                steps[job] = amount
            remaining -= amount
        finish = now + alpha
        for job, amount in steps.items():
            finish += beta * amount + gamma * (attained[job] + amount)
        while cursor < n and arrivals[cursor] < finish:
            if len(waiting) < queue_capacity:
                waiting.append(cursor)
                admitted[cursor] = True
            cursor += 1
        for job, amount in steps.items():
            attained[job] += amount
            if job in decoding and np.isnan(first[job]):
                first[job] = finish
            if attained[job] == sum(marks[job]):
                ends[job] = finish
        active = [job for job in active if np.isnan(ends[job])]
        now = finish
    mask = admitted & (np.arange(n) >= warmup)
    return {
        "w": float(np.mean(starts[mask] - arrivals[mask])),
        "ttft": float(np.mean(first[mask] - arrivals[mask])),
        "v": float(np.mean(ends[mask] - arrivals[mask])),
        "loss": float(1 - admitted[warmup:].mean()),
        "accepted": int(mask.sum()),
    }
