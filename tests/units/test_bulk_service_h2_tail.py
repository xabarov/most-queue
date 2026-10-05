"""
Unit tests for the exact waiting-time tail P(W>t) on the batch-size-dependent
M/H2(p1,mu1,mu2)^[a,b]/1 bulk-service queue (EPIC-066) -- the branching
phase-type case (each full batch ahead redraws its own H2 phase).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc


def _monte_carlo_tail(a, b, lam, p1_fn, mu1_fn, mu2_fn, points, total_served=1_200_000, warmup_fraction=0.02, seed=11):
    """Independent from-scratch sampler: batch picks ONE H2 phase at batch start."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    server_busy = False
    server_done = inf
    batch: list[float] = []
    batch_start = 0.0
    waits = []
    warmup = int(total_served * warmup_fraction)
    served = 0

    def maybe_start():
        nonlocal server_busy, server_done, batch, batch_start
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = [queue.pop(0) for _ in range(take)]
            server_busy = True
            batch_start = t
            rate = mu1_fn(take) if rng.random() < p1_fn(take) else mu2_fn(take)
            server_done = t + rng.exponential(1.0 / rate)

    while served < total_served + warmup:
        if next_arrival <= server_done:
            t = next_arrival
            queue.append(t)
            next_arrival = t + rng.exponential(1 / lam)
            maybe_start()
        else:
            t = server_done
            for arr in batch:
                if served >= warmup:
                    waits.append(batch_start - arr)
                served += 1
            server_busy = False
            server_done = inf
            batch = []
            maybe_start()

    waits = np.array(waits)
    return {tt: float((waits > tt).mean()) for tt in points}, len(waits)


def test_p1_one_get_tail_reduces_exactly_to_bulk_service_mm1():
    """p1=1 (phase 2 unreachable) must reduce exactly to BulkServiceMM1Calc."""
    lam, b = 0.6, 4

    def mu(size):
        return 1.0 + 0.3 * size

    mm1 = BulkServiceMM1Calc(a=1, b=b, queue_truncation=300)
    mm1.set_sources(lam)
    mm1.set_servers(mu)

    h2 = BulkServiceH2Calc(a=1, b=b, queue_truncation=300)
    h2.set_sources(lam)
    h2.set_servers(p1=1.0, mu1=mu, mu2=lambda size: 999.0)

    for t in (0.2, 1.0, 3.0):
        assert h2.get_tail(t) == pytest.approx(mm1.get_tail(t), abs=1e-7)


def test_get_tail_matches_independent_monte_carlo_branching_case():
    a, b, lam = 1, 3, 0.5

    def p1_fn(size):
        return 0.3 + 0.05 * size

    def mu1_fn(size):
        return 0.5 + 0.1 * size

    def mu2_fn(size):
        return 3.0 + 0.2 * size

    calc = BulkServiceH2Calc(a=a, b=b, queue_truncation=300)
    calc.set_sources(lam)
    calc.set_servers(p1=p1_fn, mu1=mu1_fn, mu2=mu2_fn)

    points = (0.2, 0.6, 1.5, 3.5)
    mc, n = _monte_carlo_tail(a, b, lam, p1_fn, mu1_fn, mu2_fn, points)
    for t in points:
        exact = calc.get_tail(t)
        se = (mc[t] * (1 - mc[t]) / n) ** 0.5
        assert abs(exact - mc[t]) < 6 * se + 1e-4


def test_get_cdf_is_one_minus_tail():
    calc = BulkServiceH2Calc(a=1, b=3, queue_truncation=150)
    calc.set_sources(0.7)
    calc.set_servers(p1=0.4, mu1=1.0, mu2=4.0)
    for t in (0.3, 1.0, 3.0):
        assert calc.get_cdf(t) == pytest.approx(1.0 - calc.get_tail(t))


def test_get_tail_is_monotone_decreasing():
    calc = BulkServiceH2Calc(a=1, b=4, queue_truncation=150)
    calc.set_sources(0.7)
    calc.set_servers(p1=0.35, mu1=0.8, mu2=3.5)
    values = [calc.get_tail(t) for t in np.linspace(0, 10, 20)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_get_tail_rejects_a_greater_than_one():
    calc = BulkServiceH2Calc(a=2, b=4, queue_truncation=100)
    calc.set_sources(1.0)
    calc.set_servers(p1=0.4, mu1=1.0, mu2=4.0)
    with pytest.raises(ValueError):
        calc.get_tail(1.0)


def test_get_tail_rejects_negative_t():
    calc = BulkServiceH2Calc(a=1, b=4, queue_truncation=100)
    calc.set_sources(1.0)
    calc.set_servers(p1=0.4, mu1=1.0, mu2=4.0)
    with pytest.raises(ValueError):
        calc.get_tail(-1.0)
