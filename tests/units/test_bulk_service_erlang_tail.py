"""
Unit tests for the exact waiting-time tail P(W>t) on the batch-size-dependent
M/Erlang(k,rate)^[a,b]/1 bulk-service queue (EPIC-066).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc


def _monte_carlo_tail(a, b, k, lam, rate_fn, points, total_served=1_000_000, warmup_fraction=0.02, seed=7):
    """Independent from-scratch sampler: Erlang(k, rate) batch service = sum of k iid Exp(rate)."""
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
            server_done = t + rng.exponential(1.0 / rate_fn(take), size=k).sum()

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


def test_k1_get_tail_reduces_exactly_to_bulk_service_mm1():
    lam, b = 0.6, 4

    def mu(size):
        return 1.0 + 0.3 * size

    mm1 = BulkServiceMM1Calc(a=1, b=b, queue_truncation=300)
    mm1.set_sources(lam)
    mm1.set_servers(mu)

    erl = BulkServiceErlangCalc(a=1, b=b, k=1, queue_truncation=300)
    erl.set_sources(lam)
    erl.set_servers(mu)

    for t in (0.2, 1.0, 3.0):
        assert erl.get_tail(t) == pytest.approx(mm1.get_tail(t), rel=1e-9)


def test_get_tail_matches_independent_monte_carlo_at_k3():
    a, b, k, lam = 1, 3, 3, 0.5

    def rate_fn(size):
        return 2.0 + 0.4 * size

    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=300)
    calc.set_sources(lam)
    calc.set_servers(rate_fn)

    points = (0.3, 0.8, 1.5, 3.0)
    mc, n = _monte_carlo_tail(a, b, k, lam, rate_fn, points)
    for t in points:
        exact = calc.get_tail(t)
        se = (mc[t] * (1 - mc[t]) / n) ** 0.5
        assert abs(exact - mc[t]) < 5 * se + 1e-4


def test_get_cdf_is_one_minus_tail():
    calc = BulkServiceErlangCalc(a=1, b=3, k=2, queue_truncation=150)
    calc.set_sources(0.7)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.1 * size))
    for t in (0.3, 1.0, 3.0):
        assert calc.get_cdf(t) == pytest.approx(1.0 - calc.get_tail(t))


def test_get_tail_is_monotone_decreasing():
    calc = BulkServiceErlangCalc(a=1, b=4, k=3, queue_truncation=150)
    calc.set_sources(0.7)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    values = [calc.get_tail(t) for t in np.linspace(0, 10, 20)]
    assert all(a >= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("a,b,k", [(2, 4, 2), (3, 5, 2), (2, 2, 3)])
def test_get_tail_matches_independent_monte_carlo_at_a_gt_1(a, b, k):
    """EPIC-067: idle-refill race-aware tail, any 1 <= a <= b."""
    lam = 0.5

    def rate_fn(size):
        return 2.0 + 0.4 * size

    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=300)
    calc.set_sources(lam)
    calc.set_servers(rate_fn)

    points = (0.3, 0.8, 1.5, 3.0)
    mc, n = _monte_carlo_tail(a, b, k, lam, rate_fn, points)
    for t in points:
        exact = calc.get_tail(t)
        se = (mc[t] * (1 - mc[t]) / n) ** 0.5
        assert abs(exact - mc[t]) < 5 * se + 1e-4


def test_get_tail_rejects_negative_t():
    calc = BulkServiceErlangCalc(a=1, b=4, k=2, queue_truncation=100)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    with pytest.raises(ValueError):
        calc.get_tail(-1.0)
