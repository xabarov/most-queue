"""
Unit tests for the exact waiting-time tail P(W>t) on the batch-size-dependent
M/M^[a,b]/1 bulk-service queue (most_queue.theory.batch.bulk_service, EPIC-066).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.utils.sla import mm1_deadline_violation_prob


def _monte_carlo_tail(a, b, lam, mu, points, total_served=800_000, warmup_fraction=0.02, seed=42):
    """Independent from-scratch sampler (not BulkServiceSim) for the wait tail."""
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
            server_done = t + rng.exponential(1.0 / mu(take))

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


def test_b1_get_tail_reduces_exactly_to_mm1():
    """No batching (b=1): the tail must match the classical M/M/1 formula exactly."""
    lam, mu = 0.5, 1.0
    calc = BulkServiceMM1Calc(a=1, b=1, queue_truncation=200)
    calc.set_sources(lam)
    calc.set_servers(mu)
    for t in (0.1, 0.5, 1.0, 2.0, 5.0):
        assert np.isclose(calc.get_tail(t), mm1_deadline_violation_prob(lam, mu, t), rtol=1e-9)


def test_get_tail_matches_independent_monte_carlo_at_a1():
    a, b, lam = 1, 4, 0.6

    def mu(size):
        return 1.0 + 0.3 * size

    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=500)
    calc.set_sources(lam)
    calc.set_servers(mu)

    points = (0.2, 0.5, 1.0, 2.0, 4.0)
    mc, n = _monte_carlo_tail(a, b, lam, mu, points)
    for t in points:
        exact = calc.get_tail(t)
        se = (mc[t] * (1 - mc[t]) / n) ** 0.5
        assert abs(exact - mc[t]) < 5 * se + 1e-4


def test_get_cdf_is_one_minus_tail():
    calc = BulkServiceMM1Calc(a=1, b=3, queue_truncation=150)
    calc.set_sources(0.8)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.1 * size))
    for t in (0.3, 1.0, 3.0):
        assert calc.get_cdf(t) == pytest.approx(1.0 - calc.get_tail(t))


def test_get_tail_is_monotone_decreasing():
    calc = BulkServiceMM1Calc(a=1, b=4, queue_truncation=150)
    calc.set_sources(0.7)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    values = [calc.get_tail(t) for t in np.linspace(0, 10, 20)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_get_tail_at_zero_equals_busy_probability():
    """W=0 exactly when idle (a=1); P(W>0) = 1 - P(idle)."""
    calc = BulkServiceMM1Calc(a=1, b=4, queue_truncation=150)
    calc.set_sources(0.6)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    pi = calc._solve_pi()  # pylint: disable=protected-access
    ig, _ = np.divmod(np.arange(len(pi)), calc.N + 1)
    busy_prob = 1.0 - float(pi[ig == 0].sum())
    assert calc.get_tail(0.0) == pytest.approx(busy_prob)


@pytest.mark.parametrize("a,b", [(2, 4), (3, 5), (2, 2), (4, 6)])
def test_get_tail_matches_independent_monte_carlo_at_a_gt_1(a, b):
    """EPIC-067: idle-refill race-aware tail, any 1 <= a <= b."""
    lam = 0.6

    def mu(size):
        return 1.0 + 0.3 * size

    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=300)
    calc.set_sources(lam)
    calc.set_servers(mu)

    points = (0.2, 0.5, 1.0, 2.0, 4.0)
    mc, n = _monte_carlo_tail(a, b, lam, mu, points)
    for t in points:
        exact = calc.get_tail(t)
        se = (mc[t] * (1 - mc[t]) / n) ** 0.5
        assert abs(exact - mc[t]) < 5 * se + 1e-4


def test_get_tail_rejects_negative_t():
    calc = BulkServiceMM1Calc(a=1, b=4, queue_truncation=100)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    with pytest.raises(ValueError):
        calc.get_tail(-1.0)


if __name__ == "__main__":
    test_b1_get_tail_reduces_exactly_to_mm1()
    test_get_tail_matches_independent_monte_carlo_at_a1()
    test_get_tail_matches_independent_monte_carlo_at_a_gt_1(2, 4)
    test_get_cdf_is_one_minus_tail()
    test_get_tail_is_monotone_decreasing()
    test_get_tail_at_zero_equals_busy_probability()
    print("all bulk-service tail tests passed")
