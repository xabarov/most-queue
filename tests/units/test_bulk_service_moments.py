"""
Unit tests for exact raw moments of N and W on the batch-size-dependent
M/M^[a,b]/1 bulk-service queue (most_queue.theory.batch.bulk_service, EPIC-032).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.fifo.mg1 import MG1Calc


def _monte_carlo_w_moments(a, b, lam, mu, total_served=400_000, warmup_fraction=0.05, seed=7):
    """Independent from-scratch sampler (not BulkServiceSim) for W moments."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    server_busy = False
    server_done = inf
    batch: list[float] = []
    batch_start = 0.0
    wait_samples = []
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
                    wait_samples.append(batch_start - arr)
                served += 1
            server_busy = False
            server_done = inf
            batch = []
            maybe_start()

    w = np.array(wait_samples)
    return [float(np.mean(w**k)) for k in (1, 2, 3, 4)]


def test_get_w_matches_independent_monte_carlo_at_a1():
    a, b, lam = 1, 4, 1.0
    mu = lambda size: 1.0 / (0.2 + 0.07 * size)  # noqa: E731

    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=150)
    calc.set_sources(lam)
    calc.set_servers(mu)
    theory = calc.get_w(num=4)

    mc = _monte_carlo_w_moments(a, b, lam, mu, total_served=1_500_000)
    # higher moments have higher MC variance; loosen tolerance accordingly
    assert np.isclose(theory[0], mc[0], rtol=0.03)
    assert np.isclose(theory[1], mc[1], rtol=0.05)
    assert np.isclose(theory[2], mc[2], rtol=0.08)
    assert np.isclose(theory[3], mc[3], rtol=0.12)


def test_get_w_rejects_a_greater_than_one():
    calc = BulkServiceMM1Calc(a=2, b=4, queue_truncation=100)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    with pytest.raises(ValueError):
        calc.get_w()


def test_b1_get_w_reduces_exactly_to_mm1():
    """No batching (b=1): W moments must match the classical M/M/1 formula exactly."""
    lam, mu = 0.6, 1.0
    calc = BulkServiceMM1Calc(a=1, b=1, queue_truncation=400)
    calc.set_sources(lam)
    calc.set_servers(mu)
    w = calc.get_w(num=3)

    mg1 = MG1Calc()
    mg1.set_sources(lam)
    mg1.set_servers([1 / mu, 2 / mu**2, 6 / mu**3, 24 / mu**4])
    ref = mg1.run().w

    assert np.allclose(w, ref, rtol=1e-6)


def test_run_uses_exact_w_at_a1():
    """run()'s res.w must equal get_w() exactly when a=1."""
    calc = BulkServiceMM1Calc(a=1, b=4, queue_truncation=150)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w()[0])


def test_get_n_moments_are_consistent_with_run():
    """E[N] from get_n_moments must match E[V]*lambda (Little's law, exact regardless of a)."""
    lam = 1.0
    calc = BulkServiceMM1Calc(a=2, b=4, queue_truncation=150)
    calc.set_sources(lam)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    res = calc.run()
    n_moments = calc.get_n_moments(num=2)
    assert np.isclose(n_moments[0], res.v[0] * lam)


def test_get_n_moments_variance_is_nonnegative_and_monotone_in_second_moment():
    calc = BulkServiceMM1Calc(a=1, b=4, queue_truncation=150)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.2 + 0.07 * size))
    m1, m2 = calc.get_n_moments(num=2)
    assert m2 >= m1**2


if __name__ == "__main__":
    test_get_w_matches_independent_monte_carlo_at_a1()
    test_get_w_rejects_a_greater_than_one()
    test_b1_get_w_reduces_exactly_to_mm1()
    test_run_uses_exact_w_at_a1()
    test_get_n_moments_variance_is_nonnegative_and_monotone_in_second_moment()
    print("all bulk-service moment tests passed")
