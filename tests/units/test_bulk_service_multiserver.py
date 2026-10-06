"""
Unit tests for M/M^[a,b]/c bulk-service (c identical servers, shared FCFS queue)
(most_queue.theory.batch.bulk_service_multiserver, EPIC-069).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_multiserver import BulkServiceMultiserverCalc


def _des_mean_n(a, b, c, lam, mu_fn, total_time=300_000.0, warmup=5_000.0, seed=1):
    """Independent from-scratch DES (not reusing BulkServiceSim): time-average E[N]."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    j = 0
    busy = []  # [done_time, size]
    area = 0.0
    last_t = 0.0

    def n_in_system():
        return j + sum(sz for _, sz in busy)

    def maybe_dispatch():
        nonlocal j
        while len(busy) < c and j >= a:
            take = min(b, j)
            j -= take
            busy.append([t + rng.exponential(1.0 / mu_fn(take)), take])

    while t < total_time:
        next_done = min((x[0] for x in busy), default=inf)
        t_next = min(next_arrival, next_done)
        if t_next > last_t:
            w = max(last_t, warmup)
            if t_next > w:
                area += n_in_system() * (t_next - w)
        t = t_next
        last_t = t
        if t == next_arrival:
            j += 1
            next_arrival = t + rng.exponential(1 / lam)
            maybe_dispatch()
        else:
            idx = min(range(len(busy)), key=lambda i: busy[i][0])
            busy.pop(idx)
            maybe_dispatch()

    return area / (total_time - warmup)


def _des_w_samples(a, b, c, lam, mu, total_customers=800_000, warmup=10_000, seed=5):
    """Independent from-scratch DES: wait = DISPATCH time minus arrival time (not batch
    completion time -- EPIC-068 found that exact confusion is an easy, silent bug)."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    busy: list[list] = []  # [done_time, dispatch_time, batch]
    n_generated = 0
    waits = []

    def maybe_dispatch():
        while len(busy) < c and len(queue) >= a:
            take = min(b, len(queue))
            batch = queue[:take]
            del queue[:take]
            busy.append([t + rng.exponential(1.0 / mu), t, batch])

    while n_generated < total_customers + warmup:
        next_done = min((x[0] for x in busy), default=inf)
        t = min(next_arrival, next_done)
        if t == next_arrival:
            queue.append(t)
            next_arrival = t + rng.exponential(1 / lam)
            n_generated += 1
            maybe_dispatch()
        else:
            idx = min(range(len(busy)), key=lambda i: busy[i][0])
            _, dispatch_t, batch = busy.pop(idx)
            for arr in batch:
                if n_generated > warmup:
                    waits.append(dispatch_t - arr)
            maybe_dispatch()
    return np.array(waits)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BulkServiceMultiserverCalc(a=4, b=1, c=2)
    with pytest.raises(ValueError):
        BulkServiceMultiserverCalc(a=1, b=4, c=0)


@pytest.mark.parametrize("a,b,lam,mu", [(1, 4, 0.6, 1.5), (2, 4, 0.6, 1.5), (3, 5, 0.3, 1.0)])
def test_c1_reduces_exactly_to_mm1(a, b, lam, mu):
    """c=1: must exactly match the already-validated BulkServiceMM1Calc (pi, W moments, tail)."""
    ms = BulkServiceMultiserverCalc(a=a, b=b, c=1, queue_truncation=200)
    ms.set_sources(lam)
    ms.set_servers(mu)

    ref = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200)
    ref.set_sources(lam)
    ref.set_servers(mu)

    assert np.allclose(ms.get_n_moments(num=2), ref.get_n_moments(num=2), rtol=1e-9)
    assert np.allclose(ms.get_w(num=3), ref.get_w(num=3), rtol=1e-9)
    for t in (0.2, 1.0, 2.0):
        assert ms.get_tail(t) == pytest.approx(ref.get_tail(t), rel=1e-9)


@pytest.mark.parametrize("a,b,c", [(1, 4, 2), (2, 4, 3), (1, 2, 4)])
def test_get_n_moments_matches_independent_des(a, b, c):
    lam, mu = 0.8, 1.0
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150)
    calc.set_sources(lam)
    calc.set_servers(mu)
    e_n_theory = calc.get_n_moments(num=1)[0]

    e_n_des = _des_mean_n(a, b, c, lam, lambda _s, _mu=mu: _mu)
    assert np.isclose(e_n_theory, e_n_des, rtol=0.03)


def test_get_n_moments_supports_batch_size_dependent_mu():
    """pi/get_n_moments support batch-size-dependent mu even though get_w/get_tail don't yet."""
    a, b, c, lam = 1, 4, 2, 0.8

    def mu_fn(size):
        return 1.0 / (0.3 + 0.1 * size)

    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150)
    calc.set_sources(lam)
    calc.set_servers(mu_fn)
    e_n_theory = calc.get_n_moments(num=1)[0]

    e_n_des = _des_mean_n(a, b, c, lam, mu_fn)
    assert np.isclose(e_n_theory, e_n_des, rtol=0.04)


@pytest.mark.parametrize("a,b,c", [(1, 4, 2), (2, 4, 3)])
def test_get_w_matches_independent_des(a, b, c):
    lam, mu = {(1, 4, 2): 0.8, (2, 4, 3): 1.0}[(a, b, c)], {(1, 4, 2): 1.0, (2, 4, 3): 1.2}[(a, b, c)]
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150)
    calc.set_sources(lam)
    calc.set_servers(mu)
    w_theory = calc.get_w(num=2)

    waits = _des_w_samples(a, b, c, lam, mu, total_customers=1_200_000)
    w_des = [float((waits**k).mean()) for k in (1, 2)]
    assert np.allclose(w_theory, w_des, rtol=0.05)


@pytest.mark.parametrize("a,b,c", [(1, 4, 2), (2, 4, 3)])
def test_get_tail_matches_independent_des(a, b, c):
    lam, mu = {(1, 4, 2): 0.8, (2, 4, 3): 1.0}[(a, b, c)], {(1, 4, 2): 1.0, (2, 4, 3): 1.2}[(a, b, c)]
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150)
    calc.set_sources(lam)
    calc.set_servers(mu)

    waits = _des_w_samples(a, b, c, lam, mu, total_customers=1_200_000, seed=11)
    for t in (0.2, 0.5, 1.0, 2.0):
        exact = calc.get_tail(t)
        des = float((waits > t).mean())
        se = (des * (1 - des) / len(waits)) ** 0.5
        assert abs(exact - des) < 6 * se + 1e-4


def test_get_cdf_is_one_minus_tail():
    calc = BulkServiceMultiserverCalc(a=1, b=4, c=2, queue_truncation=150)
    calc.set_sources(0.8)
    calc.set_servers(1.0)
    for t in (0.3, 1.0, 3.0):
        assert calc.get_cdf(t) == pytest.approx(1.0 - calc.get_tail(t))


def test_get_w_and_get_tail_reject_batch_size_dependent_mu():
    calc = BulkServiceMultiserverCalc(a=1, b=4, c=2, queue_truncation=100)
    calc.set_sources(0.8)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.1 * size))
    with pytest.raises(NotImplementedError):
        calc.get_w()
    with pytest.raises(NotImplementedError):
        calc.get_tail(1.0)


def test_run_uses_exact_w_for_constant_mu():
    calc = BulkServiceMultiserverCalc(a=2, b=4, c=3, queue_truncation=150)
    calc.set_sources(1.0)
    calc.set_servers(1.2)
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w()[0])
    assert 0.0 < res.utilization < 1.0


def test_more_servers_reduces_mean_wait():
    """Sanity: adding servers (same per-server rate) should not increase E[W]."""
    a, b, lam, mu = 2, 4, 1.0, 1.2
    prev_w = None
    for c in (2, 3, 4):
        calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150)
        calc.set_sources(lam)
        calc.set_servers(mu)
        w = calc.get_w(num=1)[0]
        if prev_w is not None:
            assert w <= prev_w + 1e-9
        prev_w = w


if __name__ == "__main__":
    test_invalid_params_rejected()
    for params in [(1, 4, 0.6, 1.5), (2, 4, 0.6, 1.5), (3, 5, 0.3, 1.0)]:
        test_c1_reduces_exactly_to_mm1(*params)
    for params in [(1, 4, 2), (2, 4, 3), (1, 2, 4)]:
        test_get_n_moments_matches_independent_des(*params)
    test_get_n_moments_supports_batch_size_dependent_mu()
    for params in [(1, 4, 2), (2, 4, 3)]:
        test_get_w_matches_independent_des(*params)
        test_get_tail_matches_independent_des(*params)
    test_get_cdf_is_one_minus_tail()
    test_get_w_and_get_tail_reject_batch_size_dependent_mu()
    test_run_uses_exact_w_for_constant_mu()
    test_more_servers_reduces_mean_wait()
    print("all bulk-service multiserver tests passed")
