"""
Unit tests for batch-size-DEPENDENT mu(size) on the multi-channel batch-service
queue (most_queue.theory.batch.bulk_service_multiserver.BulkServiceMultiserverCalc).

This is the third and last of three "combination" reserve items flagged after
EPIC-068/069/070/071 -- see docs/epics/EPIC-072-bulk-service-multiserver-phase-type.md.
Unlike EPIC-070 (pure reuse), batch-size-dependent mu genuinely breaks EPIC-069's
"aggregate rate c*mu" collapse, so the tagged-customer wait needs the occupancy
vector tracked INSIDE the absorbing chain (`_occupancy_busy_chain`) -- but only
while all `c` servers stay busy; the idle-refill race turned out to stay
occupancy-free, keeping the actual state space far smaller than the "full
combinatorial blowup" flagged as a risk when the epic was proposed.
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service_multiserver import BulkServiceMultiserverCalc


def _full_system_des(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    a, b, c, lam, mu_fn, total_customers=900_000, warmup=15_000, seed=5
):
    """Independent from-scratch DES: c servers, shared FCFS queue, batch-size-dependent
    service rate. Wait = dispatch time minus arrival time (not batch completion time --
    EPIC-068 found that exact confusion is an easy, silent bug)."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    busy: list[list] = []  # [done_time, dispatch_time, batch]
    n_generated = 0
    served_waits: list[float] = []

    def maybe_dispatch():
        while len(busy) < c and len(queue) >= a:
            take = min(b, len(queue))
            batch = queue[:take]
            del queue[:take]
            svc = rng.exponential(1.0 / mu_fn(take))
            busy.append([t + svc, t, batch])

    while n_generated < total_customers + warmup:
        next_service_done = min((x[0] for x in busy), default=inf)
        t = min(next_arrival, next_service_done)
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
                    served_waits.append(dispatch_t - arr)
            maybe_dispatch()
    return np.array(served_waits)


def test_constant_callable_mu_matches_scalar_regression():
    """A callable mu(size) that happens to be constant must exactly reproduce the
    scalar-mu (EPIC-069) result, even though it's routed through the NEW dependent-mu
    code path (`_is_constant_mu` is only set for a bare scalar, not a callable)."""
    a, b, c, lam, mu = 2, 4, 3, 1.0, 0.5
    ref = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=100)
    ref.set_sources(lam)
    ref.set_servers(mu)

    dep = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=100)
    dep.set_sources(lam)
    dep.set_servers(lambda _size, _mu=mu: _mu)
    assert not dep._is_constant_mu  # pylint: disable=protected-access

    assert np.allclose(dep.get_w(num=3), ref.get_w(num=3), rtol=1e-9)
    for t in (0.5, 1.0, 2.0):
        assert dep.get_tail(t) == pytest.approx(ref.get_tail(t), rel=1e-9)


@pytest.mark.parametrize("a,b,c", [(2, 4, 2), (1, 3, 3)])
def test_get_w_matches_independent_des(a, b, c):
    lam = 1.0 if (a, b, c) == (2, 4, 2) else 1.2

    def mu_fn(size):
        return 1.0 / (0.3 + 0.25 * size)

    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=80)
    calc.set_sources(lam)
    calc.set_servers(mu_fn)
    theory_w = calc.get_w(num=1)[0]

    waits = _full_system_des(a, b, c, lam, mu_fn)
    assert np.isclose(theory_w, waits.mean(), rtol=0.05)


@pytest.mark.parametrize("a,b,c", [(2, 4, 2), (1, 3, 3)])
def test_get_tail_matches_independent_des(a, b, c):
    lam = 1.0 if (a, b, c) == (2, 4, 2) else 1.2

    def mu_fn(size):
        return 1.0 / (0.3 + 0.25 * size)

    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=80)
    calc.set_sources(lam)
    calc.set_servers(mu_fn)

    waits = _full_system_des(a, b, c, lam, mu_fn)
    n = len(waits)
    for t in (0.5, 1.0, 2.0):
        theory_tail = calc.get_tail(t)
        des_tail = float((waits > t).sum()) / n
        se = (des_tail * (1 - des_tail) / n) ** 0.5
        assert abs(theory_tail - des_tail) < 6 * se + 1e-3


def test_get_n_moments_exact_for_dependent_mu_unaffected():
    """get_n_moments() was already exact for any mu before EPIC-072 (direct
    summation over pi) -- confirm it's untouched."""
    calc = BulkServiceMultiserverCalc(a=2, b=4, c=2, queue_truncation=80)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.25 * size))
    moments = calc.get_n_moments(num=2)
    assert moments[0] > 0
    assert moments[1] >= moments[0] ** 2


def test_dependent_mu_with_gamma_still_raises():
    """The remaining reserve: batch-size-dependent mu COMBINED with gamma > 0
    abandonment -- EPIC-072 explicitly scopes this out (no R-abandonment
    transitions in _occupancy_busy_chain)."""
    calc = BulkServiceMultiserverCalc(a=2, b=4, c=2, queue_truncation=80, gamma=0.3)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.25 * size))
    with pytest.raises(NotImplementedError):
        calc.get_w()
    with pytest.raises(NotImplementedError):
        calc.get_tail(1.0)
    with pytest.raises(NotImplementedError):
        calc.get_abandonment_prob()


def test_run_uses_dependent_mu_w_at_gamma_zero():
    calc = BulkServiceMultiserverCalc(a=2, b=4, c=2, queue_truncation=80)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.25 * size))
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w()[0])


if __name__ == "__main__":
    test_constant_callable_mu_matches_scalar_regression()
    for params in [(2, 4, 2), (1, 3, 3)]:
        test_get_w_matches_independent_des(*params)
        test_get_tail_matches_independent_des(*params)
    test_get_n_moments_exact_for_dependent_mu_unaffected()
    test_dependent_mu_with_gamma_still_raises()
    test_run_uses_dependent_mu_w_at_gamma_zero()
    print("all bulk-service multiserver dependent-mu tests passed")
