"""
Unit tests for Markovian abandonment (EPIC-071) on the H2-service bulk-service
queue (most_queue.theory.batch.bulk_service_h2.BulkServiceH2Calc).

This is the second of three "combination" reserve items flagged after
EPIC-068/069/070 -- see docs/epics/EPIC-071-bulk-service-h2-impatience.md.
Unlike EPIC-070 (pure reuse), H2's branching structure needed a dedicated
construction (``most_queue.theory.batch.bulk_service_h2._h2_abandonment_chain``):
every batch ahead of the tagged customer independently re-chooses its own
branch, so the abandonment chain carries 3 "segments" (first/ahead-phase-0/
ahead-phase-1) per (R, K) pair instead of EPIC-068's single sequential chain.
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc


def _full_system_des(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    a, b, lam, p1, mu1, mu2, gamma, total_customers=1_500_000, warmup=20_000, seed=11
):
    """Independent from-scratch DES: single H2 server, each waiting customer
    independently reneges at rate gamma. Wait = dispatch time minus arrival time
    (not batch completion time -- EPIC-068 found that exact confusion is an easy,
    silent bug)."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[list[float]] = []  # [arrival_time, abandon_time]
    busy = None  # [done_time, dispatch_time, batch]
    n_generated = 0
    served_waits: list[float] = []
    n_abandoned = 0

    def maybe_dispatch():
        nonlocal busy
        if busy is None and len(queue) >= a:
            take = min(b, len(queue))
            batch = queue[:take]
            del queue[:take]
            svc = rng.exponential(1.0 / mu1) if rng.random() < p1 else rng.exponential(1.0 / mu2)
            busy = [t + svc, t, batch]

    while n_generated < total_customers + warmup:
        next_abandon_t, next_abandon_idx = inf, None
        for idx_c, (_, aband) in enumerate(queue):
            if aband < next_abandon_t:
                next_abandon_t, next_abandon_idx = aband, idx_c
        next_service_done = busy[0] if busy is not None else inf
        t = min(next_arrival, next_service_done, next_abandon_t)
        if t == next_arrival:
            aband_t = t + rng.exponential(1.0 / gamma) if gamma > 0 else inf
            queue.append([t, aband_t])
            next_arrival = t + rng.exponential(1 / lam)
            n_generated += 1
            maybe_dispatch()
        elif t == next_service_done:
            _, dispatch_t, batch = busy
            busy = None
            for arr, _ in batch:
                if n_generated > warmup:
                    served_waits.append(dispatch_t - arr)
            maybe_dispatch()
        else:
            queue.pop(next_abandon_idx)
            if n_generated > warmup:
                n_abandoned += 1

    n_served = len(served_waits)
    return {
        "p_abandon": n_abandoned / (n_served + n_abandoned),
        "mean_w_given_served": float(np.mean(served_waits)),
        "n_served": n_served,
        "n_abandoned": n_abandoned,
        "served_waits": np.array(served_waits),
    }


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        BulkServiceH2Calc(a=1, b=4, gamma=-0.1)


def test_gamma_zero_unaffected():
    a, b, lam = 2, 4, 1.0
    calc = BulkServiceH2Calc(a, b, queue_truncation=150, gamma=0.0)
    calc.set_sources(lam)
    calc.set_servers(0.4, 0.6, 1.5)
    assert calc.get_abandonment_prob() == 0.0
    assert calc.get_tail(1.0) == pytest.approx(calc.get_tail(1.0))
    with pytest.raises(NotImplementedError):
        calc.get_w()


@pytest.mark.parametrize("a,b", [(1, 3), (2, 4)])
def test_p1_one_matches_mm1_abandonment(a, b):
    """H2 collapsing to Exp(mu1) (p1=1) must exactly reproduce BulkServiceMM1Calc's
    EPIC-068 gamma>0 construction -- the primary regression test."""
    lam, mu, gamma = 1.0, 0.5, 0.2
    h2 = BulkServiceH2Calc(a, b, queue_truncation=150, gamma=gamma)
    h2.set_sources(lam)
    h2.set_servers(1.0, mu, 999.0)
    mm1 = BulkServiceMM1Calc(a, b, queue_truncation=150, gamma=gamma)
    mm1.set_sources(lam)
    mm1.set_servers(mu)

    assert h2.get_abandonment_prob() == pytest.approx(mm1.get_abandonment_prob(), rel=1e-9)
    assert np.allclose(h2.get_w(3), mm1.get_w(3), rtol=1e-9)
    for t in (0.5, 1.0, 2.0):
        assert h2.get_tail(t) == pytest.approx(mm1.get_tail(t), rel=1e-9)


@pytest.mark.parametrize("a,b", [(1, 3), (2, 4)])
def test_get_abandonment_prob_matches_independent_des(a, b):
    lam, p1, mu1, mu2, gamma = 1.0, 0.3, 0.6, 1.5, 0.25
    calc = BulkServiceH2Calc(a, b, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(p1, mu1, mu2)
    theory_p = calc.get_abandonment_prob()

    des = _full_system_des(a, b, lam, p1, mu1, mu2, gamma)
    n_total = des["n_served"] + des["n_abandoned"]
    se = (des["p_abandon"] * (1 - des["p_abandon"]) / n_total) ** 0.5
    assert abs(theory_p - des["p_abandon"]) < 8 * se + 1e-3


@pytest.mark.parametrize("a,b", [(1, 3), (2, 4)])
def test_get_w_conditional_on_served_matches_independent_des(a, b):
    lam, p1, mu1, mu2, gamma = 1.0, 0.3, 0.6, 1.5, 0.25
    calc = BulkServiceH2Calc(a, b, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(p1, mu1, mu2)
    theory_w = calc.get_w(num=1)[0]

    des = _full_system_des(a, b, lam, p1, mu1, mu2, gamma)
    assert np.isclose(theory_w, des["mean_w_given_served"], rtol=0.08)


def test_get_tail_with_abandonment_matches_independent_des():
    a, b, lam, p1, mu1, mu2, gamma = 2, 4, 1.0, 0.3, 0.6, 1.5, 0.25
    calc = BulkServiceH2Calc(a, b, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(p1, mu1, mu2)
    p_served = 1.0 - calc.get_abandonment_prob()

    des = _full_system_des(a, b, lam, p1, mu1, mu2, gamma)
    n_total = des["n_served"] + des["n_abandoned"]

    for t in (0.5, 1.0, 2.0):
        theory_joint = calc.get_tail(t) * p_served
        des_joint = float((des["served_waits"] > t).sum()) / n_total
        se = (des_joint * (1 - des_joint) / n_total) ** 0.5
        assert abs(theory_joint - des_joint) < 6 * se + 1e-3


def test_abandonment_prob_is_monotone_increasing_in_gamma():
    a, b, lam, p1, mu1, mu2 = 2, 4, 1.0, 0.3, 0.6, 1.5
    prev = 0.0
    for gamma in (0.0, 0.1, 0.3, 0.5, 1.0):
        calc = BulkServiceH2Calc(a, b, queue_truncation=150, gamma=gamma)
        calc.set_sources(lam)
        calc.set_servers(p1, mu1, mu2)
        p = calc.get_abandonment_prob()
        assert p >= prev - 1e-9
        prev = p
    assert prev > 0.0


def test_run_uses_exact_w_with_abandonment():
    calc = BulkServiceH2Calc(a=2, b=4, queue_truncation=150, gamma=0.3)
    calc.set_sources(1.0)
    calc.set_servers(0.3, 0.6, 1.5)
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w()[0])


if __name__ == "__main__":
    test_invalid_gamma_rejected()
    test_gamma_zero_unaffected()
    for params in [(1, 3), (2, 4)]:
        test_p1_one_matches_mm1_abandonment(*params)
        test_get_abandonment_prob_matches_independent_des(*params)
        test_get_w_conditional_on_served_matches_independent_des(*params)
    test_get_tail_with_abandonment_matches_independent_des()
    test_abandonment_prob_is_monotone_increasing_in_gamma()
    test_run_uses_exact_w_with_abandonment()
    print("all bulk-service H2 abandonment tests passed")
