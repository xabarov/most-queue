"""
Unit tests for Markovian abandonment (EPIC-070) on the multi-channel batch-service
queue (most_queue.theory.batch.bulk_service_multiserver.BulkServiceMultiserverCalc).

This is the first of three "combination" reserve items flagged after EPIC-068/069
(impatience + multiple channels, batch-size-independent mu) -- see
docs/epics/EPIC-070-bulk-service-multiserver-impatience.md for why this combination
needed no new construction: the aggregate "next completion" rate among all `c` busy
channels is always exactly `c*mu` (EPIC-069's own finding), so the tagged-customer
wait reduces to `most_queue.theory.batch._idle_refill.abandonment_chain`'s own
"ahead" segment -- which already natively supports `gamma>0` (built that way in
EPIC-068). Threading the real `gamma` through instead of fixing it at 0.0 is the
entire extension.
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service_multiserver import BulkServiceMultiserverCalc


def _full_system_des(a, b, c, lam, mu, gamma, total_customers=1_200_000, warmup=10_000, seed=7):
    """Independent from-scratch DES: c servers, shared FCFS queue, each waiting customer
    independently reneges at rate gamma. Wait = dispatch time minus arrival time (not
    batch completion time -- EPIC-068 found that exact confusion is an easy, silent bug)."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[list[float]] = []  # [arrival_time, abandon_time]
    busy: list[list] = []  # [done_time, dispatch_time, batch]
    n_generated = 0
    served_waits: list[float] = []
    n_abandoned = 0

    def maybe_dispatch():
        while len(busy) < c and len(queue) >= a:
            take = min(b, len(queue))
            batch = queue[:take]
            del queue[:take]
            busy.append([t + rng.exponential(1.0 / mu), t, batch])

    while n_generated < total_customers + warmup:
        next_abandon_t, next_abandon_idx = inf, None
        for idx_c, (_, aband) in enumerate(queue):
            if aband < next_abandon_t:
                next_abandon_t, next_abandon_idx = aband, idx_c
        next_service_done = min((x[0] for x in busy), default=inf)
        t = min(next_arrival, next_service_done, next_abandon_t)
        if t == next_arrival:
            aband_t = t + rng.exponential(1.0 / gamma) if gamma > 0 else inf
            queue.append([t, aband_t])
            next_arrival = t + rng.exponential(1 / lam)
            n_generated += 1
            maybe_dispatch()
        elif t == next_service_done:
            idx = min(range(len(busy)), key=lambda i: busy[i][0])
            _, dispatch_t, batch = busy.pop(idx)
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
        BulkServiceMultiserverCalc(a=1, b=4, c=2, gamma=-0.1)


def test_gamma_zero_matches_no_abandonment_baseline():
    a, b, c, lam, mu = 2, 4, 3, 1.0, 0.5
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150, gamma=0.0)
    calc.set_sources(lam)
    calc.set_servers(mu)
    assert calc.get_abandonment_prob() == 0.0

    ref = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150)
    ref.set_sources(lam)
    ref.set_servers(mu)
    assert np.allclose(calc.get_w(num=3), ref.get_w(num=3), rtol=1e-9)
    assert calc.get_tail(1.0) == pytest.approx(ref.get_tail(1.0), rel=1e-9)


@pytest.mark.parametrize("a,b,c", [(1, 4, 2), (2, 4, 3)])
def test_get_abandonment_prob_matches_independent_des(a, b, c):
    lam, mu, gamma = 1.0, 0.5, 0.3
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    theory_p = calc.get_abandonment_prob()

    des = _full_system_des(a, b, c, lam, mu, gamma)
    n_total = des["n_served"] + des["n_abandoned"]
    se = (des["p_abandon"] * (1 - des["p_abandon"]) / n_total) ** 0.5
    assert abs(theory_p - des["p_abandon"]) < 8 * se + 1e-3


@pytest.mark.parametrize("a,b,c", [(1, 4, 2), (2, 4, 3)])
def test_get_w_conditional_on_served_matches_independent_des(a, b, c):
    lam, mu, gamma = 1.0, 0.5, 0.3
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    theory_w = calc.get_w(num=1)[0]

    des = _full_system_des(a, b, c, lam, mu, gamma)
    assert np.isclose(theory_w, des["mean_w_given_served"], rtol=0.08)


def test_get_tail_with_abandonment_matches_independent_des():
    a, b, c, lam, mu, gamma = 2, 4, 3, 1.0, 0.5, 0.3
    calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    p_served = 1.0 - calc.get_abandonment_prob()

    des = _full_system_des(a, b, c, lam, mu, gamma)
    n_total = des["n_served"] + des["n_abandoned"]

    for t in (0.5, 1.0, 2.0):
        theory_joint = calc.get_tail(t) * p_served
        des_joint = float((des["served_waits"] > t).sum()) / n_total
        se = (des_joint * (1 - des_joint) / n_total) ** 0.5
        assert abs(theory_joint - des_joint) < 6 * se + 1e-3


def test_abandonment_prob_is_monotone_increasing_in_gamma():
    a, b, c, lam, mu = 2, 4, 3, 1.0, 0.5
    prev = 0.0
    for gamma in (0.0, 0.1, 0.3, 0.5, 1.0):
        calc = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=150, gamma=gamma)
        calc.set_sources(lam)
        calc.set_servers(mu)
        p = calc.get_abandonment_prob()
        assert p >= prev - 1e-9
        prev = p
    assert prev > 0.0


def test_get_abandonment_prob_requires_constant_mu():
    calc = BulkServiceMultiserverCalc(a=1, b=4, c=2, queue_truncation=100, gamma=0.3)
    calc.set_sources(1.0)
    calc.set_servers(lambda size: 1.0 / (0.3 + 0.1 * size))
    with pytest.raises(NotImplementedError):
        calc.get_abandonment_prob()


def test_run_uses_exact_w_with_abandonment():
    calc = BulkServiceMultiserverCalc(a=2, b=4, c=3, queue_truncation=150, gamma=0.3)
    calc.set_sources(1.0)
    calc.set_servers(0.5)
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w()[0])


if __name__ == "__main__":
    test_invalid_gamma_rejected()
    test_gamma_zero_matches_no_abandonment_baseline()
    for params in [(1, 4, 2), (2, 4, 3)]:
        test_get_abandonment_prob_matches_independent_des(*params)
        test_get_w_conditional_on_served_matches_independent_des(*params)
    test_get_tail_with_abandonment_matches_independent_des()
    test_abandonment_prob_is_monotone_increasing_in_gamma()
    test_get_abandonment_prob_requires_constant_mu()
    test_run_uses_exact_w_with_abandonment()
    print("all bulk-service multiserver abandonment tests passed")
