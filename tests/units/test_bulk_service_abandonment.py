"""
Unit tests for Markovian abandonment (EPIC-068) on the batch-size-dependent
M/M^[a,b]/1 bulk-service queue (most_queue.theory.batch.bulk_service).

Each of the ``j`` currently-waiting customers independently reneges at rate
``gamma`` (same convention as ``most_queue.theory.impatience.mm1.MM1Impatience``).
Validated against an independent, from-scratch DES that tracks each
customer's own WAIT (batch-START time minus arrival time, not batch
completion/sojourn -- conflating the two was a real bug found while deriving
this feature, see docs/epics/EPIC-068-bulk-service-impatience.md) and
whether they abandoned before their own batch started.
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc


def _full_system_des(a, b, lam, mu, gamma, total_customers=1_000_000, warmup=20_000, seed=2024):
    """Independent from-scratch DES (not reusing BulkServiceSim): Poisson arrivals, FCFS
    batch formation, each waiting customer independently abandons at rate gamma. Resolves
    customers by ARRIVAL INDEX (not wall-clock cutoff) so neither warmup nor end-of-run
    censoring biases the sample."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[list[float]] = []  # [arrival_time, abandon_time]
    server_busy = False
    server_done = inf
    batch: list[list[float]] = []
    batch_start = 0.0
    n_generated = 0
    served_waits = []
    n_abandoned = 0

    def maybe_start():
        nonlocal server_busy, server_done, batch, batch_start
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = queue[:take]
            del queue[:take]
            server_busy = True
            batch_start = t
            server_done = t + rng.exponential(1.0 / mu(take))

    while n_generated < total_customers + warmup:
        next_abandon_t, next_abandon_idx = inf, None
        for idx_c, (_, aband) in enumerate(queue):
            if aband < next_abandon_t:
                next_abandon_t, next_abandon_idx = aband, idx_c
        t = min(next_arrival, server_done, next_abandon_t)
        if t == next_arrival:
            aband_t = t + rng.exponential(1.0 / gamma) if gamma > 0 else inf
            queue.append([t, aband_t])
            next_arrival = t + rng.exponential(1 / lam)
            n_generated += 1
            maybe_start()
        elif t == server_done:
            for arr, _ in batch:
                if n_generated > warmup:
                    served_waits.append(batch_start - arr)
            server_busy = False
            server_done = inf
            batch = []
            maybe_start()
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
        BulkServiceMM1Calc(a=1, b=4, gamma=-0.1)


def test_gamma_zero_matches_no_abandonment_baseline():
    """gamma=0 must reduce exactly to the pre-EPIC-068 unconditional W/tail."""
    a, b, lam, mu = 2, 4, 0.6, 1.5
    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=0.0)
    calc.set_sources(lam)
    calc.set_servers(mu)
    assert calc.get_abandonment_prob() == 0.0

    ref = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200)
    ref.set_sources(lam)
    ref.set_servers(mu)
    assert np.allclose(calc.get_w(num=3), ref.get_w(num=3), rtol=1e-9)
    assert calc.get_tail(1.0) == pytest.approx(ref.get_tail(1.0), rel=1e-9)


@pytest.mark.parametrize("a,b", [(1, 4), (2, 4)])
def test_get_abandonment_prob_matches_independent_des(a, b):
    lam, mu, gamma = 0.6, 1.5, 0.4
    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    theory_p = calc.get_abandonment_prob()

    des = _full_system_des(a, b, lam, lambda _i, _mu=mu: _mu, gamma, total_customers=1_500_000)
    n_total = des["n_served"] + des["n_abandoned"]
    se = (des["p_abandon"] * (1 - des["p_abandon"]) / n_total) ** 0.5
    assert abs(theory_p - des["p_abandon"]) < 8 * se + 1e-3


@pytest.mark.parametrize("a,b", [(1, 4), (2, 4)])
def test_get_w_conditional_on_served_matches_independent_des(a, b):
    lam, mu, gamma = 0.6, 1.5, 0.4
    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    theory_w = calc.get_w(num=1)[0]

    des = _full_system_des(a, b, lam, lambda _i, _mu=mu: _mu, gamma, total_customers=1_500_000)
    assert np.isclose(theory_w, des["mean_w_given_served"], rtol=0.08)


def test_get_tail_with_abandonment_matches_independent_des():
    """get_tail(t) is P(W>t | served); cross-check the JOINT P(W>t AND served) = get_tail(t)
    * P(served) against the DES's own joint frequency (served AND wait>t among ALL resolved
    customers), which has a simple binomial SE."""
    a, b, lam, mu, gamma = 2, 4, 0.6, 1.5, 0.4
    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    p_served = 1.0 - calc.get_abandonment_prob()

    des = _full_system_des(a, b, lam, lambda _i, _mu=mu: _mu, gamma, total_customers=1_500_000, seed=11)
    n_total = des["n_served"] + des["n_abandoned"]

    for t in (0.2, 1.0, 2.0):
        theory_joint = calc.get_tail(t) * p_served
        des_joint = float((des["served_waits"] > t).sum()) / n_total
        se = (des_joint * (1 - des_joint) / n_total) ** 0.5
        assert abs(theory_joint - des_joint) < 6 * se + 1e-3


def test_abandonment_prob_is_monotone_increasing_in_gamma():
    a, b, lam, mu = 2, 4, 0.6, 1.5
    prev = 0.0
    for gamma in (0.0, 0.2, 0.5, 1.0, 2.0):
        calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=gamma)
        calc.set_sources(lam)
        calc.set_servers(mu)
        p = calc.get_abandonment_prob()
        assert p >= prev - 1e-9
        prev = p
    assert prev > 0.0


def test_abandonment_prob_zero_at_j_equals_a_minus_1_idle_state():
    """W=0 deterministically for the idle state one arrival short of threshold a --
    no time to abandon, so this state must never contribute to p_abandon."""
    a, b, lam, mu, gamma = 3, 5, 0.3, 1.0, 1.0
    calc = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(mu)
    pi = calc._solve_pi()  # pylint: disable=protected-access
    idle_at_threshold = pi[calc._index(0, a - 1)]  # pylint: disable=protected-access
    assert idle_at_threshold > 0  # sanity: state is actually reachable


if __name__ == "__main__":
    test_invalid_gamma_rejected()
    test_gamma_zero_matches_no_abandonment_baseline()
    test_get_abandonment_prob_matches_independent_des(2, 4)
    test_get_w_conditional_on_served_matches_independent_des(2, 4)
    test_get_tail_with_abandonment_matches_independent_des()
    test_abandonment_prob_is_monotone_increasing_in_gamma()
    test_abandonment_prob_zero_at_j_equals_a_minus_1_idle_state()
    print("all bulk-service abandonment tests passed")
