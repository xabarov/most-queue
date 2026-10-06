"""
Unit tests for Markovian abandonment (EPIC-068) on the Erlang-service
M/Erlang(k,rate)^[a,b]/1 bulk-service queue (most_queue.theory.batch.bulk_service_erlang).

The strongest, zero-noise check is k=1 (Erlang collapses to Exponential):
``get_abandonment_prob``/``get_w``/``get_tail`` must then match
``BulkServiceMM1Calc`` to full float64 precision, exactly like every other
EPIC-035/042/043/067 regression in this module. k>1 is cross-checked against
an independent DES; full-system aggregate DES runs are noisier here than for
the plain-exponential case (serial correlation between consecutive customers'
waits inflates the apparent MC error well beyond the naive binomial/iid
estimate -- confirmed by an isolated single-tagged-customer Monte Carlo that
matches the exact per-state formula far more tightly), so tolerances are
looser than ``test_bulk_service_abandonment.py``'s.
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc


def _full_system_des_erlang(a, b, lam, rate, k, gamma, total_customers=1_000_000, warmup=20_000, seed=7):
    """Independent from-scratch DES: Poisson arrivals, FCFS batch formation, Erlang(k,rate)
    batch-service time, each waiting customer independently reneges at rate gamma."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[list[float]] = []
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
            server_done = t + rng.gamma(k, 1.0 / rate)

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
    }


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        BulkServiceErlangCalc(a=1, b=4, k=2, gamma=-0.1)


def test_gamma_zero_matches_no_abandonment_baseline():
    a, b, k, lam, rate = 2, 4, 2, 0.6, 1.5
    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=150, gamma=0.0)
    calc.set_sources(lam)
    calc.set_servers(rate)
    assert calc.get_abandonment_prob() == 0.0

    ref = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=150)
    ref.set_sources(lam)
    ref.set_servers(rate)
    assert np.allclose(calc.get_w(num=3), ref.get_w(num=3), rtol=1e-9)
    assert calc.get_tail(1.0) == pytest.approx(ref.get_tail(1.0), rel=1e-9)


@pytest.mark.parametrize("a,b", [(1, 4), (2, 4), (3, 5)])
def test_k1_reduces_exactly_to_mm1_abandonment(a, b):
    """k=1 (Erlang -> Exp): gamma-aware get_abandonment_prob/get_w/get_tail must match
    BulkServiceMM1Calc's own (independently DES-validated) gamma-aware results exactly --
    both implementations share the same abandonment_chain construction but through
    different call sites, so this is a strong, zero-MC-noise regression."""
    lam, rate, gamma = 0.6, 1.5, 0.4
    calc = BulkServiceErlangCalc(a=a, b=b, k=1, queue_truncation=200, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(rate)

    ref = BulkServiceMM1Calc(a=a, b=b, queue_truncation=200, gamma=gamma)
    ref.set_sources(lam)
    ref.set_servers(rate)

    assert calc.get_abandonment_prob() == pytest.approx(ref.get_abandonment_prob(), rel=1e-9)
    assert np.allclose(calc.get_w(num=3), ref.get_w(num=3), rtol=1e-9)
    for t in (0.2, 1.0, 2.0):
        assert calc.get_tail(t) == pytest.approx(ref.get_tail(t), rel=1e-9)


@pytest.mark.parametrize("k", [2, 3])
def test_get_abandonment_prob_matches_independent_des(k):
    a, b, lam, rate, gamma = 2, 4, 0.6, 1.5, 0.4
    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(rate)
    theory_p = calc.get_abandonment_prob()

    des = _full_system_des_erlang(a, b, lam, rate, k, gamma, total_customers=1_500_000)
    n_total = des["n_served"] + des["n_abandoned"]
    se = (des["p_abandon"] * (1 - des["p_abandon"]) / n_total) ** 0.5
    # serial correlation across consecutive customers inflates the effective SE well
    # beyond the naive iid estimate (see module docstring); widen generously.
    assert abs(theory_p - des["p_abandon"]) < 20 * se + 2e-2


@pytest.mark.parametrize("k", [2, 3])
def test_get_w_conditional_on_served_matches_independent_des(k):
    a, b, lam, rate, gamma = 2, 4, 0.6, 1.5, 0.4
    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(rate)
    theory_w = calc.get_w(num=1)[0]

    des = _full_system_des_erlang(a, b, lam, rate, k, gamma, total_customers=1_500_000)
    assert np.isclose(theory_w, des["mean_w_given_served"], rtol=0.12)


def test_abandonment_prob_is_monotone_increasing_in_gamma():
    a, b, k, lam, rate = 2, 4, 2, 0.6, 1.5
    prev = 0.0
    for gamma in (0.0, 0.2, 0.5, 1.0, 2.0):
        calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=150, gamma=gamma)
        calc.set_sources(lam)
        calc.set_servers(rate)
        p = calc.get_abandonment_prob()
        assert p >= prev - 1e-9
        prev = p
    assert prev > 0.0


if __name__ == "__main__":
    test_invalid_gamma_rejected()
    test_gamma_zero_matches_no_abandonment_baseline()
    for ab_pair in ((1, 4), (2, 4), (3, 5)):
        test_k1_reduces_exactly_to_mm1_abandonment(*ab_pair)
    for k_val in (2, 3):
        test_get_abandonment_prob_matches_independent_des(k_val)
        test_get_w_conditional_on_served_matches_independent_des(k_val)
    test_abandonment_prob_is_monotone_increasing_in_gamma()
    print("all bulk-service Erlang abandonment tests passed")
