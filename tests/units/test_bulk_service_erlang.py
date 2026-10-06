"""
Unit tests for M/Erlang(k,rate)^[a,b]/1 bulk-service with general
(phase-type-fitted) batch-service time (most_queue.theory.batch.bulk_service_erlang,
EPIC-035, plus batch-size-dependent params EPIC-042 and exact W moments EPIC-043).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc


def _independent_des_mean_v(a, b, k, rate, lam, total_served=400_000, warmup_fraction=0.05, seed=1):
    """From-scratch DES (not reusing BulkServiceSim) sampling Erlang(k,rate) batch-service times."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    server_busy = False
    server_done = inf
    batch: list[float] = []
    soj_sum = 0.0
    served = 0
    warmup = int(total_served * warmup_fraction)

    def maybe_start():
        nonlocal server_busy, server_done, batch
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = [queue.pop(0) for _ in range(take)]
            server_busy = True
            server_done = t + rng.gamma(k, 1 / rate)  # Erlang(k, rate)

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
                    soj_sum += t - arr
                served += 1
            server_busy = False
            server_done = inf
            batch = []
            maybe_start()
    n = max(served - warmup, 1)
    return soj_sum / n


def test_k1_reduces_exactly_to_exponential_bulk_service():
    """Erlang(1, rate) = Exp(rate): must exactly match the already-validated BulkServiceMM1Calc."""
    a, b, rate, lam = 1, 4, 1.5, 1.0

    calc = BulkServiceErlangCalc(a=a, b=b, k=1, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(rate)
    v = calc.run().v[0]

    ref = BulkServiceMM1Calc(a=a, b=b, queue_truncation=250)
    ref.set_sources(lam)
    ref.set_servers(rate)
    v_ref = ref.run().v[0]

    assert np.isclose(v, v_ref, rtol=1e-9)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_general_k_matches_independent_des(k):
    a, b, rate, lam = 1, 4, 1.5, 1.0
    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(rate)
    v = calc.run().v[0]

    v_des = _independent_des_mean_v(a, b, k, rate, lam)
    assert np.isclose(v, v_des, rtol=0.03)


def test_larger_k_same_mean_does_not_increase_wait():
    """Fixed mean batch service (k/rate constant): larger k (lower CV, more predictable) must
    not increase mean sojourn -- variability, not just the mean, drives queueing delay."""
    a, b, lam = 1, 4, 1.0
    mean_service = 2.0
    prev_v = None
    for k in (1, 2, 4, 8):
        rate = k / mean_service
        calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
        calc.set_sources(lam)
        calc.set_servers(rate)
        v = calc.run().v[0]
        if prev_v is not None:
            assert v <= prev_v + 1e-9
        prev_v = v


def _independent_des_mean_v_batch_dependent(a, b, k, rate_fn, lam, total_served=400_000, warmup_fraction=0.05, seed=1):
    """Same DES as above, but with a batch-size-dependent per-phase rate."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    server_busy = False
    server_done = inf
    batch: list[float] = []
    soj_sum = 0.0
    served = 0
    warmup = int(total_served * warmup_fraction)

    def maybe_start():
        nonlocal server_busy, server_done, batch
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = [queue.pop(0) for _ in range(take)]
            server_busy = True
            server_done = t + rng.gamma(k, 1 / rate_fn(take))

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
                    soj_sum += t - arr
                served += 1
            server_busy = False
            server_done = inf
            batch = []
            maybe_start()
    n = max(served - warmup, 1)
    return soj_sum / n


def test_batch_size_dependent_rate_matches_independent_des():
    """rate(i) = 3.0 + 0.2*i -- larger batches are (mildly) slower per phase, LLM/GPU-style."""
    a, b, k, lam = 1, 4, 3, 1.0

    def rate_fn(i):
        return 3.0 + 0.2 * i

    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(rate_fn)
    v = calc.run().v[0]

    v_des = _independent_des_mean_v_batch_dependent(a, b, k, rate_fn, lam)
    assert np.isclose(v, v_des, rtol=0.03)


def test_batch_size_dependent_rate_reduces_to_scalar_when_constant():
    a, b, k, lam, rate = 1, 4, 3, 1.0, 1.5
    calc_scalar = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
    calc_scalar.set_sources(lam)
    calc_scalar.set_servers(rate)
    v_scalar = calc_scalar.run().v[0]

    calc_fn = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
    calc_fn.set_sources(lam)
    calc_fn.set_servers(lambda _i: rate)
    v_fn = calc_fn.run().v[0]

    assert np.isclose(v_scalar, v_fn, atol=1e-9)


def _independent_des_w_moments(a, b, k, rate, lam, total_served=600_000, warmup_fraction=0.05, seed=1):
    """From-scratch DES sampling W (time from arrival to the customer's OWN batch starting
    service, not batch completion -- an earlier validation attempt conflated the two and looked
    like a bug until fixed)."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    server_busy = False
    server_done = inf
    batch: list[float] = []
    batch_start_t = 0.0
    served = 0
    warmup = int(total_served * warmup_fraction)
    w_samples = []

    def maybe_start():
        nonlocal server_busy, server_done, batch, batch_start_t
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = [queue.pop(0) for _ in range(take)]
            server_busy = True
            batch_start_t = t
            server_done = t + rng.gamma(k, 1 / rate)

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
                    w_samples.append(batch_start_t - arr)
                served += 1
            server_busy = False
            server_done = inf
            batch = []
            maybe_start()
    w = np.array(w_samples)
    return [float((w**n).mean()) for n in range(1, 4)]


def test_k1_exact_w_moments_match_bulk_service_mm1():
    """k=1 (Erlang -> Exp): exact W moments must match BulkServiceMM1Calc's own exact get_w()."""
    a, b, rate, lam = 1, 4, 1.5, 1.0

    calc = BulkServiceErlangCalc(a=a, b=b, k=1, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(rate)
    w = calc.get_w()

    ref = BulkServiceMM1Calc(a=a, b=b, queue_truncation=250)
    ref.set_sources(lam)
    ref.set_servers(rate)
    w_ref = ref.get_w()

    assert np.allclose(w, w_ref, rtol=1e-9)


@pytest.mark.parametrize("k", [2, 3])
def test_exact_w_moments_match_independent_des(k):
    a, b, rate, lam = 1, 4, 1.5, 1.0
    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(rate)
    w = calc.get_w(num=3)

    w_des = _independent_des_w_moments(a, b, k, rate, lam)
    assert np.allclose(w[:3], w_des, rtol=0.05)


@pytest.mark.parametrize(
    "a,b,k,lam",
    [(2, 4, 2, 1.0), (3, 5, 2, 1.0), (2, 2, 3, 0.5)],  # (2,2,3) at rate=1.5 saturates at lam=1.0
)
def test_get_w_matches_independent_des_at_a_gt_1(a, b, k, lam):
    """EPIC-067: idle-refill race-aware moments, any 1 <= a <= b."""
    rate = 1.5
    calc = BulkServiceErlangCalc(a=a, b=b, k=k, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(rate)
    w = calc.get_w(num=3)

    w_des = _independent_des_w_moments(a, b, k, rate, lam, total_served=900_000)
    assert np.allclose(w[:3], w_des, rtol=0.06)


def test_set_servers_from_moments_fits_erlang():
    calc = BulkServiceErlangCalc(a=1, b=4, k=1, queue_truncation=250)
    calc.set_sources(1.0)
    mean, m2 = 2.0, 4.5  # var = 0.5 < mean^2 -> cv < 1, fittable to Erlang
    calc.set_servers_from_moments([mean, m2])
    assert calc.k >= 1
    assert calc.rate_fn(1) > 0
    res = calc.run()
    assert res.v[0] > 0


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BulkServiceErlangCalc(a=4, b=1, k=1)
    with pytest.raises(ValueError):
        BulkServiceErlangCalc(a=1, b=4, k=0)
    calc = BulkServiceErlangCalc(a=1, b=4, k=1)
    with pytest.raises(ValueError):
        calc.set_servers(0.0)


if __name__ == "__main__":
    test_k1_reduces_exactly_to_exponential_bulk_service()
    for k_val in (2, 3, 5):
        test_general_k_matches_independent_des(k_val)
    test_larger_k_same_mean_does_not_increase_wait()
    test_batch_size_dependent_rate_matches_independent_des()
    test_batch_size_dependent_rate_reduces_to_scalar_when_constant()
    test_k1_exact_w_moments_match_bulk_service_mm1()
    for k_val in (2, 3):
        test_exact_w_moments_match_independent_des(k_val)
    test_get_w_matches_independent_des_at_a_gt_1(2, 4, 2, 1.0)
    test_set_servers_from_moments_fits_erlang()
    test_invalid_params_rejected()
    print("all bulk-service Erlang tests passed")
