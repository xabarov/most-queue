"""
Unit tests for M/H2(p1,mu1,mu2)^[a,b]/1 bulk-service with general
(phase-type-fitted) batch-service time (most_queue.theory.batch.bulk_service_h2, EPIC-036).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc


def _independent_des_mean_v(a, b, p1, mu1, mu2, lam, total_served=400_000, warmup_fraction=0.05, seed=1):
    """From-scratch DES (not reusing BulkServiceSim) sampling H2(p1,mu1,mu2) batch-service times."""
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

    def sample_h2():
        rate = mu1 if rng.random() < p1 else mu2
        return rng.exponential(1 / rate)

    def maybe_start():
        nonlocal server_busy, server_done, batch
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = [queue.pop(0) for _ in range(take)]
            server_busy = True
            server_done = t + sample_h2()

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


def test_p1_1_reduces_exactly_to_exponential_bulk_service():
    """H2 with p1=1 = Exp(mu1): must exactly match the already-validated BulkServiceMM1Calc."""
    a, b, mu1, mu2, lam = 1, 4, 1.5, 3.0, 1.0

    calc = BulkServiceH2Calc(a=a, b=b, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(p1=1.0, mu1=mu1, mu2=mu2)
    v = calc.run().v[0]

    ref = BulkServiceMM1Calc(a=a, b=b, queue_truncation=250)
    ref.set_sources(lam)
    ref.set_servers(mu1)
    v_ref = ref.run().v[0]

    assert np.isclose(v, v_ref, rtol=1e-9)


@pytest.mark.parametrize("p1,mu1,mu2", [(0.5, 1.0, 3.0), (0.3, 0.8, 4.0), (0.7, 2.0, 2.5)])
def test_general_case_matches_independent_des(p1, mu1, mu2):
    a, b, lam = 1, 4, 1.0
    calc = BulkServiceH2Calc(a=a, b=b, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(p1=p1, mu1=mu1, mu2=mu2)
    v = calc.run().v[0]

    v_des = _independent_des_mean_v(a, b, p1, mu1, mu2, lam)
    assert np.isclose(v, v_des, rtol=0.03)


def _independent_des_mean_v_batch_dependent(
    a, b, p1_fn, mu1_fn, mu2_fn, lam, total_served=400_000, warmup_fraction=0.05, seed=1
):
    """Same DES as above, but with batch-size-dependent H2 parameters."""
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

    def sample_h2(size):
        p1 = p1_fn(size)
        rate = mu1_fn(size) if rng.random() < p1 else mu2_fn(size)
        return rng.exponential(1 / rate)

    def maybe_start():
        nonlocal server_busy, server_done, batch
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            batch = [queue.pop(0) for _ in range(take)]
            server_busy = True
            server_done = t + sample_h2(take)

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


def test_batch_size_dependent_params_matches_independent_des():
    """p1(i) varies with batch size i -- LLM/GPU-style variability changing with batch size."""
    a, b, lam = 1, 4, 1.0

    def p1_fn(i):
        return 0.3 + 0.1 * i

    def mu1_fn(_i):
        return 1.5

    def mu2_fn(_i):
        return 3.0

    calc = BulkServiceH2Calc(a=a, b=b, queue_truncation=250)
    calc.set_sources(lam)
    calc.set_servers(p1_fn, mu1_fn, mu2_fn)
    v = calc.run().v[0]

    v_des = _independent_des_mean_v_batch_dependent(a, b, p1_fn, mu1_fn, mu2_fn, lam)
    assert np.isclose(v, v_des, rtol=0.03)


def test_batch_size_dependent_params_reduce_to_scalar_when_constant():
    a, b, lam = 1, 4, 1.0
    p1, mu1, mu2 = 0.5, 1.5, 3.0

    calc_scalar = BulkServiceH2Calc(a=a, b=b, queue_truncation=250)
    calc_scalar.set_sources(lam)
    calc_scalar.set_servers(p1, mu1, mu2)
    v_scalar = calc_scalar.run().v[0]

    calc_fn = BulkServiceH2Calc(a=a, b=b, queue_truncation=250)
    calc_fn.set_sources(lam)
    calc_fn.set_servers(lambda _i: p1, lambda _i: mu1, lambda _i: mu2)
    v_fn = calc_fn.run().v[0]

    assert np.isclose(v_scalar, v_fn, atol=1e-9)


def test_set_servers_from_moments_fits_h2():
    calc = BulkServiceH2Calc(a=1, b=4, queue_truncation=250)
    calc.set_sources(1.0)
    mean, m2, m3 = 2.0, 12.0, 100.0  # var >> mean^2 -> cv > 1, fittable to H2
    calc.set_servers_from_moments([mean, m2, m3])
    assert 0.0 <= calc.p1_fn(1) <= 1.0
    assert calc.mu1_fn(1) > 0
    assert calc.mu2_fn(1) > 0
    res = calc.run()
    assert res.v[0] > 0


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BulkServiceH2Calc(a=4, b=1)
    calc = BulkServiceH2Calc(a=1, b=4)
    with pytest.raises(ValueError):
        calc.set_servers(p1=1.5, mu1=1.0, mu2=1.0)
    with pytest.raises(ValueError):
        calc.set_servers(p1=0.5, mu1=0.0, mu2=1.0)


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        BulkServiceH2Calc(a=1, b=4, gamma=-0.1)


def test_gamma_abandonment_pi_level_only_matches_independent_des():
    """EPIC-068: H2 only gets gamma support at the _solve_pi level (no gamma-aware
    get_abandonment_prob/get_w/get_tail -- see module docstring for why); cross-check
    E[N] against an independent DES with per-customer abandonment, mirroring EPIC-068's
    pi-validation discipline for the other two calculators."""
    a, b, p1, mu1, mu2, lam, gamma = 2, 4, 0.5, 1.0, 3.0, 0.6, 0.5
    calc = BulkServiceH2Calc(a=a, b=b, queue_truncation=150, gamma=gamma)
    calc.set_sources(lam)
    calc.set_servers(p1=p1, mu1=mu1, mu2=mu2)
    pi = calc._solve_pi()  # pylint: disable=protected-access

    e_n_theory = 0.0
    for j in range(calc.N + 1):
        e_n_theory += pi[calc._idle_index(j)] * j  # pylint: disable=protected-access
    for i in range(1, b + 1):
        for phase in (0, 1):
            for j in range(calc.N + 1):
                e_n_theory += pi[calc._busy_index(i, phase, j)] * (i + j)  # pylint: disable=protected-access

    e_n_des = _independent_des_mean_n_with_abandonment(a, b, p1, mu1, mu2, lam, gamma)
    assert np.isclose(e_n_theory, e_n_des, rtol=0.05)


def _independent_des_mean_n_with_abandonment(
    a, b, p1, mu1, mu2, lam, gamma, total_time=400_000.0, warmup=5_000.0, seed=3
):
    """Independent from-scratch DES: time-average E[N] with per-customer abandonment."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    next_arrival = rng.exponential(1 / lam)
    queue: list[float] = []
    server_busy = False
    server_done = inf
    batch_size = 0
    area = 0.0
    last_t = 0.0

    def sample_h2():
        rate = mu1 if rng.random() < p1 else mu2
        return rng.exponential(1 / rate)

    def maybe_start():
        nonlocal server_busy, server_done, batch_size
        if not server_busy and len(queue) >= a:
            take = min(b, len(queue))
            del queue[:take]
            server_busy = True
            batch_size = take
            server_done = t + sample_h2()

    while t < total_time:
        next_abandon_t, next_abandon_idx = inf, None
        for idx_c, aband in enumerate(queue):
            if aband < next_abandon_t:
                next_abandon_t, next_abandon_idx = aband, idx_c
        t = min(next_arrival, server_done, next_abandon_t)
        if t > last_t and last_t >= warmup:
            n_in_system = len(queue) + batch_size
            area += n_in_system * (t - last_t)
        elif t > warmup > last_t:
            n_in_system = len(queue) + batch_size
            area += n_in_system * (t - warmup)
        last_t = t
        if t == next_arrival:
            aband_t = t + rng.exponential(1.0 / gamma) if gamma > 0 else inf
            queue.append(aband_t)
            next_arrival = t + rng.exponential(1 / lam)
            maybe_start()
        elif t == server_done:
            server_busy = False
            server_done = inf
            batch_size = 0
            maybe_start()
        else:
            queue.pop(next_abandon_idx)

    return area / (total_time - warmup)


if __name__ == "__main__":
    test_p1_1_reduces_exactly_to_exponential_bulk_service()
    for params in [(0.5, 1.0, 3.0), (0.3, 0.8, 4.0), (0.7, 2.0, 2.5)]:
        test_general_case_matches_independent_des(*params)
    test_batch_size_dependent_params_matches_independent_des()
    test_batch_size_dependent_params_reduce_to_scalar_when_constant()
    test_set_servers_from_moments_fits_h2()
    test_invalid_params_rejected()
    test_invalid_gamma_rejected()
    test_gamma_abandonment_pi_level_only_matches_independent_des()
    print("all bulk-service H2 tests passed")
