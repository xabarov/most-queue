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


def test_set_servers_from_moments_fits_h2():
    calc = BulkServiceH2Calc(a=1, b=4, queue_truncation=250)
    calc.set_sources(1.0)
    mean, m2, m3 = 2.0, 12.0, 100.0  # var >> mean^2 -> cv > 1, fittable to H2
    calc.set_servers_from_moments([mean, m2, m3])
    assert 0.0 <= calc.p1 <= 1.0
    assert calc.mu1 > 0
    assert calc.mu2 > 0
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


if __name__ == "__main__":
    test_p1_1_reduces_exactly_to_exponential_bulk_service()
    for params in [(0.5, 1.0, 3.0), (0.3, 0.8, 4.0), (0.7, 2.0, 2.5)]:
        test_general_case_matches_independent_des(*params)
    test_set_servers_from_moments_fits_h2()
    test_invalid_params_rejected()
    print("all bulk-service H2 tests passed")
