"""
Unit tests for M/G/n egalitarian Processor Sharing
(most_queue.theory.fifo.mgn_ps.MGnPSCalc) -- generalizes MG1PSCalc (n=1) to
n identical servers, added alongside EPIC-073 (occupancy-dependent
continuous batching) to complete the model lineage: MGnPSCalc is the
unbounded (no admission cap, no external queue) sibling of
OccupancyDependentQueueCalc's finite-k, FCFS-queued construction.
"""

import numpy as np
import pytest

from most_queue.random.distributions import GammaDistribution
from most_queue.theory.fifo.mg1_ps import MG1PSCalc
from most_queue.theory.fifo.mgn_ps import MGnPSCalc
from most_queue.theory.fifo.mmnr import MMnrCalc


def _independent_des_mean_sojourn(n, lam, service_sampler, total_served=200_000, warmup_fraction=0.05, seed=1):
    """From-scratch "fluid work" DES for M/G/n egalitarian PS (not reusing
    most_queue code): each present job's remaining work drains at rate
    min(n, m)/m while m jobs are present (dedicated service below n, shared
    equally above it); the job reaching zero first departs.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    next_arrival = rng.exponential(1.0 / lam)
    jobs: list[list[float]] = []  # [arrival_time, remaining_work]
    served = 0
    warmup = int(total_served * warmup_fraction)
    sojourn_sum = 0.0
    n_counted = 0

    while served < total_served + warmup:
        m = len(jobs)
        if m == 0:
            t = next_arrival
            jobs.append([t, service_sampler(rng)])
            next_arrival = t + rng.exponential(1.0 / lam)
            continue
        rate_each = 1.0 if m <= n else n / m
        min_remaining = min(job[1] for job in jobs)
        time_to_completion = min_remaining / rate_each
        if next_arrival <= t + time_to_completion:
            dt = next_arrival - t
            for job in jobs:
                job[1] -= rate_each * dt
            t = next_arrival
            jobs.append([t, service_sampler(rng)])
            next_arrival = t + rng.exponential(1.0 / lam)
        else:
            dt = time_to_completion
            for job in jobs:
                job[1] -= rate_each * dt
            t += dt
            idx = min(range(len(jobs)), key=lambda i: jobs[i][1])
            arr_t, _ = jobs.pop(idx)
            served += 1
            if served > warmup:
                sojourn_sum += t - arr_t
                n_counted += 1
    return sojourn_sum / n_counted


def test_n1_reduces_exactly_to_mg1_ps():
    """n=1 must regress exactly to the already-validated MG1PSCalc."""
    lam, b1 = 0.6, 1.0
    calc = MGnPSCalc(n=1)
    calc.set_sources(lam)
    calc.set_servers([b1])
    res = calc.run()

    ref = MG1PSCalc()
    ref.set_sources(lam)
    ref.set_servers([b1])
    res_ref = ref.run()

    assert res.v[0] == pytest.approx(res_ref.v[0], rel=1e-9)
    assert res.w[0] == pytest.approx(res_ref.w[0], rel=1e-9)
    for p, p_ref in zip(res.p[:30], res_ref.p[:30]):
        assert p == pytest.approx(p_ref, abs=1e-9)


def test_queue_length_distribution_matches_classical_mmn():
    """PS and FCFS with n servers share the same aggregate occupancy process
    (load-dependent throughput min(n,j)/b1) -- the queue-length distribution
    must match the classical M/M/n (Erlang-C) one exactly."""
    n, lam, mu = 4, 2.5, 1.0
    calc = MGnPSCalc(n=n)
    calc.set_sources(lam)
    calc.set_servers([1.0 / mu])
    p = calc.get_p()

    ref = MMnrCalc(n=n, r=150)
    ref.set_sources(lam)
    ref.set_servers(mu)
    p_ref = ref.get_p()

    for a, b in zip(p[:50], p_ref[:50]):
        assert a == pytest.approx(b, abs=1e-9)


def test_mean_sojourn_matches_independent_des_with_nonexponential_service():
    """Gamma service (CV != 1) checks the BCMP/Kelly insensitivity claim:
    mean sojourn should match an independent, from-scratch DES regardless of
    service-time shape (only the mean enters the exact formula)."""
    n, lam, mean, cv = 3, 2.0, 1.0, 1.7
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean, cv)
    b = GammaDistribution.calc_theory_moments(gamma_params, 2)

    calc = MGnPSCalc(n=n)
    calc.set_sources(lam)
    calc.set_servers(b)
    v_exact = calc.get_v()[0]

    def sampler(rng):
        return GammaDistribution.generate_static(gamma_params, rng)

    v_des = _independent_des_mean_sojourn(n, lam, sampler, total_served=200_000, seed=3)

    assert v_des == pytest.approx(v_exact, rel=0.05)


def test_rejects_unstable_system():
    calc = MGnPSCalc(n=2)
    calc.set_sources(3.0)
    calc.set_servers([1.0])  # rho = 3*1/2 = 1.5 >= 1
    with pytest.raises(ValueError):
        calc.run()


def test_rejects_nonpositive_arrival_rate():
    calc = MGnPSCalc(n=2)
    with pytest.raises(ValueError):
        calc.set_sources(0.0)
