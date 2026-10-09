"""
Unit tests for EPIC-073: exact waiting-time distribution for an occupancy-
dependent queue with a hard concurrency cap (most_queue.theory.
continuous_batching.occupancy_dependent), the Markovian core of LLM-serving
continuous batching.
"""

from collections import deque

import numpy as np
import pytest
from scipy import integrate

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.theory.continuous_batching import OccupancyDependentQueueCalc
from most_queue.theory.fifo.mmnr import MMnrCalc


def _independent_des_wait(k, mu_fn, lam, total_arrivals=400_000, warmup_fraction=0.05, seed=1):
    """From-scratch DES (no most_queue code reused) for the occupancy-dependent
    queue: `k` concurrency cap, per-request completion rate `mu_fn(occupancy)`,
    FCFS admission beyond the cap. Returns raw moments of the waiting time
    (queueing delay before admission into the active pool).
    """
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    active = 0
    waiting = deque()  # arrival timestamps of customers not yet admitted
    next_arrival = rng.exponential(1.0 / lam)
    next_departure = inf

    def redraw_departure(cur_t, occ):
        if occ == 0:
            return inf
        return cur_t + rng.exponential(1.0 / (occ * mu_fn(occ)))

    wait_samples = []
    warmup = int(total_arrivals * warmup_fraction)
    n_arrivals = 0

    while n_arrivals < total_arrivals + warmup:
        if next_arrival <= next_departure:
            t = next_arrival
            n_arrivals += 1
            if active < k:
                active += 1
                next_departure = redraw_departure(t, active)
                if n_arrivals > warmup:
                    wait_samples.append(0.0)
            else:
                waiting.append(t)
            next_arrival = t + rng.exponential(1.0 / lam)
        else:
            t = next_departure
            if waiting:
                arr = waiting.popleft()
                if n_arrivals > warmup:
                    wait_samples.append(t - arr)
                # active count unchanged (instantly refilled) -- redraw at the SAME rate
                next_departure = redraw_departure(t, active)
            else:
                active -= 1
                next_departure = redraw_departure(t, active)

    w = np.array(wait_samples)
    return [float((w**p).mean()) for p in range(1, 3)]


def test_constant_mu_matches_classical_mmc():
    """mu(occupancy) == const must regress exactly to the classical M/M/k/N queue."""
    k, truncation, lam, mu = 5, 80, 3.0, 1.0
    calc = OccupancyDependentQueueCalc(k=k, queue_truncation=truncation)
    calc.set_sources(lam)
    calc.set_servers(mu)
    w = calc.get_w(4)

    ref = MMnrCalc(n=k, r=truncation)
    ref.set_sources(lam)
    ref.set_servers(mu)
    w_ref = ref.get_w(4)

    for a, b in zip(w, w_ref):
        assert a == pytest.approx(b, rel=1e-6)


def test_k1_reduces_to_single_server_erlang_style_wait():
    """k=1: a single, occupancy-independent 'server' -- classical M/M/1/N wait moments."""
    truncation, lam, mu = 100, 0.5, 1.0
    calc = OccupancyDependentQueueCalc(k=1, queue_truncation=truncation)
    calc.set_sources(lam)
    calc.set_servers(mu)
    w = calc.get_w(2)

    ref = MMnrCalc(n=1, r=truncation)
    ref.set_sources(lam)
    ref.set_servers(mu)
    w_ref = ref.get_w(2)

    for a, b in zip(w, w_ref):
        assert a == pytest.approx(b, rel=1e-6)


def test_tail_integrates_to_mean_waiting_time():
    """Internal consistency: integral_0^inf P(W>t) dt must equal E[W] from get_w()."""

    def mu_fn(occ):
        return 1.0 / (0.2 + 0.05 * occ)

    calc = OccupancyDependentQueueCalc(k=8, queue_truncation=150)
    calc.set_sources(2.0)
    calc.set_servers(mu_fn)
    e_w = calc.get_w(1)[0]
    e_w_integral, _ = integrate.quad(calc.get_tail, 0, 500)
    assert e_w_integral == pytest.approx(e_w, rel=1e-4)


def test_p_wait_matches_tail_distribution_mass():
    """get_p_wait() must equal get_tail(0) (probability that W > 0)."""
    calc = OccupancyDependentQueueCalc(k=6, queue_truncation=120)
    calc.set_sources(2.5)
    calc.set_servers(lambda occ: 1.2 / occ**0.3)
    assert calc.get_p_wait() == pytest.approx(calc.get_tail(0.0), rel=1e-9)


def test_occupancy_dependent_rate_matches_independent_des():
    """Genuine occupancy-dependent rate (slowdown with batch size), validated
    against an independent, from-scratch DES."""
    k, lam = 4, 1.5

    def mu_fn(occ):
        return 2.0 / (0.5 + 0.3 * occ)  # per-request rate drops as occupancy grows

    calc = OccupancyDependentQueueCalc(k=k, queue_truncation=200)
    calc.set_sources(lam)
    calc.set_servers(mu_fn)
    w_exact = calc.get_w(2)

    w_des = _independent_des_wait(k, mu_fn, lam, total_arrivals=500_000, seed=7)

    assert w_des[0] == pytest.approx(w_exact[0], rel=0.05)
    assert w_des[1] == pytest.approx(w_exact[1], rel=0.12)


def test_erlang_mixture_matches_manual_formula():
    """get_w()/get_tail() must equal a direct, hand-written mixture-of-Erlang
    formula built from pi and the (pinned-at-k) departure rate -- an
    independent re-derivation of the same closed form used internally."""
    k, truncation, lam = 3, 60, 1.0

    def mu_fn(occ):
        return 1.0 / occ

    calc = OccupancyDependentQueueCalc(k=k, queue_truncation=truncation)
    calc.set_sources(lam)
    calc.set_servers(mu_fn)
    pi = calc._solve_pi()  # pylint: disable=protected-access
    rate = k * mu_fn(k)

    manual_tail = sum(
        pi[j] * ErlangDistribution.get_tail(ErlangParams(r=j - k + 1, mu=rate), 2.0)
        for j in range(k, len(pi))
        if pi[j] > 0
    )
    assert manual_tail == pytest.approx(calc.get_tail(2.0), rel=1e-9)


def test_rejects_truncation_below_k():
    with pytest.raises(ValueError):
        OccupancyDependentQueueCalc(k=5, queue_truncation=3)


def test_rejects_nonpositive_constant_mu():
    calc = OccupancyDependentQueueCalc(k=2, queue_truncation=50)
    with pytest.raises(ValueError):
        calc.set_servers(0.0)
