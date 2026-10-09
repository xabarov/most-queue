"""
Unit tests for EPIC-074 (M2): occupancy-modulated two-branch service with a
hard concurrency cap (most_queue.theory.continuous_batching.
occupancy_dependent_h2) -- the heterogeneous-output-length companion to
EPIC-073's exponential OccupancyDependentQueueCalc.
"""

from collections import deque

import numpy as np
import pytest

from most_queue.theory.continuous_batching import OccupancyDependentQueueCalc
from most_queue.theory.continuous_batching.occupancy_dependent_h2 import OccupancyDependentH2QueueCalc


def _independent_des(k, lam, p1_fn, mu1_fn, mu2_fn, total=2_000_000, warmfrac=0.10, seed=1):
    """From-scratch DES (no most_queue code reused) for occupancy-modulated
    two-branch service. Returns (mean wait before admission, time-average N).

    NOTE on run length: this system's level distribution decays slowly enough
    (ratio ~0.8 at the tested load) that 6e5 arrivals gives a visibly biased
    ~1% answer purely from autocorrelation; 2e6 with a 10% warmup brings the
    deviation inside one standard error. Do not shorten without re-checking.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    next_arr = rng.exponential(1.0 / lam)
    active: list[int] = []
    waiting: deque = deque()
    n_arr = 0
    warm = int(total * warmfrac)
    wsum = 0.0
    wcnt = 0
    nsum = 0.0
    tacc = 0.0

    while n_arr < total + warm:
        occ = len(active)
        if occ > 0:
            m = sum(1 for b in active if b == 0)
            r0 = m * mu1_fn(occ)
            r1 = (occ - m) * mu2_fn(occ)
            next_dep = t + rng.exponential(1.0 / (r0 + r1))
        else:
            r0 = r1 = 0.0
            next_dep = float("inf")

        tnext = min(next_arr, next_dep)
        if n_arr > warm:
            nsum += (occ + len(waiting)) * (tnext - t)
            tacc += tnext - t

        if next_arr <= next_dep:
            t = next_arr
            n_arr += 1
            if len(active) < k:
                active.append(0 if rng.random() < p1_fn(len(active) + 1) else 1)
                if n_arr > warm:
                    wcnt += 1
            else:
                waiting.append(t)
            next_arr = t + rng.exponential(1.0 / lam)
        else:
            t = next_dep
            leaves0 = rng.random() < r0 / (r0 + r1)
            idx = next(i for i, b in enumerate(active) if b == (0 if leaves0 else 1))
            active.pop(idx)
            if waiting:
                arr = waiting.popleft()
                if n_arr > warm:
                    wsum += t - arr
                    wcnt += 1
                active.append(0 if rng.random() < p1_fn(k) else 1)
    return wsum / wcnt, nsum / tacc


def test_p1_one_reduces_exactly_to_exponential_sibling():
    """p1 == 1 means every request takes branch 0, so the model must collapse
    exactly onto EPIC-073's exponential OccupancyDependentQueueCalc at mu=mu1."""
    k, truncation, lam = 4, 150, 2.0

    def mu_fn(occ):
        return 2.0 / (0.5 + 0.3 * occ)

    calc = OccupancyDependentH2QueueCalc(k=k, queue_truncation=truncation)
    calc.set_sources(lam)
    calc.set_servers(p1=1.0, mu1=mu_fn, mu2=lambda _occ: 99.0)  # mu2 unreachable at p1=1

    ref = OccupancyDependentQueueCalc(k=k, queue_truncation=truncation)
    ref.set_sources(lam)
    ref.set_servers(mu_fn)

    p = calc.get_level_probs()
    p_ref = ref._solve_pi()  # pylint: disable=protected-access
    for a, b in zip(p[:60], p_ref[:60]):
        assert a == pytest.approx(b, abs=1e-10)
    assert calc.get_w_mean() == pytest.approx(ref.get_w(1)[0], rel=1e-7)


def test_equal_branches_reduce_to_exponential_sibling():
    """mu1 == mu2 makes the branch label irrelevant, whatever p1 is -- another
    independent route to the same exponential reduction."""
    k, truncation, lam = 3, 120, 1.5

    def mu_fn(occ):
        return 1.6 / (0.4 + 0.2 * occ)

    calc = OccupancyDependentH2QueueCalc(k=k, queue_truncation=truncation)
    calc.set_sources(lam)
    calc.set_servers(p1=0.42, mu1=mu_fn, mu2=mu_fn)

    ref = OccupancyDependentQueueCalc(k=k, queue_truncation=truncation)
    ref.set_sources(lam)
    ref.set_servers(mu_fn)

    assert calc.get_w_mean() == pytest.approx(ref.get_w(1)[0], rel=1e-7)
    assert calc.get_p_wait() == pytest.approx(ref.get_p_wait(), rel=1e-9)


def test_level_count_and_homogeneity_structure():
    """Structural invariants this model relies on: per-level microstate count is
    min(j,k)+1, and the level blocks stop changing from level k+1 onward (the
    boundary is 0..k, not 0..k-1 -- at j==k a completion is NOT refilled)."""
    k = 3
    calc = OccupancyDependentH2QueueCalc(k=k, queue_truncation=12)
    for j in range(13):
        assert calc._level_size(j) == min(j, k) + 1  # pylint: disable=protected-access
    offsets = calc._offsets()  # pylint: disable=protected-access
    # levels 0..k grow, levels >k are constant-size
    sizes = [offsets[j + 1] - offsets[j] for j in range(13)]
    assert sizes[:5] == [1, 2, 3, 4, 4]
    assert len(set(sizes[k:])) == 1


def test_mean_wait_and_n_match_independent_des():
    """Genuine two-branch heterogeneity (mu1/mu2 differ ~4x) against an
    independent from-scratch DES."""
    k, lam = 3, 1.6

    def p1_fn(_occ):
        return 0.35

    def mu1_fn(occ):
        return 2.5 / (0.4 + 0.25 * occ)

    def mu2_fn(occ):
        return 0.6 / (0.4 + 0.25 * occ)

    calc = OccupancyDependentH2QueueCalc(k=k, queue_truncation=400)
    calc.set_sources(lam)
    calc.set_servers(p1=p1_fn, mu1=mu1_fn, mu2=mu2_fn)
    ew = calc.get_w_mean()
    en = calc.get_n_moments(1)[0]

    w_des, n_des = _independent_des(k, lam, p1_fn, mu1_fn, mu2_fn, total=2_000_000, seed=5)
    assert w_des == pytest.approx(ew, rel=0.03)
    assert n_des == pytest.approx(en, rel=0.03)


def test_little_law_consistency():
    """E[W] (queue-only Little's law) must be consistent with the level
    distribution it was derived from, as an internal cross-check."""
    k, lam = 4, 2.2

    def mu1_fn(occ):
        return 2.0 / (0.3 + 0.2 * occ)

    def mu2_fn(occ):
        return 0.8 / (0.3 + 0.2 * occ)

    calc = OccupancyDependentH2QueueCalc(k=k, queue_truncation=300)
    calc.set_sources(lam)
    calc.set_servers(p1=0.5, mu1=mu1_fn, mu2=mu2_fn)

    p = np.asarray(calc.get_level_probs())
    j = np.arange(len(p))
    lq = float((p * np.maximum(j - k, 0)).sum())
    assert calc.get_w_mean() == pytest.approx(lq / lam, rel=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-10)


def test_phase_type_wait_distribution_is_explicit_reserve():
    """The full waiting-time DISTRIBUTION is a documented reserve for this
    model (unlike the exponential sibling, the above-cap departure rate is
    composition-dependent, so it is not a plain Erlang mixture)."""
    calc = OccupancyDependentH2QueueCalc(k=2, queue_truncation=40)
    calc.set_sources(1.0)
    calc.set_servers(p1=0.4, mu1=2.0, mu2=1.0)
    with pytest.raises(NotImplementedError):
        calc._wait_phase_generator()  # pylint: disable=protected-access


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        OccupancyDependentH2QueueCalc(k=5, queue_truncation=3)
    calc = OccupancyDependentH2QueueCalc(k=2, queue_truncation=40)
    with pytest.raises(ValueError):
        calc.set_sources(0.0)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers(p1=1.5, mu1=1.0, mu2=1.0)
    with pytest.raises(ValueError):
        calc.set_servers(p1=0.5, mu1=0.0, mu2=1.0)
