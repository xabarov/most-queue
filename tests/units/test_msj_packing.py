"""Exact packing, nonclairvoyance, preemptive-resume and capacity contracts."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.msj_packing import PACKING_POLICIES, NonpreemptivePacking, server_filling_selection


def replay(policy, trace, needs=(1, 4), **kwargs):
    """Replay with explicit resource classes and optional MSFQ threshold."""
    sim = MsjGeneralSim(4, policy, **kwargs)
    sim.set_servers(needs)
    return sim.run_trace(trace)


def test_first_fit_and_msf_are_different_nonpreemptive_orders():
    """FIFO scanning and resource-descending scanning must remain distinct."""
    trace = [MsjTraceJob(0, cls, 1) for cls in (0, 1, 2)]
    assert replay("first_fit", trace, (1, 2, 4)).start_times == [0, 0, 1]
    assert replay("msf", trace, (1, 2, 4)).start_times == [1, 1, 0]


def test_adaptive_drain_latches_and_blocks_new_small_jobs():
    """A trigger persists across small arrivals until the wide job starts."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 1, 1), MsjTraceJob(2, 0, 1)]
    assert replay("msf", trace).start_times == [0, 10, 2]
    assert replay("adaptive_quickswap", trace).start_times == [0, 10, 11]


def test_adaptive_groups_equal_needs_and_requires_all_running_queues_empty():
    """Every running resource class must have an empty waiting queue."""
    planner = NonpreemptivePacking(4, "adaptive_quickswap")
    # A waiting need-1 job prevents a trigger, even if a need-4 job is absent
    # from service. Duplicate configured class labels cannot change this.
    assert planner.select([(2, 4), (3, 1)], [(0, 1), (1, 3)]) == []
    assert planner.phase == "working"
    planner.select([(2, 4)], [(0, 1), (1, 3)])
    assert planner.phase == "draining"


def test_adaptive_drain_targets_current_largest_not_old_frozen_target():
    """A newly arrived larger job becomes the draining target."""
    planner = NonpreemptivePacking(4, "adaptive_quickswap", phase="draining")
    assert planner.select([(1, 2), (2, 4)], [(0, 1)]) == []
    assert planner.select([(1, 2), (2, 4)], []) == [2]


def test_msfq_inclusive_threshold_and_initial_small_batch():
    """Threshold one closes admission even at one, but zero reduces to MSF."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 1, 1), MsjTraceJob(2, 0, 1)]
    assert replay("msfq", trace, msfq_threshold=1).start_times == [0, 10, 11]
    assert replay("msfq", trace, msfq_threshold=0).start_times == [0, 10, 2]


def test_msfq_literal_phase_drains_even_without_wide_jobs():
    """The four-phase variant does not add an absent-wide exemption."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 0, 1)]
    assert replay("msfq", trace, msfq_threshold=1).start_times == [0, 10]
    assert replay("msfq", trace, msfq_threshold=0).start_times == [0, 1]


def test_msfq_serves_all_wide_jobs_before_returning_to_small():
    """Wide-phase priority remains until the wide class empties."""
    trace = [MsjTraceJob(0, 1, 2), MsjTraceJob(1, 0, 1), MsjTraceJob(1, 1, 1)]
    assert replay("msfq", trace, msfq_threshold=3).start_times == [0, 3, 2]


def test_msfq_crosses_threshold_after_full_small_phase():
    """A full small batch enters drain after completions reach ell exactly."""
    trace = [MsjTraceJob(0, 0, service) for service in (1, 2, 3, 10)]
    trace += [MsjTraceJob(0.5, 1, 1), MsjTraceJob(3.5, 0, 20)]
    assert replay("msfq", trace, msfq_threshold=1).start_times == [0, 0, 0, 0, 10, 11]
    assert replay("msf", trace).start_times == [0, 0, 0, 0, 23.5, 3.5]


def test_server_filling_exact_preemption_resume_and_total_wait():
    """Interrupted jobs keep earned service and count all paused time as W."""
    trace = [MsjTraceJob(0, cls, service) for cls, service in ((1, 1), (0, 10), (1, 3), (2, 2))]
    result = replay("server_filling", trace, (1, 2, 4))
    assert result.start_times == [0, 3, 0, 1]
    assert result.completion_times == [1, 13, 5, 3]
    assert result.wait_samples == [0, 3, 2, 1]
    assert result.preemptions_per_job == [0, 0, 1, 0]
    assert result.preemptions == 1
    assert sorted(result.service_segments) == [(0, 0, 1), (1, 3, 13), (2, 0, 1), (2, 3, 5), (3, 1, 3)]
    assert result.utilization is None  # zero arrival observation horizon
    assert result.reservations == result.backfilled == 0


def test_server_filling_minimal_prefix_excludes_late_large_job():
    """Selection is FCFS-prefix based, not global largest-job priority."""
    assert server_filling_selection(4, [(0, 1), (1, 1), (2, 2), (3, 4)]) == [2, 0, 1]
    assert server_filling_selection(4, [(0, 1), (1, 2)]) == [1, 0]


def test_server_filling_arrival_can_preempt_underfull_prefix():
    """Adding to an underfull prefix may interrupt even an older running job."""
    result = replay("server_filling", [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 1, 2)])
    assert result.start_times == [0, 1]
    assert result.completion_times == [12, 3]
    assert result.wait_samples == [2, 0]
    assert result.preemptions == 1


def test_power_two_prefix_always_fills_when_full_exhaustive():
    """Exhaust all short power-of-two need sequences and verify full packing."""
    for size in range(1, 6):
        for needs in product((1, 2, 4), repeat=size):
            selected = server_filling_selection(4, enumerate(needs))
            assert sum(needs[idx] for idx in selected) == min(4, sum(needs))


@pytest.mark.parametrize("policy", PACKING_POLICIES)
def test_predictions_never_influence_packing(policy):
    """Explicit estimates are ignored and age callbacks are rejected."""
    trace = [MsjTraceJob(0, 0, 2), MsjTraceJob(0.1, 1, 1), MsjTraceJob(0.2, 0, 3)]
    expected = replay(policy, trace)
    altered = replay(policy, [replace(job, estimate=1e300) for job in trace])
    assert expected.start_times == altered.start_times
    assert expected.completion_times == altered.completion_times
    assert expected.service_segments == altered.service_segments
    sim = MsjGeneralSim(4, policy)
    sim.set_servers([1, 4])
    with pytest.raises(ValueError, match="backfilling"):
        sim.run_trace(trace, remaining_predictor=lambda cls, age: 1)


@pytest.mark.parametrize("policy", PACKING_POLICIES)
def test_unobserved_service_changes_do_not_change_earlier_decisions(policy):
    """Before either trace completes work, schedules cannot distinguish S."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 1, 20), MsjTraceJob(2, 0, 30)]
    left = replay(policy, trace)
    right = replay(policy, [replace(job, service=job.service * 2) for job in trace])
    assert [t if t < 10 else None for t in left.start_times] == [t if t < 10 else None for t in right.start_times]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("needs", [(1, 4), (1, 2, 4)])
def test_random_traces_capacity_work_and_msfq_zero_reduction(seed, needs):
    """Seeded general-service paths conserve work and enforce resource limits."""
    rng = np.random.default_rng(seed)
    trace = tuple(
        MsjTraceJob(float(a), int(c), float(s))
        for a, c, s in zip(
            np.cumsum(rng.exponential(0.3, 200)), rng.integers(0, len(needs), 200), rng.lognormal(size=200)
        )
    )
    if len(needs) == 2:
        assert replay("msfq", trace).start_times == replay("msf", trace).start_times
    for policy in PACKING_POLICIES:
        if policy == "msfq" and len(needs) != 2:
            continue  # MSFQ has no definition for an intermediate resource class.
        result = replay(policy, trace, needs)
        segments = result.service_segments or [
            (idx, start, end) for idx, (start, end) in enumerate(zip(result.start_times, result.completion_times))
        ]
        work, events = np.zeros(len(trace)), []
        for idx, start, end in segments:
            assert trace[idx].arrival <= start < end <= result.completion_times[idx]
            work[idx] += end - start
            need = needs[trace[idx].cls]
            events.extend([(start, need), (end, -need)])
        used = 0
        for _, delta in sorted(events):
            used += delta
            assert 0 <= used <= 4
        assert used == 0
        np.testing.assert_allclose(work, [job.service for job in trace], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.wait_samples, np.array(result.sojourn_samples) - work, atol=1e-12)
        if policy == "server_filling":
            assert len(segments) == len(trace) + result.preemptions


@pytest.mark.parametrize("policy", PACKING_POLICIES)
def test_replay_state_resets_ties_and_warmup(policy):
    """Repeated runs reset phases and apply completion-first event ties."""
    trace = [MsjTraceJob(0, 1, 1), MsjTraceJob(1, 1, 1), MsjTraceJob(1, 0, 2)]
    sim = MsjGeneralSim(4, policy)
    sim.set_servers([1, 4])
    result = sim.run_trace(trace, warmup_jobs=1)
    again = sim.run_trace(trace, warmup_jobs=1)
    assert result.start_times == again.start_times == [0, 1, 2]
    assert result.wait_samples == [0, 1]
    assert result.counts_per_class == [1, 1]


@pytest.mark.parametrize("threshold", [-1, 4, 1.5, True])
def test_invalid_msfq_threshold(threshold):
    """Refuse out-of-domain thresholds including booleans and fractions."""
    with pytest.raises(ValueError, match="msfq_threshold"):
        MsjGeneralSim(4, "msfq", msfq_threshold=threshold)


def test_invalid_policy_and_resource_domains():
    """Unsupported needs raise instead of rounding or silently switching policy."""
    with pytest.raises(ValueError, match="unknown"):
        MsjGeneralSim(4, "srpt")
    with pytest.raises(ValueError, match="only valid"):
        MsjGeneralSim(4, "msf", msfq_threshold=1)
    with pytest.raises(ValueError, match="power-of-two k"):
        MsjGeneralSim(3, "server_filling")
    for policy, message in (("server_filling", "power-of-two needs"), ("msfq", "one-or-all")):
        sim = MsjGeneralSim(4, policy)
        with pytest.raises(ValueError, match=message):
            sim.set_servers([1, 3, 4])


@pytest.mark.parametrize("policy", PACKING_POLICIES)
def test_single_server_and_generator_api(policy):
    """Every new discipline supports the common generation interface at k=1."""
    sim = MsjGeneralSim(1, policy, seed=52)
    sim.set_servers([1], [lambda rng: rng.uniform(0.5, 1.5)])
    sim.set_sources([0.4])
    result = sim.run(200)
    assert len(result.wait_samples) == 200
    assert result.preemptions == 0
