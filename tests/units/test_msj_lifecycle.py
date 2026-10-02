"""Exact lifecycle schedules and compatibility, with no external data."""

from dataclasses import replace

import numpy as np
import pytest

from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.sim.msj_lifecycle import LIFECYCLE_POLICIES, MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim


def run(trace, *, capacity=2, needs=(1, 2), policy="fcfs", **kwargs):
    sim = MsjLifecycleSim(capacity, policy)
    sim.set_servers(needs)
    return sim.run_lifecycle(trace, **kwargs)


@pytest.mark.parametrize("policy", LIFECYCLE_POLICIES)
def test_no_extensions_matches_existing_dispatch_and_statistics(policy):
    rng = np.random.default_rng(571)
    trace = tuple(
        MsjLifecycleJob(float(i), int(c), float(s), 4.0)
        for i, c, s in zip(np.cumsum(rng.exponential(2, 70)), rng.integers(0, 2, 70), rng.lognormal(1, 1, 70))
    )
    actual = run(trace, policy=policy)
    sim = MsjGeneralSim(2, policy)
    sim.set_servers([1, 2])
    expected = sim.run_trace(trace)
    assert actual.start_times == expected.start_times
    assert actual.release_times == expected.completion_times
    assert actual.utilization == expected.utilization
    assert actual.idle_with_queue == expected.idle_with_queue
    assert actual.reservation_violations == expected.reservation_violations
    assert actual.reserved_start_times == expected.reserved_start_times


@pytest.mark.parametrize("policy", LIFECYCLE_POLICIES)
def test_carry_seeded_before_dispatch_and_initial_queue_precedes_new_jobs(policy):
    running = MsjCarryIn(MsjLifecycleJob(0, 1, 10, 10), age=7)
    queued = MsjLifecycleJob(0, 0, 2, 2)
    trace = [MsjLifecycleJob(0, 0, 1, 1), MsjLifecycleJob(1, 1, 1, 1)]
    result = run(trace, initial_running=[running], initial_waiting=[queued], policy=policy)
    assert result.start_times[0] == -7
    assert result.release_times[0] == 3
    assert result.consumed_services[0] == 3
    assert all(x >= 3 for x in result.start_times[1:])
    assert result.initial_count == 2
    assert result.resource_time_by_outcome["completed"] == 11
    assert result.utilization == 1


def test_initial_queue_can_start_before_first_new_arrival():
    result = run([MsjLifecycleJob(10, 0, 2)], initial_waiting=[MsjLifecycleJob(0, 0, 3)])
    assert result.start_times == [0, 10]
    assert result.release_times == [3, 12]
    assert result.utilization is None


def test_cancelled_release_is_not_success_and_frees_capacity():
    trace = [MsjLifecycleJob(0, 1, 5, outcome="cancelled"), MsjLifecycleJob(1, 1, 2)]
    result = run(trace)
    assert result.start_times == [0, 5]
    assert result.release_times == [5, 7]
    assert result.outcomes == ["cancelled", "completed"]
    assert result.resource_time_by_outcome == {"completed": 4, "cancelled": 10, "timed_out": 0}


@pytest.mark.parametrize("policy", LIFECYCLE_POLICIES)
def test_hard_limit_binds_after_start_not_after_arrival(policy):
    trace = [MsjLifecycleJob(0, 1, 5, 5), MsjLifecycleJob(1, 1, 10, 10, runtime_limit=2)]
    result = run(trace, policy=policy)
    assert result.start_times == [0, 5]
    assert result.release_times == [5, 7]
    assert result.outcomes == ["completed", "timed_out"]
    assert result.resource_time_by_outcome["timed_out"] == 4
    assert trace[1].service == 10  # no mutation and no hidden retrial


@pytest.mark.parametrize("outcome", ["completed", "cancelled"])
def test_equal_budget_preserves_original_outcome(outcome):
    result = run([MsjLifecycleJob(0, 0, 2, outcome=outcome, runtime_limit=2)])
    assert result.outcomes == [outcome]


def test_limit_before_recorded_cancel_retains_timeout_label():
    result = run([MsjLifecycleJob(0, 0, 5, outcome="cancelled", runtime_limit=2)])
    assert result.outcomes == ["timed_out"]
    assert result.consumed_services == [2]


@pytest.mark.parametrize("estimate", [None, 1.0])
def test_unknown_or_overdue_carry_forecast_does_not_use_true_residual(estimate):
    carry = MsjCarryIn(MsjLifecycleJob(0, 0, 10, estimate), age=5)
    result = run(
        [MsjLifecycleJob(0, 1, 1, 1), MsjLifecycleJob(1, 0, 1, 1)],
        initial_running=[carry],
        policy="easy",
    )
    assert result.start_times == [-5, 5, 6]


def test_cancellation_completion_and_arrival_tie_releases_first():
    result = run([MsjLifecycleJob(0, 1, 2, outcome="cancelled"), MsjLifecycleJob(2, 1, 1)])
    assert result.start_times == [0, 2]
    assert result.utilization == 1


def test_lindley_recursion_with_backlog_and_limits_is_independent_reference():
    trace = [MsjLifecycleJob(a, 0, s, runtime_limit=l) for a, s, l in [(0, 9, 3), (1, 2, None), (8, 6, 1)]]
    result = run(trace, capacity=1, needs=(1,), initial_running=[MsjCarryIn(MsjLifecycleJob(0, 0, 5), 2)])
    end, starts, releases = 3, [-2], [3]
    for job in trace:
        start = max(job.arrival, end)
        end = start + min(job.service, job.runtime_limit or job.service)
        starts.append(start)
        releases.append(end)
    assert result.start_times == starts
    assert result.release_times == releases


def test_window_excludes_drain_and_accounts_carry_work_once():
    carry = MsjCarryIn(MsjLifecycleJob(0, 0, 10, outcome="cancelled"), 8)
    result = run(
        [MsjLifecycleJob(0, 0, 4), MsjLifecycleJob(4, 1, 1)], initial_running=[carry], observation_window=(1, 3)
    )
    assert result.utilization == 0.75
    assert result.observation_time == 2
    assert sum(result.resource_time_by_outcome.values()) == 8


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "2"])
def test_invalid_budgets(budget):
    with pytest.raises(ValueError):
        run([MsjLifecycleJob(0, 0, 2, runtime_limit=budget)])


@pytest.mark.parametrize("age", [-1, 2, 3, float("nan"), True])
def test_invalid_initial_age(age):
    with pytest.raises(ValueError):
        run([MsjLifecycleJob(1, 0, 1)], initial_running=[MsjCarryIn(MsjLifecycleJob(0, 0, 2), age)])


@pytest.mark.parametrize("window", [(-1, 1), (2, 1), (0, 3), (0,), (False, 1)])
def test_invalid_observation_window(window):
    with pytest.raises(ValueError):
        run([MsjLifecycleJob(0, 0, 1), MsjLifecycleJob(2, 0, 1)], observation_window=window)


def test_invalid_initial_capacity_and_contract():
    trace = [MsjLifecycleJob(1, 0, 1)]
    carry = MsjCarryIn(MsjLifecycleJob(0, 1, 2), 1)
    with pytest.raises(ValueError):
        run(trace, initial_running=[carry, carry])
    for job in (replace(carry.job, arrival=1), replace(carry.job, runtime_limit=1)):
        with pytest.raises(ValueError):
            run(trace, initial_waiting=[job])
    with pytest.raises(ValueError):
        run([MsjLifecycleJob(0, 0, 1, outcome="unknown")])
    with pytest.raises(ValueError):
        run([])
    with pytest.raises(ValueError):
        run(trace, initial_running=[trace[0]])
    with pytest.raises(ValueError):
        MsjLifecycleSim(2, "server_filling")
