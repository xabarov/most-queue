"""Exact capacity-calendar schedules and calendar primitive, no external data."""

import numpy as np
import pytest

from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.sim.msj_lifecycle import LIFECYCLE_POLICIES, MsjLifecycleJob, MsjLifecycleSim
from most_queue.sim.utils.msj_capacity_calendar import CapacityCalendar, constant_calendar, daily_calendar


def run(trace, calendar, *, capacity=2, needs=(1, 2), policy="fcfs"):
    sim = MsjLifecycleSim(capacity, policy)
    sim.set_servers(needs)
    return sim.run_capacity_calendar(trace, calendar)


# --- CapacityCalendar primitive -------------------------------------------------


def test_calendar_rejects_nonincreasing_or_mismatched_breakpoints():
    with pytest.raises(ValueError):
        CapacityCalendar((0.0, 0.0), (2, 1), 10.0)
    with pytest.raises(ValueError):
        CapacityCalendar((1.0, 0.0), (2, 1), 10.0)
    with pytest.raises(ValueError):
        CapacityCalendar((0.0,), (2, 1), 10.0)
    with pytest.raises(ValueError):
        CapacityCalendar((), (), 10.0)
    with pytest.raises(ValueError):
        CapacityCalendar((0.0, 1.0), (2, -1), 10.0)
    with pytest.raises(ValueError):
        CapacityCalendar((0.0, 1.0), (2, 1), 1.0)  # domain_end must exceed last breakpoint


def test_calendar_accepts_zero_capacity_as_a_genuine_drained_day():
    cal = CapacityCalendar((0.0, 1.0), (2, 0), 10.0)
    assert cal.capacity_at(1.0) == 0
    assert cal.max_capacity() == 2


def test_calendar_capacity_at_is_right_continuous_and_bounded_to_domain():
    cal = CapacityCalendar((0.0, 5.0, 8.0), (3, 1, 2), 10.0)
    assert [cal.capacity_at(t) for t in (0.0, 4.999, 5.0, 7.999, 8.0, 10.0)] == [3, 3, 1, 1, 2, 2]
    assert cal.max_capacity() == 3
    assert cal.start == 0.0
    with pytest.raises(ValueError):
        cal.capacity_at(-0.001)
    with pytest.raises(ValueError):
        cal.capacity_at(10.001)


def test_calendar_breakpoints_between_excludes_endpoint_already_in_effect():
    cal = CapacityCalendar((0.0, 5.0, 8.0), (3, 1, 2), 10.0)
    assert cal.breakpoints_between(0.0, 10.0) == [5.0, 8.0]
    assert cal.breakpoints_between(5.0, 10.0) == [8.0]  # 5.0 already in effect at "now"==5.0
    assert cal.breakpoints_between(8.0, 10.0) == []
    assert cal.breakpoints_between(6.0, 7.0) == []
    with pytest.raises(ValueError):
        cal.breakpoints_between(5.0, 4.0)


def test_constant_calendar_is_a_single_breakpoint():
    cal = constant_calendar(4, 0.0, 100.0)
    assert cal.capacities == (4,)
    assert cal.capacity_at(50.0) == 4


def test_daily_calendar_rejects_gaps_and_builds_day_boundaries():
    cal = daily_calendar({0: 5, 1: 3, 2: 7}, day_seconds=10.0)
    assert cal.times == (0.0, 10.0, 20.0)
    assert cal.domain_end == 30.0
    assert cal.capacity_at(9.999) == 5
    assert cal.capacity_at(10.0) == 3
    with pytest.raises(ValueError):
        daily_calendar({0: 5, 2: 7})  # day 1 missing
    with pytest.raises(ValueError):
        daily_calendar({})


# --- run_capacity_calendar: regression against the unbounded dispatchers --------


@pytest.mark.parametrize("policy", LIFECYCLE_POLICIES)
def test_constant_wide_calendar_matches_run_trace(policy):
    rng = np.random.default_rng(571)
    trace = tuple(
        MsjLifecycleJob(float(i), int(c), float(s), 4.0)
        for i, c, s in zip(np.cumsum(rng.exponential(2, 40)), rng.integers(0, 2, 40), rng.lognormal(1, 1, 40))
    )
    cal = constant_calendar(2, 0.0, 10_000.0)
    actual = run(trace, cal, policy=policy)
    sim = MsjGeneralSim(2, policy)
    sim.set_servers([1, 2])
    expected = sim.run_trace(trace)
    assert actual.start_times == expected.start_times
    assert actual.release_times == expected.completion_times
    assert actual.utilization == expected.utilization
    assert actual.idle_with_queue == expected.idle_with_queue
    assert actual.reservation_violations == expected.reservation_violations
    assert actual.reserved_start_times == expected.reserved_start_times
    assert actual.status == ["completed"] * len(trace)
    assert actual.infeasible_indices == []
    assert actual.reservation_mismatch_time is None


# --- Grandfathering: a capacity drop never preempts an active job ---------------


def test_capacity_drop_does_not_preempt_running_job_and_blocks_new_starts():
    # A 2-need job starts at t=0 under capacity 2; at t=1 capacity drops to 1,
    # which must not kill it, but must block the 1-need job waiting behind it.
    cal = CapacityCalendar((0.0, 1.0), (2, 1), 20.0)
    trace = [MsjLifecycleJob(0, 1, 5), MsjLifecycleJob(0.5, 0, 1)]
    result = run(trace, cal)
    assert result.start_times[0] == 0  # the 2-need job started before the drop
    assert result.status[0] == "completed"
    assert result.release_times[0] == 5
    # the 1-need job cannot start until the wide job frees both servers at t=5
    assert result.start_times[1] == 5
    assert result.status[1] == "completed"


def test_capacity_event_fires_without_any_new_arrival():
    # One job running; capacity rises mid-service with nothing else happening.
    # The calendar breakpoint must still be a loop event (no crash, no stall).
    cal = CapacityCalendar((0.0, 3.0), (1, 5), 20.0)
    trace = [MsjLifecycleJob(0, 0, 10)]
    result = run(trace, cal, capacity=5, needs=(1,))
    assert result.status == ["completed"]
    assert result.release_times == [10]


# --- Infeasible demand: excluded from dispatch, never blocks others -------------


def test_infeasible_head_of_line_job_does_not_block_fcfs_followers():
    cal = constant_calendar(2, 0.0, 20.0)
    trace = [MsjLifecycleJob(0, 2, 1), MsjLifecycleJob(0.1, 0, 1)]  # class 2 needs 3 > calendar ceiling 2
    result = run(trace, cal, capacity=3, needs=(1, 2, 3), policy="fcfs")
    assert result.status == ["infeasible", "completed"]
    assert result.infeasible_indices == [0]
    assert result.start_times[1] == 0.1
    assert result.resource_time_by_status["infeasible"] == 0.0


def test_infeasible_against_max_calendar_capacity_not_initial_capacity():
    # Need 3 never fits in a calendar whose ceiling is 2, even though an
    # earlier, now-irrelevant breakpoint reads higher than the current one.
    cal = CapacityCalendar((0.0, 1.0), (2, 2), 20.0)
    trace = [MsjLifecycleJob(0, 0, 1)]
    result = run(trace, cal, needs=(3,), capacity=3)
    assert result.status == ["infeasible"]
    assert result.start_times == [np.nan] or np.isnan(result.start_times[0])


# --- Unresolved / nonterminal results at the horizon -----------------------------


def test_job_still_waiting_at_domain_end_is_unresolved_not_force_completed():
    # Capacity 1 the whole time; a 1-need job occupies the only server for the
    # entire bounded horizon, so a second 1-need job never gets to start.
    cal = constant_calendar(1, 0.0, 5.0)
    trace = [MsjLifecycleJob(0, 0, 100), MsjLifecycleJob(0.1, 0, 1)]
    result = run(trace, cal, capacity=1, needs=(1,))
    assert result.status == ["running", "waiting"]
    assert result.horizon_end == 5.0
    assert np.isnan(result.release_times[1])
    assert result.resource_time_by_status["waiting"] == 0.0


def test_job_still_running_at_domain_end_is_nonterminal_with_partial_work():
    cal = constant_calendar(1, 0.0, 5.0)
    trace = [MsjLifecycleJob(0, 0, 100)]
    result = run(trace, cal, capacity=1, needs=(1,))
    assert result.status == ["running"]
    assert result.horizon_end == 5.0
    assert result.release_times[0] == 100  # scheduled completion, beyond the horizon
    assert result.resource_time_by_status["running"] == 5.0  # only elapsed work counts


# --- Domain validation: no forward/backfill beyond the known calendar -----------


@pytest.mark.parametrize(
    "arrival",
    [-0.001, 10.001],
)
def test_trace_arrival_outside_calendar_domain_raises(arrival):
    cal = constant_calendar(2, 0.0, 10.0)
    with pytest.raises(ValueError):
        run([MsjLifecycleJob(arrival, 0, 1)], cal)


def test_run_capacity_calendar_requires_a_calendar_instance():
    with pytest.raises(ValueError):
        run([MsjLifecycleJob(0, 0, 1)], calendar=5.0)


# --- Conservative reservations vs a shrinking calendar: explicit mismatch -------


def test_conservative_reservation_invalidated_by_capacity_drop_is_an_explicit_mismatch():
    # Under capacity 3, Conservative reserves the 2-need job J2 a start at
    # t=4, once the shorter 1-need job J1 frees a server (J0 keeps running).
    # A drop to capacity 2 at t=2 means J2 now needs BOTH J0 and J1 gone
    # (1 + 2 > 2), pushing its only feasible start to t=100 (J0's release) --
    # later than the t=4 already promised, which compress_reservations
    # detects and refuses to silently accept.
    cal = CapacityCalendar((0.0, 2.0), (3, 2), 20.0)
    trace = [
        MsjLifecycleJob(0, 0, 100, estimate=100),
        MsjLifecycleJob(0, 0, 4, estimate=4),
        MsjLifecycleJob(0, 1, 5, estimate=5),
    ]
    result = run(trace, cal, capacity=3, needs=(1, 2), policy="conservative")
    assert result.reservation_mismatch_time == 2.0
    assert result.status[2] == "waiting"  # the mismatch froze J2 before it ever started


def test_easy_adapts_to_capacity_drop_without_a_dedicated_mismatch_path():
    # EASY only protects the head job's own shadow, recomputed fresh every
    # dispatch call, so it naturally tracks a shrinking calendar.
    cal = CapacityCalendar((0.0, 1.0), (2, 1), 20.0)
    trace = [MsjLifecycleJob(0, 1, 5, estimate=5), MsjLifecycleJob(0.1, 0, 1, estimate=1)]
    result = run(trace, cal, policy="easy")
    assert result.reservation_mismatch_time is None
    assert "completed" in result.status
