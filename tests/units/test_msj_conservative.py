"""Exact schedules and independent invariants for conservative reservations."""

from dataclasses import replace

import numpy as np
import pytest

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.msj_calendar import Reservation, first_fit


def replay(trace, needs=(1, 2), k=2, discipline="conservative"):
    """Replay without random service generation."""
    sim = MsjGeneralSim(k, discipline)
    sim.set_servers(needs)
    return sim.run_trace(trace)


def assert_capacity(trace, result, needs, capacity):
    """Check actual intervals, independently of the reservation calendar."""
    changes = []
    for job, start, end in zip(trace, result.start_times, result.completion_times):
        assert start >= job.arrival
        assert end - start == pytest.approx(job.service)
        changes.extend([(start, needs[job.cls]), (end, -needs[job.cls])])
    used = 0
    for _, delta in sorted(changes):
        used += delta
        assert 0 <= used <= capacity
    assert used == 0


def test_conservative_protects_second_job_not_only_head():
    trace = [
        MsjTraceJob(0, 0, 10, 10),
        MsjTraceJob(1, 0, 1, 1),
        MsjTraceJob(1.5, 1, 1, 1),
        MsjTraceJob(2, 2, 20, 20),
    ]
    easy = replay(trace, [3, 4, 1], 4, "easy")
    conservative = replay(trace, [3, 4, 1], 4)
    assert easy.start_times == [0, 10, 22, 2]
    assert conservative.start_times == [0, 10, 11, 12]
    assert conservative.reservations == 3
    assert conservative.reservation_violations == 0
    assert_capacity(trace, conservative, [3, 4, 1], 4)


def test_early_completion_preserves_a_future_backfill_reservation():
    # At t=2: head C is reserved at 10, D at 5. Restarting arrival-order
    # scheduling at t=3 would move C to 5 and D to 15, breaking D's promise.
    trace = [
        MsjTraceJob(0, 0, 3, 10),
        MsjTraceJob(0, 1, 5, 5),
        MsjTraceJob(1, 2, 10, 10),
        MsjTraceJob(2, 1, 3, 3),
    ]
    result = replay(trace, [3, 1, 4], 4)
    assert result.start_times == [0, 0, 6, 3]
    assert result.reserved_start_times == {2: 6, 3: 3}
    assert result.backfilled == 1
    assert result.reservation_violations == 0


def test_safe_backfill_and_long_surplus_job():
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 2, 2)]
    assert replay(trace).start_times == [0, 10, 2]
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 0, 1, 1), MsjTraceJob(2, 1, 20, 20)]
    assert replay(trace, [3, 1], 4).start_times == [0, 10, 2]


def test_reservation_timer_detects_overrun_without_arrival_or_completion():
    trace = [
        MsjTraceJob(0, 0, 10, 5),
        MsjTraceJob(1, 1, 1, 1),
        MsjTraceJob(2, 0, 1, 1),
        MsjTraceJob(6, 0, 1, 1),
    ]
    result = replay(trace)
    assert result.start_times == [0, 10, 2, 11]
    assert result.reserved_start_times[1] == 5
    assert result.reservation_violations == 1
    assert result.backfilled == 1  # no new backfill at t=6 while overdue


def test_underpredicted_backfill_not_given_hidden_oracle_information():
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 20, 2)]
    result = replay(trace)
    assert result.start_times == [0, 22, 2]
    assert result.reservation_violations == 1
    assert result.reserved_start_times[1] == 10
    assert sum(result.counts_per_class) == 3  # no termination on overrun


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("forecast", ["oracle", "upper", "noisy"])
def test_seeded_capacity_and_upper_bound_guarantees(seed, forecast):
    rng = np.random.default_rng(seed)
    trace = [
        MsjTraceJob(float(a), int(c), float(s), float(s * factor))
        for a, c, s, factor in zip(
            np.cumsum(rng.exponential(0.8, 200)),
            rng.integers(0, 3, 200),
            rng.lognormal(0, 1, 200),
            np.ones(200) if forecast == "oracle" else rng.uniform(1 if forecast == "upper" else 0.2, 3, 200),
        )
    ]
    result = replay(trace, [1, 2, 4], 4)
    assert_capacity(trace, result, [1, 2, 4], 4)
    assert sum(result.counts_per_class) == len(trace)
    if forecast != "noisy":
        assert result.reservation_violations == 0
        assert all(result.start_times[idx] <= promised for idx, promised in result.reserved_start_times.items())


def test_all_resources_jobs_reduce_to_fcfs_even_with_loose_bounds():
    rng = np.random.default_rng(48)
    trace = tuple(
        MsjTraceJob(float(a), 0, float(s), float(3 * s))
        for a, s in zip(np.cumsum(rng.exponential(0.8, 100)), rng.gamma(2, 0.5, 100))
    )
    cons, fcfs = replay(trace, [4], 4), replay(trace, [4], 4, "fcfs")
    assert cons.start_times == fcfs.start_times
    assert cons.w == fcfs.w
    assert cons.reservation_violations == cons.backfilled == 0


def test_estimates_required_and_replay_state_is_not_reused():
    sim = MsjGeneralSim(2, "conservative")
    sim.set_servers([1, 2])
    trace = [MsjTraceJob(0, 0, 2, 2), MsjTraceJob(1, 1, 1, 1)]
    with pytest.raises(ValueError, match="explicit estimates"):
        sim.run_trace([replace(trace[0], estimate=None)])
    first, second = sim.run_trace(trace), sim.run_trace(trace)
    assert first.reserved_start_times == second.reserved_start_times == {1: 2}
    assert first.start_times == second.start_times


def test_calendar_checks_whole_interval_and_half_open_boundaries():
    profile = {0: Reservation(0, 2, 2), 1: Reservation(4, 6, 4)}
    assert first_fit(4, profile, 2, 4, 0) == Reservation(0, 4, 2)
    assert first_fit(4, profile, 3, 2, 0) == Reservation(2, 4, 3)
    assert first_fit(4, profile, 3, 3, 0) == Reservation(6, 9, 3)


def test_unrepresentable_forecast_calendar_is_rejected():
    profile = {0: Reservation(0, 1e308, 1)}
    with pytest.raises(ValueError, match="overflows"):
        first_fit(1, profile, 1, 1e308, 0)
    with pytest.raises(ValueError, match="precision"):
        first_fit(1, {}, 1, 1, 1e20)
