"""Causal runtime refresh, calendar invalidation and safe unavailable tails."""

from dataclasses import replace

import numpy as np
import pytest

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob


def replay(trace, policy="conservative", predictor=None):
    """Replay two resource classes with optional runtime refresh."""
    sim = MsjGeneralSim(2, policy)
    sim.set_servers([1, 2])
    return sim.run_trace(trace, remaining_predictor=predictor)


@pytest.mark.parametrize("policy", ["easy", "conservative"])
def test_fixed_endpoint_callback_preserves_static_schedule(policy):
    """A consistent fixed absolute endpoint preserves existing behavior."""
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 2, 10)]
    static = replay(trace, policy)
    refreshed = replay(trace, policy, lambda cls, age: [10, 1][cls] - age)
    assert refreshed.start_times == static.start_times
    assert refreshed.reserved_start_times == static.reserved_start_times
    assert refreshed.runtime_updates > 0
    assert refreshed.forecast_calendar_resets == 0
    assert static.runtime_updates == static.unavailable_runtime_updates == 0


def test_extended_forecast_rebuilds_calendar_without_erasing_old_promises():
    """Longer active forecasts cannot silently reset the promise baseline."""
    trace = [MsjTraceJob(0, 0, 10, 2), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 1, 1)]
    result = replay(trace, predictor=lambda cls, age: 2)
    assert result.reserved_start_times[1] == 3  # best promise, not last rebuilt promise
    assert result.start_times[1] == 10
    assert result.reservation_violations >= 1
    assert result.forecast_calendar_resets > 0
    assert len(result.completion_times) == 3


@pytest.mark.parametrize("policy", ["easy", "conservative"])
def test_none_suspends_backfill_but_keeps_fcfs_drain_and_historical_promises(policy):
    """An unknown tail must neither enable backfill nor discard jobs."""
    trace = [MsjTraceJob(0, 0, 10, 4), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 1, 1)]
    result = replay(trace, policy, lambda cls, age: 4 - age if age < 2 else None)
    assert result.start_times == [0, 10, 11]
    assert result.reserved_start_times[1] == 4
    assert result.reservation_violations >= 1
    assert result.backfilled == 0
    assert result.unavailable_runtime_updates > 0
    assert result.runtime_updates > result.unavailable_runtime_updates


def test_callback_receives_service_age_not_wait_or_arrival_and_completions_win_ties():
    """Do not predict completed jobs or count waiting time as service age."""
    seen = []

    def predict(cls, age):
        seen.append((cls, age))
        return 10

    trace = [MsjTraceJob(0, 1, 4, 4), MsjTraceJob(1, 0, 2, 2), MsjTraceJob(4, 0, 2, 2), MsjTraceJob(5, 0, 1, 1)]
    result = replay(trace, predictor=predict)
    assert result.start_times[:3] == [0, 4, 4]
    assert seen == [(1, 1.0), (0, 1.0), (0, 1.0)]
    assert result.runtime_updates == 3


@pytest.mark.parametrize("policy", ["easy", "conservative"])
def test_unobserved_future_service_does_not_change_prefix_decisions(policy):
    """Identical observed prefixes must induce identical prefix decisions."""
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 2, 2), MsjTraceJob(2, 0, 1, 1)]
    changed = [replace(trace[0], service=100), *trace[1:]]
    results = [replay(t, policy, lambda cls, age: 10 if cls == 0 else 2) for t in (trace, changed)]
    assert results[0].start_times[2] == results[1].start_times[2] == 2
    assert [s for s in results[0].start_times if s < 10] == [s for s in results[1].start_times if s < 10]
    # Final best promises can differ: compression after the observed early
    # completion at 10 is legitimate, not information available at time 2.
    assert results[0].start_times[1] != results[1].start_times[1]  # observed completions later differ


@pytest.mark.parametrize("policy", ["easy", "conservative"])
@pytest.mark.parametrize("seed", range(4))
def test_seeded_refresh_preserves_capacity_and_every_job(policy, seed):
    """Exercise expanding and missing forecasts on mixed-resource traces."""
    rng = np.random.default_rng(seed)
    trace = [
        MsjTraceJob(float(a), int(c), float(s), 2)
        for a, c, s in zip(np.cumsum(rng.exponential(0.5, 100)), rng.integers(2, size=100), rng.lognormal(size=100))
    ]
    result = replay(trace, policy, lambda cls, age: None if age > 4 else 2 + age)
    changes = []
    for job, start, done in zip(trace, result.start_times, result.completion_times):
        assert start >= job.arrival
        assert done - start == pytest.approx(job.service)
        changes.extend([(start, job.cls + 1), (done, -job.cls - 1)])
    used = 0
    for _, delta in sorted(changes):
        used += delta
        assert 0 <= used <= 2
    assert sum(result.counts_per_class) == 100


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "1", 1j])
def test_invalid_residual_is_not_silently_replaced(value):
    """Reject programmer errors rather than silently treating them as None."""
    with pytest.raises(ValueError, match="remaining_predictor"):
        replay([MsjTraceJob(0, 0, 10, 2), MsjTraceJob(1, 1, 1, 1)], predictor=lambda cls, age: value)


def test_refresh_validates_timestamp_precision_and_options():
    """Sub-ULP positive residuals safely freeze; invalid API options fail."""
    trace = [MsjTraceJob(0, 0, 10, 2), MsjTraceJob(1, 1, 1, 1)]
    result = replay(trace, predictor=lambda cls, age: 1e-20)
    assert result.start_times == [0, 10]
    assert result.unavailable_runtime_updates == 1
    huge = [MsjTraceJob(1e308, 0, 3e307, 3e307), MsjTraceJob(1.1e308, 1, 1e307, 1e307)]
    with pytest.raises(ValueError, match="overflows"):
        replay(huge, predictor=lambda cls, age: 1e308)
    for policy, predictor in (("fcfs", lambda cls, age: 1), ("easy", 1)):
        with pytest.raises(ValueError, match="remaining_predictor"):
            replay(trace, policy, predictor)


def test_run_forwards_callback_and_replay_resets_counters():
    """The generated-trace API forwards refresh and does not retain counters."""
    sim = MsjGeneralSim(2, "easy", seed=50)
    sim.set_servers([1, 2], [(1, "M"), (1, "M")])
    sim.set_sources([1, 1])
    generated = sim.run(50, estimates=[2, 2], remaining_predictor=lambda cls, age: None)
    assert generated.runtime_updates == generated.unavailable_runtime_updates > 0
    result = sim.run_trace([MsjTraceJob(0, 0, 1, 1)], remaining_predictor=lambda cls, age: 1)
    assert result.runtime_updates == result.forecast_calendar_resets == 0
