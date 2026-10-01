"""Exact trace schedules test EASY semantics independently of Monte Carlo."""

from dataclasses import replace

import numpy as np
import pytest

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob


def replay(trace, needs=(1, 2), k=2, discipline="easy", **kwargs):
    sim = MsjGeneralSim(k, discipline)
    sim.set_servers(needs)
    return sim.run_trace(trace, **kwargs)


def test_safe_backfill_and_fcfs_blocking():
    trace = (MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 2, 2))
    fcfs, easy = replay(trace, discipline="fcfs"), replay(trace)
    assert fcfs.start_times == [0, 10, 11]
    assert easy.start_times == [0, 10, 2]
    assert easy.backfilled == 1 and easy.reservation_violations == 0
    assert easy.reservations == 1
    assert easy.v_per_class[0] < fcfs.v_per_class[0]
    assert trace[2].arrival == 2  # input not mutated


def test_unsafe_backfill_is_rejected():
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 9, 9)]
    result = replay(trace)
    assert result.start_times == [0, 10, 11]
    assert result.backfilled == 0


def test_multiple_backfills_share_capacity_and_protect_the_head():
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1)]
    trace += [MsjTraceJob(2, 2, 2, 2)] * 3
    result = replay(trace, needs=[2, 4, 1], k=4)
    assert result.start_times == [0, 10, 2, 2, 4]
    assert result.reservation_violations == 0
    changes = []
    for job, start, finish in zip(trace, result.start_times, result.completion_times):
        need = [2, 4, 1][job.cls]
        changes.extend([(start, need), (finish, -need)])
    used = 0
    for _, delta in sorted(changes):  # negative completion changes win ties
        used += delta
        assert 0 <= used <= 4


def test_long_backfill_can_use_surplus_capacity():
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 2, 20, 20)]
    result = replay(trace, needs=[3, 3, 1], k=4)
    assert result.start_times == [0, 10, 2]
    assert result.reservation_violations == 0


def test_underprediction_is_observed_not_oracle_corrected():
    trace = [MsjTraceJob(0, 0, 10, 10), MsjTraceJob(1, 1, 1, 1), MsjTraceJob(2, 0, 20, 2)]
    result = replay(trace)
    assert result.start_times == [0, 22, 2]  # scheduler only saw duration estimate 2
    assert result.reservation_violations == 1
    assert sum(result.counts_per_class) == 3  # no job killed


def test_overdue_running_job_freezes_new_backfill():
    trace = [MsjTraceJob(0, 0, 10, 1), MsjTraceJob(2, 1, 1, 1), MsjTraceJob(3, 0, 1, 1)]
    result = replay(trace)
    assert result.start_times == [0, 10, 11]
    assert result.backfilled == 0


def test_all_completions_at_tie_release_resources_before_arrivals():
    trace = [MsjTraceJob(0, 0, 2, 2), MsjTraceJob(0, 0, 2, 2), MsjTraceJob(2, 1, 1, 1)]
    result = replay(trace)
    assert result.start_times == [0, 0, 2]
    assert result.utilization == pytest.approx(1)
    assert result.p[2] == pytest.approx(1)


def test_time_average_and_arrival_cohort_drain():
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 1, 1), MsjTraceJob(2, 0, 2)]
    result = replay(trace, discipline="fcfs", warmup_jobs=1)
    assert result.start_times == [0, 10, 11]
    assert result.wait_samples == [9, 9]
    assert result.sojourn_samples == [10, 11]
    assert result.observation_time == 1
    assert result.utilization == pytest.approx(0.5)
    assert result.idle_with_queue == pytest.approx(0.5)
    assert result.p == pytest.approx([0, 0, 1])
    assert result.throughput == 0  # no departures during [1, 2]


def test_one_arrival_has_no_time_average_and_missing_class_is_nan():
    result = replay([MsjTraceJob(0, 0, 1, 1)])
    assert result.utilization is None and result.p is None
    assert result.counts_per_class == [1, 0]
    assert np.isnan(result.v_per_class[1])
    assert np.isnan(result.v_quantiles_per_class[1][0.99])


def test_seeded_generators_make_identical_traces_across_policies():
    traces = []
    for discipline in ("fcfs", "easy"):
        sim = MsjGeneralSim(2, discipline, seed=10)
        sim.set_sources([0.2, 0.1])
        sim.set_servers([1, 2], [(1, "M"), lambda rng: rng.lognormal(0, 0.5)])
        traces.append(sim.make_trace(100, estimates="oracle"))
    assert traces[0] == traces[1]
    assert all(job.service == job.estimate for job in traces[0])


def test_prediction_does_not_change_sampled_work_and_deadline_api():
    traces = []
    for estimates in (None, "oracle", [2]):
        sim = MsjGeneralSim(1, seed=42)
        sim.set_sources([0.5])
        sim.set_servers([1], [(1, "M")])
        traces.append(sim.make_trace(30, estimates))
    assert [replace(job, estimate=None) for job in traces[1]] == list(traces[0])
    assert [replace(job, estimate=None) for job in traces[2]] == list(traces[0])
    sim.set_deadline_thresholds([1])
    result = sim.run_trace(traces[0], warmup_jobs=2)
    assert sim.get_empirical_violation_prob(1) == pytest.approx(np.mean(np.array(result.wait_samples) > 1))
    sim.run_trace(traces[0], warmup_jobs=2)
    assert sim.deadline_n == 28  # repeated replay resets counters


@pytest.mark.parametrize(
    "trace",
    [
        [],
        [MsjTraceJob(-1, 0, 1, 1)],
        [MsjTraceJob(0, 2, 1, 1)],
        [MsjTraceJob(0, True, 1, 1)],
        [MsjTraceJob(0, 0, 0, 1)],
        [MsjTraceJob(0, 0, np.nan, 1)],
        [MsjTraceJob(0, 0, 1, -1)],
        [MsjTraceJob(0, 0, 1)],
        [MsjTraceJob(0, 0, 1, np.inf)],
        [MsjTraceJob(2, 0, 1, 1), MsjTraceJob(1, 0, 1, 1)],
    ],
)
def test_bad_trace_rejected(trace):
    with pytest.raises(ValueError):
        replay(trace)


def test_configuration_validation():
    with pytest.raises(ValueError):
        MsjGeneralSim(2, "unknown")
    for k in (0, True, 1.5):
        with pytest.raises(ValueError):
            MsjGeneralSim(k)
    sim = MsjGeneralSim(2)
    with pytest.raises(ValueError):
        sim.run(10)
    with pytest.raises(ValueError):
        sim.set_sources([0])
    with pytest.raises(ValueError):
        sim.set_servers([3])
    sim.set_sources([0.2])
    sim.set_servers([1], [(1, "M")])
    for estimates in ("automatic", [-1], [1, 2], [np.nan]):
        with pytest.raises(ValueError):
            sim.make_trace(10, estimates)
    for warmup in (-1, 1, np.nan):
        with pytest.raises(ValueError):
            sim.run(10, warmup)
    for warmup in (-1, 1, True):
        with pytest.raises(ValueError):
            sim.run_trace([MsjTraceJob(0, 0, 1)], warmup_jobs=warmup)


def test_run_measures_requested_arrival_count():
    sim = MsjGeneralSim(2, "easy", seed=123)
    sim.set_sources([0.2, 0.2])
    sim.set_servers([1, 2], [(1, "M"), (2, "D")])
    result = sim.run(100, warmup_fraction=0.1, estimates="oracle")
    assert sum(result.counts_per_class) == 100
    assert len(result.start_times) == 110
    assert result.reservation_violations == 0
