"""Useful-service protection: real review events, information and path audits."""

from dataclasses import fields, replace

import numpy as np
import pytest

from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.sim.msj_general import MsjTraceJob
from most_queue.structs import MsjCheckpointResults
from tests.units.test_msj_checkpoint import tick_reference


def replay(jobs, protection=2, costs=(0.5, 0.25)):
    """Replay one-or-all work with an explicit useful protection interval."""
    sim = MsjCheckpointSim(4, *costs, min_service_time=protection)
    sim.set_servers([1, 4])
    return sim.run_trace([MsjTraceJob(*job) for job in jobs])


def test_protection_expires_without_an_external_event():
    """A waiting wide job starts at expiry plus c, not at the next completion."""
    result = replay([(0, 0, 10), (1, 1, 2)])
    assert result.start_times == [0, 2.5]
    assert result.completion_times == [12.75, 4.5]
    assert result.wait_samples == [2.75, 1.5]
    assert result.checkpoint_segments == [(0, 2, 2.5)]
    assert result.resume_segments == [(0, 4.5, 4.75)]
    assert result.protected_preemptions == result.protection_expirations == 1


def test_protection_clock_resets_after_resume_not_during_it():
    """The second preemption needs q NEW useful service after restoration."""
    result = replay([(0, 0, 10), (1, 1, 2), (5, 1, 1)])
    assert result.checkpoint_segments == [(0, 2, 2.5), (0, 6.75, 7.25)]
    assert result.resume_segments == [(0, 4.5, 4.75), (0, 8.25, 8.5)]
    assert result.start_times == [0, 2.5, 7.25]
    assert result.completion_times == [14.5, 4.5, 8.25]
    assert result.preemptions_per_job == [2, 0, 0]
    assert result.protection_expirations == 2


@pytest.mark.parametrize("service", [1.5, 2])
def test_completion_cancels_or_wins_a_protection_timer(service):
    """Finish before/at eligibility, without a fictitious checkpoint or work."""
    result = replay([(0, 0, service), (1, 1, 2)])
    assert result.start_times == [0, service]
    assert result.completion_times == [service, service + 2]
    assert not result.preemptions and not result.checkpoint_segments
    assert result.protection_expirations == int(service == 2)


def test_expiry_and_arrival_tie_dispatch_all_arrivals_together():
    """The earlier wide job keeps FIFO order when another arrives at expiry."""
    result = replay([(0, 0, 10), (1, 1, 2), (2, 1, 1)])
    assert result.start_times == [0, 2.5, 4.5]
    assert result.preemptions_per_job == [1, 0, 0]
    assert result.protection_expirations == 1


def test_zero_overhead_does_not_disable_positive_protection():
    """Only c=r=q=0 delegates; q still changes a free-preemption schedule."""
    result = replay([(0, 0, 10), (1, 1, 2)], costs=(0, 0))
    assert result.start_times == [0, 2]
    assert result.completion_times == [12, 4]
    assert result.preemptions == result.protection_expirations == 1
    assert result.checkpoint_resource_time == result.resume_resource_time == 0


def test_expiry_is_not_a_forced_time_slice_switch():
    """Selected service is uninterrupted; it incurs no unnecessary timer."""
    result = replay([(0, 0, 10), (1, 0, 10), (2, 0, 10)], protection=0.5)
    assert result.preemptions == result.protected_preemptions == result.protection_expirations == 0
    assert result.completion_times == [10, 11, 12]
    assert len(result.service_segments) == 3


@pytest.mark.parametrize("costs", [(0, 0), (1, 0), (0, 2), (1, 2)])
@pytest.mark.parametrize("protection", [1, 3, 20])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_independent_tick_reference_with_protection(k, protection, costs):
    """Counter-based oracle uses integer useful ticks, not DES timestamps."""
    needs = [2**i for i in range(k.bit_length())]
    for seed in range(6):
        rng = np.random.default_rng(540 + seed)
        arrivals = np.cumsum(rng.integers(0, 3, 35)).tolist()
        classes = rng.integers(0, len(needs), 35).tolist()
        services = rng.integers(1, 9, 35).tolist()
        sim = MsjCheckpointSim(k, *costs, min_service_time=protection)
        sim.set_servers(needs)
        result = sim.run_trace([MsjTraceJob(a, c, s) for a, c, s in zip(arrivals, classes, services)])
        expected = tick_reference(k, [needs[c] for c in classes], arrivals, services, (*costs, protection))
        assert (
            result.start_times,
            result.completion_times,
            result.wait_samples,
            result.preemptions_per_job,
        ) == expected


@pytest.mark.parametrize("seed", range(6))
def test_nonexponential_paths_conserve_work_and_bound_preemption(seed):
    """Protection uses useful time; all logged allocations respect capacity."""
    rng = np.random.default_rng(seed)
    trace = [
        MsjTraceJob(float(a), int(c), float(s))
        for a, c, s in zip(np.cumsum(rng.exponential(0.3, 180)), rng.integers(0, 3, 180), rng.lognormal(size=180))
    ]
    q = 0.3
    sim = MsjCheckpointSim(4, 0.1, 0.2, min_service_time=q)
    sim.set_servers([1, 2, 4])
    result = sim.run_trace(trace, 10)
    work, allocations, previous = np.zeros(len(trace)), [], {}
    for idx, start, end in result.service_segments:
        work[idx] += end - start
        previous[(idx, end)] = start
    for idx, start, _ in result.checkpoint_segments:
        assert start - previous[(idx, start)] >= q - 1e-12
        assert not any(a < start < b for _, a, b in result.checkpoint_segments + result.resume_segments)
    for idx, count in enumerate(result.preemptions_per_job):
        assert count * q <= trace[idx].service + 1e-12
    areas = []
    begin, finish = trace[10].arrival, trace[-1].arrival
    for segments in (result.service_segments, result.checkpoint_segments, result.resume_segments):
        area = 0.0
        for idx, start, end in segments:
            need = (1, 2, 4)[trace[idx].cls]
            allocations.extend([(start, need), (end, -need)])
            area += need * max(0, min(end, finish) - max(start, begin))
        areas.append(area / (4 * (finish - begin)))
    used = 0
    for _, change in sorted(allocations):
        used += change
        assert 0 <= used <= 4
    assert used == 0
    np.testing.assert_allclose(work, [job.service for job in trace], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        areas, [result.productive_utilization, result.checkpoint_utilization, result.resume_utilization]
    )
    np.testing.assert_allclose(result.wait_samples, np.array(result.first_wait_samples) + result.interruption_samples)
    np.testing.assert_allclose(
        result.wait_samples, np.array(result.queue_wait_samples) + result.checkpoint_samples + result.resume_samples
    )
    np.testing.assert_allclose(result.sojourn_samples, np.array(result.wait_samples) + work[10:])


def test_hidden_service_forecasts_and_replay_reset():
    """Eligibility depends on observed episode age, not future service."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 1, 20), MsjTraceJob(5, 1, 30)]
    sim = MsjCheckpointSim(4, 0.5, 0.25, min_service_time=2)
    sim.set_servers([1, 4])
    expected = sim.run_trace(trace)
    estimates = sim.run_trace([replace(job, estimate=1000) for job in trace])
    again = sim.run_trace(trace)
    for item in fields(MsjCheckpointResults):
        if item.name != "duration":
            assert getattr(again, item.name) == getattr(estimates, item.name) == getattr(expected, item.name)
    longer = sim.run_trace([replace(job, service=job.service * 2) for job in trace])
    assert expected.start_times[:2] == longer.start_times[:2] == [0, 2.5]
    assert expected.checkpoint_segments[0] == longer.checkpoint_segments[0]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1", None])
def test_invalid_protection_is_rejected(value):
    """Keep cost/protection validation consistent; no coercion or clipping."""
    with pytest.raises(ValueError, match="min_service_time"):
        MsjCheckpointSim(4, min_service_time=value)


def test_protection_below_timestamp_precision_fails_loudly():
    """No epsilon or silent rounding of a requested positive protection."""
    with pytest.raises(ValueError, match="precision"):
        replay([(1e16, 0, 20)], protection=0.1)


def test_old_positional_seed_and_new_keyword_are_independent():
    """Keep constructor compatibility and generator reproducibility."""
    sims = [MsjCheckpointSim(4, 0.1, 0.2, 54), MsjCheckpointSim(4, 0.1, 0.2, seed=54, min_service_time=0)]
    outputs = []
    for sim in sims:
        sim.set_servers([1, 4], [lambda rng: rng.gamma(2, 0.5)] * 2)
        sim.set_sources([0.3, 0.1])
        outputs.append(sim.run(150))
    assert outputs[0].completion_times == outputs[1].completion_times
    assert outputs[0].protected_preemptions == outputs[1].protection_expirations == 0
