"""Checkpoint accounting, exact traces and an independent tick-based oracle."""

from dataclasses import fields, replace

import numpy as np
import pytest

from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.structs import MsjSimulationResults


def replay(trace, checkpoint=0.5, resume=0.25, warmup=0):
    """Replay the three power-of-two classes with explicit overhead times."""
    sim = MsjCheckpointSim(4, checkpoint, resume)
    sim.set_servers([1, 2, 4])
    return sim.run_trace(trace, warmup_jobs=warmup)


def exact_trace():
    """A completion moves the prefix and interrupts job two exactly once."""
    return [MsjTraceJob(0, cls, service) for cls, service in ((1, 1), (0, 10), (1, 3), (2, 2))]


@pytest.mark.parametrize("checkpoint,resume", [(0, 0), (0.5, 0), (0, 0.25), (0.5, 0.25)])
def test_exact_checkpoint_resource_delay_and_resume(checkpoint, resume):
    """Checkpoint delays the preemptor; resume delays the interrupted job."""
    result = replay(exact_trace(), checkpoint, resume)
    assert result.start_times == [0, 3 + checkpoint, 0, 1 + checkpoint]
    assert result.completion_times == [1, 13 + checkpoint, 5 + checkpoint + resume, 3 + checkpoint]
    assert result.wait_samples == [0, 3 + checkpoint, 2 + checkpoint + resume, 1 + checkpoint]
    assert result.queue_wait_samples == [0, 3 + checkpoint, 2, 1 + checkpoint]
    assert result.first_wait_samples == [0, 3 + checkpoint, 0, 1 + checkpoint]
    assert result.interruption_samples == [0, 0, 2 + checkpoint + resume, 0]
    assert result.checkpoint_samples == [0, 0, checkpoint, 0]
    assert result.resume_samples == [0, 0, resume, 0]
    assert result.checkpoint_resource_time == 2 * checkpoint
    assert result.resume_resource_time == 2 * resume
    assert result.preemptions_per_job == [0, 0, 1, 0]
    assert result.utilization is result.productive_utilization is None


def test_checkpoint_holds_servers_until_finished():
    """The wide arrival cannot use resources still allocated to checkpoint."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 2, 2)]
    result = replay(trace)
    assert result.start_times == [0, 1.5]
    assert result.completion_times == [12.75, 3.5]
    assert result.checkpoint_segments == [(0, 1, 1.5)]
    assert result.resume_segments == [(0, 3.5, 3.75)]


@pytest.mark.parametrize("seed", range(4))
def test_zero_cost_matches_every_existing_result_field(seed):
    """Zero overhead delegates to the original SF, not an approximate limit."""
    source = MsjGeneralSim(4, "server_filling", seed)
    source.set_servers([1, 2, 4], [lambda rng: rng.lognormal()] * 3)
    source.set_sources([0.3, 0.2, 0.1])
    trace = source.make_trace(150)
    expected = source.run_trace(trace, 10)
    actual = replay(trace, 0, 0, 10)
    for item in fields(MsjSimulationResults):
        if item.name != "duration":
            assert getattr(actual, item.name) == getattr(expected, item.name), item.name


def tick_reference(k, needs, arrivals, services, costs):
    """Independent integer-time simulation, without the DES engine or selector.

    It decrements useful work/phase counters once per tick and dispatches only
    at actual events. All arrays are per-job; no event heap or actual end-time
    arithmetic is shared with the implementation under test.
    """
    checkpoint, resume = costs[:2]
    protection = costs[2] if len(costs) > 2 else 0
    count = len(arrivals)
    mode = ["future"] * count
    left = list(services)
    overhead = [0] * count
    useful_ticks = [0] * count
    starts, ends, waits, preemptions = [-1] * count, [-1] * count, [0] * count, [0] * count
    now = 0
    while any(end < 0 for end in ends):
        event = False
        for idx in range(count):
            if mode[idx] == "service" and left[idx] == 0:
                mode[idx], ends[idx], event = "done", now, True
            elif mode[idx] in ("checkpoint", "resume") and overhead[idx] == 0:
                useful_ticks[idx] = 0
                mode[idx], event = ("queue" if mode[idx] == "checkpoint" else "service"), True
            elif mode[idx] == "future" and arrivals[idx] == now:
                mode[idx], event = "queue", True
            elif mode[idx] == "service" and protection and useful_ticks[idx] == protection:
                event = True
        if event:
            prefix, total = [], 0
            for idx in range(count):
                if mode[idx] not in ("done", "future"):
                    prefix.append(idx)
                    total += needs[idx]
                    if total >= k:
                        break
            chosen, free = [], k
            for idx in sorted(prefix, key=lambda i: (-needs[i], i)):
                if needs[idx] <= free:
                    chosen.append(idx)
                    free -= needs[idx]
            if not any(value in ("checkpoint", "resume") for value in mode):
                for idx in range(count):
                    if mode[idx] == "service" and idx not in chosen and useful_ticks[idx] >= protection:
                        preemptions[idx] += 1
                        mode[idx] = "checkpoint" if checkpoint else "queue"
                        overhead[idx] = checkpoint
            free = k - sum(needs[idx] for idx in range(count) if mode[idx] in ("service", "checkpoint", "resume"))
            for idx in chosen:
                if mode[idx] == "queue" and needs[idx] <= free:
                    restored = starts[idx] >= 0
                    mode[idx] = "resume" if restored and resume else "service"
                    overhead[idx] = resume if restored else 0
                    starts[idx] = starts[idx] if restored else now
                    useful_ticks[idx] = 0
                    free -= needs[idx]
        for idx, value in enumerate(mode):
            if value == "service":
                left[idx] -= 1
                useful_ticks[idx] += 1
            elif value in ("checkpoint", "resume"):
                overhead[idx] -= 1
            if value in ("queue", "checkpoint", "resume"):
                waits[idx] += 1
        now += 1
        assert now < 100000, "reference failed to drain a finite trace"
    return starts, ends, waits, preemptions


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("checkpoint,resume", [(0, 0), (1, 0), (0, 2), (1, 2)])
def test_independent_integer_time_reference(k, checkpoint, resume):
    """Exact agreement at arrivals, completions and overhead ties, with gating."""
    needs = [2**i for i in range(k.bit_length())]
    for seed in range(6):
        rng = np.random.default_rng(seed)
        arrivals = np.cumsum(rng.integers(0, 3, 35)).tolist()
        classes = rng.integers(0, len(needs), 35).tolist()
        services = rng.integers(1, 9, 35).tolist()
        trace = [MsjTraceJob(a, c, s) for a, c, s in zip(arrivals, classes, services)]
        sim = MsjCheckpointSim(k, checkpoint, resume)
        sim.set_servers(needs)
        result = sim.run_trace(trace)
        starts, ends, waits, preemptions = tick_reference(
            k, [needs[cls] for cls in classes], arrivals, services, (checkpoint, resume)
        )
        assert result.start_times == starts
        assert result.completion_times == ends
        assert result.wait_samples == waits
        assert result.preemptions_per_job == preemptions


@pytest.mark.parametrize("seed", range(6))
def test_random_paths_preserve_work_capacity_and_accounting(seed):
    """Audit every logged interval, gated batch and observation-window integral."""
    rng = np.random.default_rng(seed)
    trace = [
        MsjTraceJob(float(a), int(c), float(s))
        for a, c, s in zip(np.cumsum(rng.exponential(0.4, 160)), rng.integers(0, 3, 160), rng.lognormal(size=160))
    ]
    warmup = 10
    result = replay(trace, warmup=warmup)
    work, events = np.zeros(len(trace)), []
    begin, end = trace[warmup].arrival, trace[-1].arrival
    areas = {}
    for kind, segments in (
        ("service", result.service_segments),
        ("checkpoint", result.checkpoint_segments),
        ("resume", result.resume_segments),
    ):
        areas[kind] = 0
        for idx, start, finish in segments:
            assert trace[idx].arrival <= start < finish <= result.completion_times[idx]
            need = (1, 2, 4)[trace[idx].cls]
            events.extend([(start, need), (finish, -need)])
            areas[kind] += need * max(0, min(finish, end) - max(start, begin))
            if kind == "service":
                work[idx] += finish - start
            else:
                assert finish - start == pytest.approx(0.5 if kind == "checkpoint" else 0.25)
    used = 0
    for _, delta in sorted(events):
        used += delta
        assert 0 <= used <= 4
    np.testing.assert_allclose(work, [job.service for job in trace], atol=1e-12, rtol=1e-12)
    assert used == 0
    np.testing.assert_allclose(
        result.wait_samples, np.array(result.queue_wait_samples) + result.checkpoint_samples + result.resume_samples
    )
    np.testing.assert_allclose(result.wait_samples, np.array(result.first_wait_samples) + result.interruption_samples)
    np.testing.assert_allclose(result.sojourn_samples, work[warmup:] + result.wait_samples, atol=1e-12)
    assert result.productive_utilization == pytest.approx(areas["service"] / (4 * (end - begin)))
    assert result.checkpoint_utilization == pytest.approx(areas["checkpoint"] / (4 * (end - begin)))
    assert result.resume_utilization == pytest.approx(areas["resume"] / (4 * (end - begin)))
    assert result.utilization == pytest.approx(
        result.productive_utilization + result.checkpoint_utilization + result.resume_utilization
    )
    assert len(result.checkpoint_segments) == len(result.resume_segments) == result.preemptions
    for _, start, _ in result.checkpoint_segments:
        assert not any(left < start < right for _, left, right in result.checkpoint_segments + result.resume_segments)


def test_predictions_hidden_future_and_replay_reset():
    """Neither estimates nor unobserved longer service affects earlier choices."""
    trace = [MsjTraceJob(0, 0, 10), MsjTraceJob(1, 2, 20), MsjTraceJob(2, 0, 30)]
    original = replay(trace)
    changed = replay([replace(job, estimate=1e300) for job in trace])
    assert changed.start_times == original.start_times
    assert changed.completion_times == original.completion_times
    longer = replay([replace(job, service=2 * job.service) for job in trace])
    assert [t if t < 10 else None for t in longer.start_times] == [t if t < 10 else None for t in original.start_times]
    sim = MsjCheckpointSim(4, 0.5, 0.25)
    sim.set_servers([1, 2, 4])
    sim.set_deadline_thresholds([1, 2, 3])
    first = sim.run_trace(exact_trace(), 1)
    second = sim.run_trace(exact_trace(), 1)
    assert first.wait_samples == second.wait_samples == [3.5, 2.75, 1.5]
    assert sim.get_empirical_violation_prob(2) == pytest.approx(2 / 3)
    assert first.counts_per_class == [1, 1, 1]
    assert len(first.checkpoint_samples) == 3 and len(first.preemptions_per_job) == 4


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1", None])
@pytest.mark.parametrize("name", ["checkpoint_time", "resume_time"])
def test_invalid_overhead_duration(name, value):
    """Reject invalid durations without clipping or coercing strings/bools."""
    with pytest.raises(ValueError, match=name):
        MsjCheckpointSim(4, **{name: value})


def test_domains_precision_and_predictor_validation():
    """Preserve SF domains and reject costs below floating timestamp precision."""
    with pytest.raises(ValueError, match="power-of-two k"):
        MsjCheckpointSim(3)
    sim = MsjCheckpointSim(4, 0.1, 0.1)
    with pytest.raises(ValueError, match="power-of-two needs"):
        sim.set_servers([3])
    sim.set_servers([1, 4])
    trace = [MsjTraceJob(1e16, 0, 20), MsjTraceJob(1e16 + 2, 1, 4)]
    with pytest.raises(ValueError, match="precision"):
        sim.run_trace(trace)
    with pytest.raises(ValueError, match="remaining_predictor"):
        sim.run_trace([MsjTraceJob(0, 0, 1)], remaining_predictor=lambda cls, age: 1)
    for warmup in (-1, 1, True):
        with pytest.raises(ValueError, match="warmup"):
            sim.run_trace([MsjTraceJob(0, 0, 1)], warmup)


def test_generator_api_and_no_preemption_limit():
    """All-wide jobs pay no overhead, even when configured costs are large."""
    sim = MsjCheckpointSim(4, 100, 100, seed=53)
    sim.set_servers([4], [lambda rng: rng.gamma(2, 0.5)])
    sim.set_sources([0.4])
    result = sim.run(200)
    assert len(result.wait_samples) == 200
    assert result.preemptions == 0
    assert result.productive_utilization == result.utilization
    assert not result.checkpoint_segments and not result.resume_segments
