"""Independent hand timelines for EngineCore polling and ahead scheduling."""

import numpy as np
import pytest

from most_queue.sim.advance_reservation import AdvanceReservationSim
from most_queue.sim.iteration_admission import IterationAdmissionSim
from most_queue.sim.prefill_decode import PrefillDecodeSim


def configured(cap=1, budget=None, **kwargs):
    sim = IterationAdmissionSim(cap, budget, **kwargs)
    sim.set_servers(1, 1, 1, 0, 0)
    return sim


@pytest.mark.parametrize("limit", [1, 2])
def test_isolated_exact_costs_and_no_extra_output(limit):
    sim = configured(inflight_batches=limit)
    sim.set_servers(4, 3, 0.1, 0.02, 0.003)
    result = sim.replay([5], [4], [3], record_iterations=True)
    durations = [0.192, 0.135, 0.138]
    assert result.output_times[0] == pytest.approx(5 + np.cumsum(durations))
    assert result.first_compute_times == [5]
    assert result.start_times == [5]
    assert len(result.iteration_trace) == 3
    assert result.first_decode_mean == 0
    assert result.itl_mean == pytest.approx(np.mean(durations[1:]))
    assert result.all_decode_mean == pytest.approx(sum(durations[1:]) / 3)


def test_two_ahead_plan_is_immutable_and_compute_timestamp_is_actual_start():
    result = configured(2).replay([0, 0.5], [1, 1], [4, 1], record_iterations=True)
    assert result.start_times == [0, 1]
    assert result.first_compute_times == [0, 2]
    assert result.output_times == [[1, 2, 3, 4], [3]]
    trace = result.iteration_trace
    assert [row["scheduled"] for row in trace] == [0, 0, 1, 2]
    assert [row["start"] for row in trace] == [0, 1, 2, 3]
    assert trace[1]["actions"] == [(0, "decode", 1)]
    assert set(trace[2]["actions"]) == {(0, "decode", 1), (1, "prefill", 1)}
    assert [row["inflight_after_submit"] for row in trace] == [1, 2, 2, 2]


def test_polling_partitions_input_wait_without_changing_absolute_execution():
    polled = configured(2, poll_inputs=True).replay([0, 0.5], [1, 1], [4, 1])
    direct = configured(2, poll_inputs=False).replay([0, 0.5], [1, 1], [4, 1])
    assert polled.engine_entry_times == [0, 1]
    assert direct.engine_entry_times == [0, 0.5]
    assert polled.input_wait_times == [0, 0.5]
    assert direct.input_wait_times == [0, 0]
    assert polled.input_wait_mean == 0.25
    for field in ("start_times", "first_compute_times", "completion_times", "output_times", "w", "v"):
        assert getattr(polled, field) == getattr(direct, field)
    assert polled.w[0] == 0.25


@pytest.mark.parametrize("limit", [1, 2])
def test_single_slot_matches_independent_lindley_recursion(limit):
    arrivals = np.array([0, 0.05, 0.07, 1, 1, 1.1, 4])
    inputs = [2, 4, 1, 3, 1, 5, 2]
    outputs = [1, 3, 2, 1, 4, 2, 1]
    sim = configured(inflight_batches=limit)
    sim.set_servers(1, 1, 0.1, 0.02, 0.003)
    result = sim.replay(arrivals, inputs, outputs)
    finish = 0
    starts, finishes = [], []
    for arrival, n, m in zip(arrivals, inputs, outputs):
        start = max(arrival, finish)
        service = 0.1 + 0.023 * n
        service += sum(0.12 + 0.003 * (n + k) for k in range(1, m))
        finish = start + service
        starts.append(start)
        finishes.append(finish)
    assert result.start_times == pytest.approx(starts)
    assert result.completion_times == pytest.approx(finishes)
    assert result.w[0] == pytest.approx(np.mean(np.array(starts) - arrivals))
    assert result.v[0] == pytest.approx(np.mean(np.array(finishes) - arrivals))


@pytest.mark.parametrize("limit", [1, 2])
def test_ties_fcfs_and_terminal_slots_are_not_freed_at_planning(limit):
    result = configured(2, 4, inflight_batches=limit).replay([0] * 5, [2] * 5, [1] * 5, record_iterations=True)
    assert result.start_times == [0, 0, 1, 1, 2]
    assert result.completion_times == [1, 1, 2, 2, 3]
    assert [row["active"] for row in result.iteration_trace] == [[0, 1], [2, 3], [4]]
    assert all(len(row) == 1 for row in result.output_times)
    assert all(len(row["active"]) <= 2 for row in result.iteration_trace)


def test_warmup_occupancy_uses_input_ready_population_and_excludes_drain():
    result = configured().replay([0, 0.5, 1.5], [1] * 3, [1] * 3, warmup_fraction=1 / 3)
    # On [.5, 1): two jobs; on [1, 1.5): one job. One active throughout.
    assert result.observation_time == 1
    assert result.p == pytest.approx([0, 0.5, 0.5])
    assert result.jobs_mean == 1.5
    assert result.batch_mean == result.utilization == result.busy_probability == 1
    assert result.queue_mean == 0.5
    assert result.w == pytest.approx([0.5, 0.25, 0.125])
    assert result.v == pytest.approx([1.5, 2.25, 3.375])


def test_existing_models_keep_distinct_arrival_and_output_semantics():
    results = []
    for cls in (PrefillDecodeSim, AdvanceReservationSim, IterationAdmissionSim):
        sim = cls(2, None)
        sim.set_servers(1, 1, 1, 0, 0)
        results.append(sim.replay([0, 0.5], [1, 1], [1, 1]))
    assert results[0].output_times == [[2], [3]]
    assert results[1].start_times == [0, 0.5]
    assert results[1].output_times == [[1], [2]]
    assert results[2].start_times == [0, 1]
    assert results[2].output_times == [[1], [2]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inflight_batches": 0},
        {"inflight_batches": 3},
        {"inflight_batches": True},
        {"inflight_batches": 1.0},
        {"poll_inputs": 1},
        {"poll_inputs": None},
        {"batch_cap": True},
        {"batch_cap": 0},
        {"token_budget": True},
        {"max_iterations": 0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        IterationAdmissionSim(**kwargs)


@pytest.mark.parametrize(
    "arrivals,inputs,outputs",
    [
        ([], [], []),
        ([1, 0], [1, 1], [1, 1]),
        ([-1], [1], [1]),
        ([np.inf], [1], [1]),
        ([0], [0], [1]),
        ([0], [1.5], [1]),
        ([0], [True], [1]),
        ([0], [1], [True]),
        ([True], [1], [1]),
        ([0], [1], [0]),
        ([0], [1_000_001], [1]),
        ([0, 1], [1], [1]),
    ],
)
def test_replay_validation(arrivals, inputs, outputs):
    with pytest.raises(ValueError):
        configured().replay(arrivals, inputs, outputs)


def test_envelope_iteration_budget_and_finite_clock_guards():
    sim = configured(2, 4)
    with pytest.raises(ValueError, match="unchunked envelope"):
        sim.replay([0], [3], [1])
    with pytest.raises(RuntimeError, match="max_iterations"):
        configured(max_iterations=1).replay([0], [1], [2])
    assert len(configured(max_iterations=1).replay([0], [1], [1]).output_times[0]) == 1
    with pytest.raises(ValueError):
        configured().replay([1e300], [1], [1])
    sim = configured()
    sim.set_servers(1, 1, 1e308, 1e308, 0)
    with pytest.raises(ValueError):
        sim.replay([0], [1], [1])


def test_seeded_sampler_run_is_repeatable_and_trace_is_opt_in():
    sim = configured(2, 10, seed=17)
    sim.set_sources(2)
    sim.set_servers(
        1, 1, 0.1, 0.02, 0.003, length_sampler=lambda rng: (int(rng.integers(1, 6)), int(rng.integers(1, 4)))
    )
    first, second = sim.run(40), sim.run(40)
    assert first.output_times == second.output_times
    assert first.engine_entry_times == second.engine_entry_times
    assert first.v == second.v
    assert first.iteration_trace == []
    assert sim.run(2, record_iterations=True).iteration_trace
