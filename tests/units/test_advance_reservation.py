"""Independent trace and queueing checks of advance reservation semantics."""

import numpy as np
import pytest

from most_queue.sim.advance_reservation import AdvanceReservationSim
from most_queue.sim.prefill_decode import PrefillDecodeSim


def configured(cap=1, budget=None, **kwargs):
    sim = AdvanceReservationSim(cap, budget, **kwargs)
    sim.set_servers(1, 1, 0.1, 0.02, 0.003)
    return sim


@pytest.mark.parametrize("n,m", [(1, 1), (4, 1), (4, 3)])
def test_isolated_hand_calculation(n, m):
    result = configured().replay([0], [n], [m], record_iterations=True)
    prefill = 0.1 + 0.023 * n
    decode = [0.12 + 0.003 * (n + k) for k in range(1, m)]
    assert result.prefill_end_times == pytest.approx([prefill])
    assert result.output_times[0] == pytest.approx(np.cumsum([prefill] + decode))
    assert result.completion_times == pytest.approx([prefill + sum(decode)])
    assert result.first_compute_times == [0]
    assert result.ttft_mean == pytest.approx(prefill)
    assert result.first_decode_mean == 0
    assert result.itl_mean == (pytest.approx(np.mean(decode)) if decode else None)
    assert result.all_decode_mean == pytest.approx(sum(decode) / m)
    assert result.v[0] == pytest.approx(result.service_mean)


def test_arrivals_reserve_slots_but_running_actions_stay_frozen():
    result = configured(2).replay([0, 0.05, 0.06], [3, 2, 1], [1, 1, 1], record_iterations=True)
    assert result.start_times == pytest.approx([0, 0.05, 0.169])
    assert result.first_compute_times == pytest.approx([0, 0.169, 0.169])
    assert result.iteration_trace[0]["actions"] == [(0, "prefill", 3)]
    assert result.iteration_trace[1]["actions"] == [(1, "prefill", 2), (2, "prefill", 1)]
    assert result.w[0] == pytest.approx((0.169 - 0.06) / 3)
    assert result.prefill_mean == pytest.approx(np.mean(np.array(result.prefill_end_times) - result.start_times))
    assert result.service_mean == pytest.approx(np.mean(np.array(result.completion_times) - result.start_times))
    assert result.queue_mean == pytest.approx(result.jobs_mean - result.batch_mean)
    assert sum(result.p) == pytest.approx(1)


def test_fcfs_cap_and_same_time_ties():
    result = configured(2, 4).replay([0] * 5, [2] * 5, [1] * 5, record_iterations=True)
    assert [r["active"] for r in result.iteration_trace] == [[0, 1], [2, 3], [4]]
    assert result.start_times == pytest.approx([0, 0, 0.192, 0.192, 0.384])
    assert result.first_compute_times == result.start_times
    assert all(len(t) == 1 for t in result.output_times)


def test_single_slot_matches_independent_lindley_recursion():
    arrivals = np.array([0, 0.05, 0.07, 1, 1, 1.1, 4])
    inputs = [2, 4, 1, 3, 1, 5, 2]
    outputs = [1, 3, 2, 1, 4, 2, 1]
    result = configured().replay(arrivals, inputs, outputs)
    finish = 0
    starts, finishes = [], []
    for arrival, n, m in zip(arrivals, inputs, outputs):
        start = max(arrival, finish)
        service = 0.1 + 0.02 * n + 0.003 * n
        service += sum(0.1 + 0.02 + 0.003 * (n + k) for k in range(1, m))
        finish = start + service
        starts.append(start)
        finishes.append(finish)
    assert result.start_times == pytest.approx(starts)
    assert result.completion_times == pytest.approx(finishes)
    assert result.w[0] == pytest.approx(np.mean(np.array(starts) - arrivals))
    assert result.v[0] == pytest.approx(np.mean(np.array(finishes) - arrivals))


def test_budget_rejection_and_invalid_inputs():
    sim = configured(2, 4)
    with pytest.raises(ValueError, match="unchunked envelope"):
        sim.set_servers(3, 1, 0.1, 0, 0)
    with pytest.raises(ValueError, match="unchunked envelope"):
        sim.replay([0], [3], [1])
    for inputs, outputs in [([0], [1]), ([1.5], [1]), ([1], [0])]:
        with pytest.raises(ValueError):
            sim.replay([0], inputs, outputs)
    assert configured(2, 6).replay([0], [3], [1]).first_decode_mean == 0


def test_iteration_limit_and_seeded_run():
    with pytest.raises(RuntimeError, match="max_iterations"):
        configured(max_iterations=1).replay([0], [1], [2])
    sim = configured(2, 10, seed=17)
    sim.set_sources(2)
    sim.set_servers(1, 1, 0.1, 0.02, 0.003, length_sampler=lambda rng: (int(rng.integers(1, 6)), 2))
    first, second = sim.run(40), sim.run(40)
    assert first.output_times == second.output_times
    assert first.first_compute_times == second.first_compute_times
    assert first.v == second.v


def test_original_first_compute_records_budget_stall_without_changing_admission():
    sim = PrefillDecodeSim(2, 2)
    sim.set_servers(2, 1, 0.1, 0, 0)
    result = sim.replay([0, 0], [2, 2], [1, 1])
    assert result.start_times == [0, 0]
    assert result.first_compute_times == pytest.approx([0, 0.1])
