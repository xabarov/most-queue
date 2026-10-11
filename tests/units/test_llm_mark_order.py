"""Exact permutation invariants and independent FCFS/token trace oracles."""

from collections import Counter

import numpy as np
import pytest

from examples.llm_mark_order_experiment import controlled_marks
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.sim.utils.llm_trace import permute_joint_marks, read_burstgpt


@pytest.mark.parametrize("mode", ["observed", "global", "local", "blocks", "shift"])
def test_pair_law_preserved(mode):
    times = np.arange(103) * 5.0
    marks = np.column_stack((np.arange(103) + 1, np.arange(103)[::-1] + 2))
    result = permute_joint_marks(times, marks, mode, 8, block_size=7)
    assert Counter(map(tuple, result)) == Counter(map(tuple, marks))
    assert result is not marks
    if mode == "local":
        for label in np.unique(times // 300):
            mask = times // 300 == label
            assert Counter(map(tuple, result[mask])) == Counter(map(tuple, marks[mask]))
    if mode == "shift":
        before = Counter(zip(map(tuple, marks), map(tuple, np.roll(marks, -1, axis=0))))
        after = Counter(zip(map(tuple, result), map(tuple, np.roll(result, -1, axis=0))))
        assert before == after


def test_audit_and_malformed(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(
        "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
        "0,ChatGPT,2,1,3,API log\n1,ChatGPT,2,0,2,API log\n"
        "2,ChatGPT,0,1,1,API log\n3,ChatGPT,2,1,3,Conversation log\n"
        "4,GPT-4,2,1,3,API log\n"
    )
    trace = read_burstgpt(path)
    assert trace.audit == {
        "records": 5,
        "eligible": 1,
        "zero_output": 1,
        "zero_input": 1,
        "not_api": 1,
        "other_model": 1,
    }
    assert trace.marks.tolist() == [[2, 1]]
    path.write_text(path.read_text().replace("2,1,3", "2,1,4"))
    with pytest.raises(ValueError, match="total"):
        read_burstgpt(path)


@pytest.mark.parametrize("mode", ["observed", "global", "local", "blocks", "shift"])
def test_warmup_and_measured_law_separately_preserved(mode):
    times = np.arange(103) * 5.0
    marks = np.column_stack((np.arange(103) + 1, np.arange(103)[::-1] + 2))
    result = controlled_marks(times, marks, mode, 92, warmup=17)
    assert np.array_equal(result[:17], marks[:17])
    assert Counter(map(tuple, result[17:])) == Counter(map(tuple, marks[17:]))


def test_b1_against_lindley_with_heterogeneous_service():
    times = np.array([0.0, 0.1, 0.7, 1.2, 8.0])
    pairs = np.array([[3, 2], [1, 3], [5, 1], [2, 2], [1, 1]])
    sim = PrefillDecodeSim(1, 2)
    sim.set_servers(1, 1, 0.1, 0.02, 0.003)
    result = sim.replay(times, *pairs.T)
    starts, ends, firsts = [], [], []
    finish = 0.0
    for arrival, (n, m) in zip(times, pairs):
        begin = max(arrival, finish)
        # Independent scalar FCFS work enumeration, no DES/analytical helper.
        chunks = list(range(2, n, 2)) + [n]
        prefill = sum(0.1 + 0.02 * (q - (chunks[i - 1] if i else 0)) + 0.003 * q for i, q in enumerate(chunks))
        decode = [0.1 + 0.02 + 0.003 * (n + k) for k in range(1, m + 1)]
        finish = begin + prefill + sum(decode)
        starts.append(begin)
        ends.append(finish)
        firsts.append(begin + prefill + decode[0])
    assert result.start_times == pytest.approx(starts)
    assert result.completion_times == pytest.approx(ends)
    assert [row[0] for row in result.output_times] == pytest.approx(firsts)


def test_blind_boundary_admission_manual_trace():
    sim = PrefillDecodeSim(2, 2)
    sim.set_servers(1, 1, 1.0, 0.0, 0.0)
    result = sim.replay([0, 0.5, 0.6], [1, 1, 1], [3, 1, 1])
    # First job prefill0..1; second joins1; third must wait for second to exit3.
    assert result.start_times == [0, 1, 3]
    assert result.completion_times == [4, 3, 5]
    assert [row[0] for row in result.output_times] == [2, 3, 5]


@pytest.mark.parametrize("mode", ["invalid", "local"])
def test_invalid_permutations(mode):
    with pytest.raises(ValueError):
        permute_joint_marks([0, 1], [[1, 2], [2, 3]], mode, 1, window=0)
