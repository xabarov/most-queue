"""Analytic adjacency, batches, circular donors and exact matched inventories."""

from collections import Counter

import numpy as np
import pytest

from most_queue.random.marked_arrivals import (
    ArrivalMark,
    MarkedArrivalBootstrap,
    anchored_permutation,
    arrival_times,
    decouple_gaps,
)


def test_fit_retains_true_gaps_ties_and_current_row_marks():
    model = MarkedArrivalBootstrap.fit([100, 100, 107, 110], [1, 2, 3, 4], ["a", "b", "c", "d"], [1, 2, 3, 4])
    assert model.donors == (ArrivalMark(0, 2, "b", 2), ArrivalMark(7, 3, "c", 3), ArrivalMark(3, 4, "d", 4))
    assert model.sample([0.1, 0.5, 0.9]) == model.donors
    np.testing.assert_array_equal(arrival_times(model.donors), [0, 7, 10])


def test_circular_blocks_wrap_and_truncate_with_known_start_indices():
    model = MarkedArrivalBootstrap(tuple(ArrivalMark(i, i + 1) for i in range(5)))
    result = model.sample([0.9, 0.1, 0.1, 0.3, 0.1], block_length=3)
    assert [r.need for r in result] == [5, 1, 2, 2, 3]


def test_anchors_preserve_horizon_and_both_phase_inventories():
    records = tuple(ArrivalMark(i, i + 1, str(i), i + 1) for i in range(8))
    order = anchored_permutation([0.2, 0.8, 0.1, 0.3, 0.4, 0.9, 0.2, 0.1], 3)
    assert order.tolist() == [0, 2, 1, 3, 7, 6, 4, 5]
    shuffled = tuple(records[i] for i in order)
    for begin, end in ((0, 3), (3, 8)):
        assert Counter(records[begin:end]) == Counter(shuffled[begin:end])
    a, b = arrival_times(records), arrival_times(shuffled)
    assert a[3] == b[3] and a[-1] == b[-1]
    independent = decouple_gaps(records, order)
    assert [r.gap for r in independent] == [r.gap for r in records]
    assert [(r.need, r.context, r.requested_time) for r in independent] == [
        (records[i].need, records[i].context, records[i].requested_time) for i in order
    ]


def test_zero_gaps_are_a_batch_not_jittered():
    records = (ArrivalMark(0, 1),) * 4
    np.testing.assert_array_equal(arrival_times(records), np.zeros(4))


def test_key_ties_are_stable_and_zero_warmup_is_supported():
    np.testing.assert_array_equal(anchored_permutation([0.5] * 4, 0), range(4))
    np.testing.assert_array_equal(anchored_permutation([0.9, 0.3, 0.2, 0.1], 0), [0, 3, 2, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gap": -1},
        {"gap": np.inf},
        {"gap": True},
        {"gap": "2"},
        {"need": 0},
        {"need": 1.5},
        {"need": True},
        {"context": ""},
        {"context": 2},
        {"requested_time": 0},
        {"requested_time": -1},
        {"requested_time": np.nan},
    ],
)
def test_invalid_marks(kwargs):
    params = {"gap": 0, "need": 1}
    params.update(kwargs)
    with pytest.raises(ValueError):
        ArrivalMark(**params)


@pytest.mark.parametrize(
    "times,needs,contexts",
    [
        ([1], [1], [None]),
        ([1, 0], [1, 1], [None, None]),
        ([0, np.nan], [1, 1], [None, None]),
        ([0, 1], [1], [None, None]),
        ([0, 1], [0, 1], [None, None]),
    ],
)
def test_invalid_fit_inputs(times, needs, contexts):
    with pytest.raises(ValueError):
        MarkedArrivalBootstrap.fit(times, needs, contexts)


@pytest.mark.parametrize(
    "uniforms,warmup",
    [
        ([], 0),
        ([0, 0.5], 0),
        ([0.5, 1], 0),
        ([np.nan, 0.5], 0),
        ([0.5, 0.5], 2),
        ([0.5, 0.5], True),
        ([[0.5]], 0),
        ([".5"], 0),
    ],
)
def test_bad_permutation_inputs(uniforms, warmup):
    with pytest.raises(ValueError):
        anchored_permutation(uniforms, warmup)


@pytest.mark.parametrize("order", [[0, 0], [1], [0.0, 1.0], [-1, 0], [False, True]])
def test_not_a_permutation_rejected(order):
    with pytest.raises(ValueError):
        decouple_gaps((ArrivalMark(1, 1),) * 2, order)


def test_bad_donors_overflow_and_blocks_fail_without_fallback():
    with pytest.raises(ValueError):
        MarkedArrivalBootstrap(())
    with pytest.raises(ValueError):
        MarkedArrivalBootstrap((object(),))
    with pytest.raises(ValueError):
        arrival_times(())
    with pytest.raises(ValueError, match="overflow"):
        arrival_times((ArrivalMark(1e308, 1),) * 3)
    with pytest.raises(ValueError, match="block_length"):
        MarkedArrivalBootstrap((ArrivalMark(0, 1),)).sample([0.5], block_length=2)
