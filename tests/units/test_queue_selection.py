"""Analytic queue-loss invariants and strict input validation."""

import numpy as np
import pytest

from most_queue.random.queue_selection import queue_log_error, select_queue_model


def test_log_loss_is_symmetric_scale_invariant_and_equal_weighted():
    reference, prediction = np.array([[1, 4], [10, 100]]), np.array([[2, 2], [10, 100]])
    assert queue_log_error(prediction, reference) == pytest.approx(np.log(2) / 2)
    assert queue_log_error(reference, prediction) == pytest.approx(np.log(2) / 2)
    assert queue_log_error(prediction * 7, reference * 7) == pytest.approx(np.log(2) / 2)
    assert queue_log_error(reference, reference) == 0


def test_finite_extremes_do_not_overflow_in_a_ratio():
    assert queue_log_error([1e300], [1e-300]) == pytest.approx(600 * np.log(10))


def test_declared_order_not_mapping_order_breaks_exact_ties():
    result = select_queue_model({"b": [2], "a": [2]}, [1], candidate_order=("a", "b"))
    assert result["selected"] == "a"
    assert list(result["losses"]) == ["a", "b"]


def test_loss_of_mean_is_not_mean_of_losses():
    samples = np.array([[0.5], [1.5]])
    assert queue_log_error(samples.mean(axis=0), [1]) == 0
    assert np.mean([queue_log_error(x, [1]) for x in samples]) > 0


@pytest.mark.parametrize("values", [[], [0], [-1], [np.nan], [np.inf], [True], ["2"], [1j]])
def test_nonpositive_or_nonnumeric_metrics_are_rejected(values):
    with pytest.raises(ValueError):
        queue_log_error(values, values)


@pytest.mark.parametrize("values", [[1, 2], [[1]]])
def test_broadcasting_is_rejected(values):
    with pytest.raises(ValueError, match="shapes"):
        queue_log_error(values, [1])


@pytest.mark.parametrize(
    "order,predictions",
    [
        ([], {}),
        (["a", "a"], {"a": [1]}),
        ([""], {"": [1]}),
        ([1], {1: [1]}),
        ("a", {"a": [1]}),
        (["a"], {"b": [1]}),
        (["a"], {"a": [1], "b": [2]}),
    ],
)
def test_bad_candidate_contracts(order, predictions):
    with pytest.raises(ValueError):
        select_queue_model(predictions, [1], candidate_order=order)
