"""Exact regression/rank identities and information-boundary validation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor


def constant_model():
    """A model independent of every subsequent calibration/test observation."""
    return LogLinearRuntimePredictor().fit(np.empty((2, 0)), [1.0, 1.0])


def test_known_loglinear_law_and_feature_scaling():
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.exp(0.2 + 0.3 * x[:, 0])
    model = LogLinearRuntimePredictor().fit(x, y)
    scaled = LogLinearRuntimePredictor().fit(x * 1000 + 1e6, y)
    test_x = np.array([[2.5], [5.5], [13.0]])
    expected = np.exp(0.2 + 0.3 * test_x[:, 0])
    assert model.predict(test_x) == pytest.approx(expected)
    assert scaled.predict(test_x * 1000 + 1e6) == pytest.approx(expected)


def test_intercept_only_smearing_equals_arithmetic_training_mean():
    model = LogLinearRuntimePredictor().fit(np.empty((3, 0)), [1, 2, 9])
    assert model.predict(np.empty((2, 0))) == pytest.approx([4, 4])
    # Constant/collinear columns are legal, without a singular-matrix failure.
    constant = LogLinearRuntimePredictor().fit(np.ones((3, 2)), [1, 2, 9])
    assert constant.predict(np.ones((1, 2))) == pytest.approx([4])


def test_exact_one_sided_rank_not_absolute_residual_or_interpolation():
    model = constant_model()
    info = model.calibrate(np.empty((4, 0)), [4, 1, 3, 2], coverage=0.6)
    assert (info.rank, info.samples) == (3, 4)
    assert info.log_margin == pytest.approx(np.log(3))
    assert model.predict(np.empty((1, 0)), upper=True) == pytest.approx([3])
    assert model.predict(np.empty((1, 0))) == pytest.approx([1])
    model.calibrate(np.empty((4, 0)), [0.1, 0.2, 0.3, 0.4], coverage=0.6)
    assert model.predict(np.empty((1, 0)), upper=True) == pytest.approx([0.3])


def test_finite_sample_coverage_by_exhaustive_exchangeable_ranks():
    # Treat each of 20 distinct scores in turn as the future observation.
    # With n_cal=19, the 95% bound is its largest calibration observation.
    # Exactly 19 of the 20 equally likely future ranks must be covered.
    values = np.arange(1, 21, dtype=float)
    hits = 0
    for held_out in range(20):
        model = constant_model()
        model.calibrate(np.empty((19, 0)), np.delete(values, held_out), coverage=0.95)
        hits += values[held_out] <= model.predict(np.empty((1, 0)), upper=True)[0]
    assert hits == 19


@pytest.mark.parametrize("value", [0.03, 0.1, 0.3, 3.0, 11.0, 1e-100, 1e100])
def test_constant_runtime_upper_bounds_round_outwards(value):
    model = LogLinearRuntimePredictor().fit(np.empty((2, 0)), [value, value])
    model.calibrate(np.empty((19, 0)), np.full(19, value))
    upper = model.predict(np.empty((1, 0)), upper=True)[0]
    assert upper >= value
    assert upper == pytest.approx(value, rel=1e-12, abs=0)


def test_insufficient_calibration_not_clipped_to_maximum():
    model = constant_model()
    with pytest.raises(ValueError, match="not enough calibration"):
        model.calibrate(np.empty((18, 0)), np.ones(18), coverage=0.95)
    assert model.calibration is None
    assert model.calibrate(np.empty((19, 0)), np.ones(19), coverage=0.95).rank == 19


def test_refit_invalidates_calibration_and_diagnostics_are_immutable():
    model = constant_model()
    info = model.calibrate(np.empty((19, 0)), np.ones(19))
    with pytest.raises(FrozenInstanceError):
        info.rank = 1
    model.fit(np.empty((2, 0)), [2, 2])
    assert model.calibration is None
    with pytest.raises(ValueError, match="calibrate"):
        model.predict(np.empty((1, 0)), upper=True)


def test_training_inputs_copied_and_calibration_does_not_refit_point():
    x = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.exp(0.2 * x[:, 0])
    model = LogLinearRuntimePredictor().fit(x, y)
    reference = model.predict([[2]])
    x[:] = -999
    y[:] = 999
    model.calibrate(np.zeros((19, 1)), np.full(19, 100))
    assert model.predict([[2]]) == pytest.approx(reference)
    before = model.calibration
    with pytest.raises(ValueError):
        model.calibrate(np.zeros((19, 2)), np.ones(19))
    assert model.calibration == before  # failed updates preserve the valid model


@pytest.mark.parametrize("coverage", [None, True, False, 0, 1, -1, 1.1, np.nan, np.inf, "0.95", 0.95 + 0j])
def test_invalid_coverage(coverage):
    with pytest.raises(ValueError):
        constant_model().calibrate(np.empty((20, 0)), np.ones(20), coverage)


@pytest.mark.parametrize(
    "features,durations",
    [
        ([], []),
        ([1, 2], [1, 2]),
        ([[1]], [1]),
        ([[1], [2]], [1]),
        ([[1], [2]], [[1], [2]]),
        ([[1], [2]], [0, 1]),
        ([[1], [2]], [1, np.nan]),
        ([[1], [2]], [1, np.inf]),
        ([[1], [np.nan]], [1, 2]),
        ([[1], [np.inf]], [1, 2]),
        ([[1j], [2]], [1, 2]),
        ([[1], [2]], [1j, 2]),
        ([[1e308], [-1e308]], [1, 2]),
    ],
)
def test_invalid_training_data(features, durations):
    with pytest.raises(ValueError):
        LogLinearRuntimePredictor().fit(features, durations)


def test_prediction_lifecycle_and_overflow_are_explicit():
    model = LogLinearRuntimePredictor()
    with pytest.raises(ValueError, match="fit"):
        model.predict([[1]])
    model.fit([[0], [1]], [1, np.e])
    for features in ([[1, 2]], [[np.nan]], [], [[1j]]):
        with pytest.raises(ValueError):
            model.predict(features)
    with pytest.raises(ValueError, match="boolean"):
        model.predict([[1]], upper="yes")
    model.calibrate(np.zeros((19, 1)), np.ones(19))
    for x in (1000, -1000):
        for upper in (False, True):
            with pytest.raises(ValueError, match="overflows or underflows"):
                model.predict([[x]], upper=upper)


def test_ties_and_calibration_row_order():
    model = constant_model()
    values = np.array([1, 2, 2, 2, 3] * 4)
    first = model.calibrate(np.empty((20, 0)), values, coverage=0.7)
    second = model.calibrate(np.empty((20, 0)), values[::-1], coverage=0.7)
    assert first == second
    assert model.predict(np.empty((1, 0)), upper=True) == pytest.approx([2])
