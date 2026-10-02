"""Group rank validity, refusal boundaries and pooled API compatibility."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor


def constant_model():
    """Fit without using any later calibration/test observation."""
    return LogLinearRuntimePredictor().fit(np.empty((2, 0)), [1, 1])


def test_exact_group_ranks_signed_scores_and_row_alignment():
    model = constant_model()
    values = [4, 0.4, 1, 0.1, 3, 0.3, 2, 0.2]
    groups = [1, 4] * 4
    info = model.calibrate_by_group(np.empty((8, 0)), values, groups, coverage=0.6)
    assert set(info) == {1, 4}
    assert [(v.samples, v.rank, v.is_finite) for v in info.values()] == [(4, 3, True)] * 2
    assert info[1].log_margin == pytest.approx(np.log(3))
    assert info[4].log_margin == pytest.approx(np.log(0.3))
    actual = model.predict(np.empty((4, 0)), upper=True, groups=[4, 1, 4, 1])
    assert actual == pytest.approx([0.3, 3, 0.3, 3])


@pytest.mark.parametrize("group", [0, 4])
def test_exhaustive_exchangeable_ranks_within_each_group(group):
    values = np.arange(1, 21, dtype=float) * (group + 1)
    other = 4 - group
    hits = 0
    for held_out in range(20):
        services = np.concatenate((np.delete(values, held_out), np.full(19, 1000)))
        model = constant_model()
        model.calibrate_by_group(np.empty((38, 0)), services, [group] * 19 + [other] * 19)
        upper = model.predict(np.empty((1, 0)), upper=True, groups=[group])[0]
        hits += values[held_out] <= upper
    assert hits == 19  # rank coverage in this group, not only across pooled jobs


def test_single_group_reduces_exactly_to_pooled_calibration():
    rng = np.random.default_rng(50)
    x = rng.normal(size=(50, 2))
    services = rng.lognormal(size=50)
    model = LogLinearRuntimePredictor().fit(x[:20], services[:20])
    pooled = model.calibrate(x[20:], services[20:])
    grouped = model.calibrate_by_group(x[20:], services[20:], np.full(30, 4))[4]
    assert (pooled.samples, pooled.rank, pooled.log_margin) == (grouped.samples, grouped.rank, grouped.log_margin)
    np.testing.assert_array_equal(model.predict(x, upper=True), model.predict(x, upper=True, groups=np.full(50, 4)))


def test_other_group_scores_do_not_change_target_group_and_permutation_is_invariant():
    model = constant_model()
    labels = np.tile([1, 3, 4], 20)
    services = np.linspace(0.1, 6, 60)
    first = model.calibrate_by_group(np.empty((60, 0)), services, labels)
    second = model.calibrate_by_group(np.empty((60, 0)), services[::-1], labels[::-1])
    assert first == second
    changed = services.copy()
    changed[labels != 4] *= 100
    third = model.calibrate_by_group(np.empty((60, 0)), changed, labels)
    assert first[4] == third[4]
    assert first[1] != third[1]


def test_rare_unseen_and_mixed_batches_refuse_without_pooled_fallback():
    model = constant_model()
    x = np.empty((37, 0))
    model.calibrate(x, np.ones(37))  # a pooled bound exists but cannot be borrowed
    info = model.calibrate_by_group(x, np.ones(37), [1] * 19 + [4] * 18)
    assert info[1].is_finite and info[1].rank == 19
    assert not info[4].is_finite and info[4].log_margin is None
    assert (info[4].rank, info[4].samples) == (19, 18)
    assert model.predict(np.empty((1, 0)), upper=True, groups=[1])[0] >= 1
    for groups in ([4], [1, 4]):
        with pytest.raises(ValueError, match="not enough calibration"):
            model.predict(np.empty((len(groups), 0)), upper=True, groups=groups)
    for groups in ([3], [1, 3]):
        with pytest.raises(ValueError, match="unseen calibration groups"):
            model.predict(np.empty((len(groups), 0)), upper=True, groups=groups)
    assert model.predict(np.empty((1, 0)), upper=True)[0] >= 1  # explicit pooled mode still works


def test_all_groups_can_be_insufficient_without_claiming_a_finite_forecast():
    model = constant_model()
    info = model.calibrate_by_group(np.empty((2, 0)), [1, 2], [0, 1])
    assert all(not record.is_finite for record in info.values())
    with pytest.raises(ValueError, match="not enough calibration"):
        model.predict(np.empty((1, 0)), upper=True, groups=[0])


def test_group_diagnostics_are_frozen_snapshots_and_refit_clears_both_modes():
    model = constant_model()
    model.calibrate(np.empty((19, 0)), np.ones(19))
    info = model.calibrate_by_group(np.empty((19, 0)), np.ones(19), [4] * 19)
    with pytest.raises(TypeError):
        info[4] = None
    with pytest.raises(FrozenInstanceError):
        info[4].rank = 1
    model.fit(np.empty((2, 0)), [2, 2])
    assert model.calibration is None and not model.group_calibrations
    assert info[4].samples == 19  # snapshots do not mutate when the model changes
    with pytest.raises(ValueError, match="calibrate_by_group"):
        model.predict(np.empty((1, 0)), upper=True, groups=[4])


def test_calibration_modes_preserve_each_other_and_failed_updates_are_atomic():
    model = constant_model()
    x = np.empty((19, 0))
    info = model.calibrate_by_group(x, np.ones(19), [4] * 19)
    model.calibrate(x, np.full(19, 2))
    assert model.group_calibrations == info
    pooled = model.calibration
    model.calibrate_by_group(x, np.full(19, 3), [4] * 19)
    assert model.calibration == pooled
    grouped = model.group_calibrations
    with pytest.raises(ValueError):
        model.calibrate_by_group(x, [0] * 19, [4] * 19)
    with pytest.raises(ValueError):
        model.fit([[1e308], [-1e308]], [1, 2])
    assert model.calibration == pooled and model.group_calibrations == grouped
    model.calibrate_by_group(x, np.ones(19), [1] * 19)
    assert set(model.group_calibrations) == {1}  # replacement, not stale merging
    assert set(grouped) == {4}


@pytest.mark.parametrize(
    "labels",
    [
        [-1, 1],
        [True, 1],
        [1, np.bool_(False)],
        [1.0, 2],
        ["1", "2"],
        [np.nan, 1],
        [np.inf, 1],
        [[1], [2]],
        [1],
        [],
        [1j, 2],
        None,
    ],
)
def test_invalid_group_labels(labels):
    model = constant_model()
    with pytest.raises(ValueError, match="groups"):
        model.calibrate_by_group(np.empty((2, 0)), [1, 2], labels)
    # Once calibrated, prediction performs the same shape/type checks.
    model.calibrate_by_group(np.empty((19, 0)), np.ones(19), [1] * 19)
    if labels is not None:  # None explicitly selects pooled mode, tested elsewhere
        with pytest.raises(ValueError, match="groups"):
            model.predict(np.empty((2, 0)), upper=True, groups=labels)


@pytest.mark.parametrize("coverage", [True, 0, 1, np.nan, np.inf, "0.95"])
def test_invalid_group_coverage(coverage):
    with pytest.raises(ValueError, match="coverage"):
        constant_model().calibrate_by_group(np.empty((19, 0)), np.ones(19), [1] * 19, coverage)


def test_group_lifecycle_and_no_implicit_mode_selection():
    model = LogLinearRuntimePredictor()
    with pytest.raises(ValueError, match="fit"):
        model.calibrate_by_group(np.empty((19, 0)), np.ones(19), [1] * 19)
    model.fit(np.empty((2, 0)), [1, 1])
    model.calibrate_by_group(np.empty((19, 0)), np.ones(19), [1] * 19)
    with pytest.raises(ValueError, match="pooled"):
        model.predict(np.empty((1, 0)), upper=True)
    with pytest.raises(ValueError, match="upper=True"):
        model.predict(np.empty((1, 0)), groups=[1])


@pytest.mark.parametrize("value", [0.03, 0.3, 11.0, 1e-100, 1e100])
def test_grouped_ties_round_outwards(value):
    model = constant_model()
    model.calibrate_by_group(np.empty((19, 0)), np.full(19, value), [4] * 19)
    upper = model.predict(np.empty((1, 0)), upper=True, groups=[4])[0]
    assert upper >= value
    assert upper == pytest.approx(value, rel=1e-12, abs=0)


def test_grouped_overflow_and_underflow_are_errors():
    model = LogLinearRuntimePredictor().fit([[0], [1]], [1, np.e])
    model.calibrate_by_group(np.zeros((19, 1)), np.ones(19), [4] * 19)
    for value in (-1000, 1000):
        with pytest.raises(ValueError, match="overflows or underflows"):
            model.predict([[value]], upper=True, groups=[4])
