"""Exact censored-risk tables and an independent SciPy Kaplan-Meier reference."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import stats

from most_queue.sim.utils.residual_runtime import KaplanMeierPoint, KaplanMeierRuntimeEstimator


def test_tied_events_precede_censoring_and_curve_is_right_continuous():
    """Check a hand-derived risk table, conditional quantiles and integrals."""
    model = KaplanMeierRuntimeEstimator().fit([1, 2, 2, 4], [True, True, False, True])
    assert model.curve == (
        KaplanMeierPoint(1, 4, 1, 0, 0.75),
        KaplanMeierPoint(2, 3, 1, 1, pytest.approx(0.5)),
        KaplanMeierPoint(4, 1, 1, 0, 0),
    )
    assert [model.survival(t) for t in (0, 1, 1.99, 2, 4, 100)] == pytest.approx([1, 0.75, 0.75, 0.5, 0, 0])
    assert model.remaining_quantile(1, 0.3) == 1
    assert model.remaining_quantile(2, 0.5) == 2
    assert model.remaining_mean(0) == pytest.approx(2.75)
    assert model.remaining_mean(1) == pytest.approx(7 / 3)
    assert model.remaining_mean(1, horizon=3) == pytest.approx(5 / 3)
    assert model.remaining_mean(1, horizon=100) == pytest.approx(7 / 3)
    assert model.remaining_quantile(4) is None
    assert model.remaining_mean(4) is None


def test_unidentified_tail_is_not_imputed_as_a_completion_or_extrapolated():
    """Positive terminal survival leaves full means and high quantiles unknown."""
    model = KaplanMeierRuntimeEstimator().fit([1, 2, 4], [True, True, False])
    assert model.remaining_quantile(0, 0.5) == 2
    assert model.remaining_quantile(0, 0.9) is None
    assert model.remaining_quantile(2, 0.1) is None
    assert model.remaining_mean(0) is None
    assert model.remaining_mean(0, horizon=4) == pytest.approx(7 / 3)
    assert model.remaining_mean(0, horizon=4.01) is None
    assert model.remaining_mean(5, horizon=6) is None
    assert model.survival(100) == pytest.approx(1 / 3)  # estimator convention only


def test_single_and_all_censored_histories():
    """Small samples have explicit, non-extrapolated semantics."""
    censored = KaplanMeierRuntimeEstimator().fit([2, 3], [False, False])
    assert censored.remaining_mean(0, horizon=3) == 3
    assert censored.remaining_mean(0) is None
    assert censored.remaining_quantile(0, 0.01) is None
    complete = KaplanMeierRuntimeEstimator().fit([3], [True])
    assert complete.remaining_quantile(2) == complete.remaining_mean(2) == 1
    assert complete.remaining_mean(2, horizon=2.5) == 0.5
    assert complete.remaining_quantile(0, 1e-20) == 3


@pytest.mark.parametrize("seed", range(6))
def test_random_tied_censored_curve_matches_scipy(seed):
    """Use SciPy as an independent product-limit implementation."""
    rng = np.random.default_rng(seed)
    times = rng.integers(1, 20, 100)
    flags = rng.random(100) < 0.65
    model = KaplanMeierRuntimeEstimator().fit(times, flags)
    reference = stats.ecdf(stats.CensoredData(uncensored=times[flags], right=times[~flags])).sf
    ages = np.linspace(0, 25, 201)
    assert np.allclose([model.survival(a) for a in ages], reference.evaluate(ages), rtol=1e-13, atol=1e-15)


def test_prediction_depends_on_observations_not_hidden_censored_lifetimes():
    """Changing hidden lifetimes behind the same censoring cannot change fit."""
    exposure = np.array([2, 3, 5, 6])
    lifetime = np.array([1, 4, 8, 2])
    hidden_changed = np.array([1, 100, 1000, 2])
    models = [
        KaplanMeierRuntimeEstimator().fit(np.minimum(s, exposure), s <= exposure) for s in (lifetime, hidden_changed)
    ]
    assert models[0].curve == models[1].curve
    assert models[0].remaining_quantile(1, 0.4) == models[1].remaining_quantile(1, 0.4)


def test_owned_immutable_state_and_atomic_refit():
    """Inputs are owned and diagnostic records cannot corrupt prediction state."""
    times, flags = np.array([1.0, 2.0]), np.array([True, True])
    model = KaplanMeierRuntimeEstimator().fit(times, flags)
    original = model.curve
    times[:] = 100
    flags[:] = False
    assert model.remaining_mean(0) == 1.5
    with pytest.raises(FrozenInstanceError):
        model.curve[0].survival = 1
    with pytest.raises(ValueError):
        model.fit([1], [1])
    assert model.curve == original
    model.fit([10], [True])
    assert model.remaining_mean(0) == 10


@pytest.mark.parametrize(
    "times,flags",
    [
        ([], []),
        ([0], [True]),
        ([-1], [True]),
        ([np.inf], [True]),
        ([np.nan], [True]),
        ([[1]], [[True]]),
        ([1], [1]),
        ([1], ["yes"]),
        ([1], []),
        ([True], [True]),
        ([1j], [True]),
        (["1"], [True]),
    ],
)
def test_invalid_history(times, flags):
    """Reject malformed durations or completion indicators."""
    with pytest.raises(ValueError):
        KaplanMeierRuntimeEstimator().fit(times, flags)


@pytest.mark.parametrize("age", [-1, np.inf, np.nan, True, "1", 1j])
def test_invalid_age(age):
    """Age is a finite, nonnegative real scalar."""
    model = KaplanMeierRuntimeEstimator().fit([1], [True])
    for method in (model.survival, model.remaining_quantile, model.remaining_mean):
        with pytest.raises(ValueError):
            method(age)


@pytest.mark.parametrize("probability", [0, 1, -0.1, np.nan, np.inf, True, "0.9"])
def test_invalid_quantile(probability):
    """Exclude probabilities outside the open unit interval."""
    with pytest.raises(ValueError):
        KaplanMeierRuntimeEstimator().fit([1], [True]).remaining_quantile(0, probability)


@pytest.mark.parametrize("horizon", [0, -1, np.nan, np.inf, True, "2"])
def test_invalid_horizon(horizon):
    """Restricted means require a finite lifetime horizon after age."""
    with pytest.raises(ValueError):
        KaplanMeierRuntimeEstimator().fit([1], [True]).remaining_mean(0, horizon)


def test_unfitted_model():
    """No prediction is available before observed-history fitting."""
    model = KaplanMeierRuntimeEstimator()
    assert not model.curve
    for method in (model.survival, model.remaining_quantile, model.remaining_mean):
        with pytest.raises(ValueError, match="fit"):
            method(0)
