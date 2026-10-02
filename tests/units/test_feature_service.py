"""Exact ECDF scores, scale invariance, context fallback and invalid inputs."""

import numpy as np
import pytest

from most_queue.random.feature_service import EmpiricalPrediction, FeatureConditionalEmpirical, request_bucket
from most_queue.random.service_calibration import ServiceCalibration


@pytest.mark.parametrize("samples,observed", [([1, 1], 1), ([1, 1, 3, 9], 5), ([4, 2, 2], 0.1), ([1, 8], 100)])
def test_crps_matches_independent_pairwise_energy(samples, observed):
    prediction = EmpiricalPrediction(ServiceCalibration.fit(samples))
    x = np.array(samples)
    expected = np.mean(abs(x - observed)) - 0.5 * np.mean(abs(x[:, None] - x))
    assert prediction.crps(observed) == pytest.approx(expected)
    scaled = EmpiricalPrediction(prediction.distribution, 7)
    assert scaled.crps(7 * observed) == pytest.approx(7 * expected)


def test_inverse_cdf_ties_mean_and_strict_exceedance():
    prediction = EmpiricalPrediction(ServiceCalibration.fit([1, 1, 3, 9]), 2)
    assert [prediction.quantile(u) for u in (0.1, 0.5, 0.500001, 0.9)] == [2, 2, 6, 18]
    assert prediction.mean == 7
    assert prediction.probability_exceeding(2) == 0.5
    assert prediction.probability_exceeding(18) == 0


def test_hierarchical_fallback_uses_only_training_counts():
    model = FeatureConditionalEmpirical.fit(
        [2, 2, 3, 3, 4, 1], [2, 4, 6, 8, 10, 12], ["a", "a", "a", "b", "b", "a"], exact=True, minimum=2
    )
    assert model.predict(2, "a").level == "exact_context"
    assert model.predict(3, "a").level == "coarse_context"
    assert model.predict(3, "unknown").level == "coarse"
    assert model.predict(1, "a").level == "pooled"
    assert model.predict(64, None).level == "pooled"
    assert model.predict(2, "a").mean == 3
    assert model.predict(3, "a").mean == 4
    assert model.predict(3, "unknown").mean == 6
    assert model.predict(64, None).mean == 7


def test_ratio_preserves_excess_demand_and_unknown_scale_falls_back():
    model = FeatureConditionalEmpirical.fit(
        [1, 1, 2, 2], [2, 12, 50, 100], ["a"] * 4, scales=[2, 4, None, None], minimum=2
    )
    prediction = model.predict(1, "a", scale=10)
    assert prediction.distribution.samples == (1, 3)
    assert prediction.mean == 20
    assert prediction.probability_exceeding(10) == 0.5
    assert prediction.quantile(0.9) == 30  # not capped at request=10
    assert model.predict(1, "a").mean == 7
    assert model.predict(1, "a").level == "absolute_coarse"
    assert model.predict(64, "new", scale=3).level == "pooled"
    assert model.audit["missing_scale_count"] == 2


@pytest.mark.parametrize("scales", [[None, None], [10, None]])
def test_no_estimable_ratio_pool_uses_absolute(scales):
    model = FeatureConditionalEmpirical.fit([1, 1], [2, 4], [None, None], scales=scales)
    assert model.predict(1, None, scale=100).mean == 3
    assert model.predict(1, None, scale=100).level == "absolute_pooled"


def test_pooled_minimum_is_two_not_cell_threshold():
    model = FeatureConditionalEmpirical.fit([1, 1], [2, 6], [None, None], scales=[2, 2], minimum=20)
    assert model.predict(1, "new", scale=4).mean == 8
    assert model.predict(1, "new", scale=4).level == "pooled"


@pytest.mark.parametrize("value,expected", [(None, None), (1, "0"), (300, "0"), (301, "1"), (86400, "4"), (86401, "5")])
def test_fixed_request_bins(value, expected):
    assert request_bucket(value) == expected


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "3"])
def test_invalid_positive_scalars(bad):
    with pytest.raises(ValueError):
        request_bucket(bad)
    distribution = ServiceCalibration.fit([1, 2])
    with pytest.raises(ValueError):
        EmpiricalPrediction(distribution, bad)
    prediction = EmpiricalPrediction(distribution)
    with pytest.raises(ValueError):
        prediction.crps(bad)
    with pytest.raises(ValueError):
        prediction.probability_exceeding(bad)


@pytest.mark.parametrize("bad", [0, 1, -1, 2, float("nan"), True, "0.5"])
def test_invalid_quantile(bad):
    with pytest.raises(ValueError):
        EmpiricalPrediction(ServiceCalibration.fit([1, 2])).quantile(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"needs": [True, 1]},
        {"needs": [1.0, 1]},
        {"needs": [0, 1]},
        {"contexts": ["a"]},
        {"contexts": ["", "a"]},
        {"contexts": [2, "a"]},
        {"services": [1]},
        {"services": [0, 1]},
        {"minimum": 1},
        {"minimum": True},
        {"exact": 1},
        {"scales": [1]},
        {"scales": [1, 0]},
    ],
)
def test_invalid_fit_inputs(kwargs):
    arguments = {"needs": [1, 1], "services": [2, 3], "contexts": ["a", "a"]}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        FeatureConditionalEmpirical.fit(**arguments)


def test_invalid_prediction_arguments_and_support_overflow():
    model = FeatureConditionalEmpirical.fit([1, 1], [2, 3], ["a", "a"])
    for need, label, scale in [(True, "a", None), (1, "", None), (1, 7, None), (1, "a", 0), (1, "a", 2)]:
        with pytest.raises(ValueError):
            model.predict(need, label, scale=scale)
    with pytest.raises(ValueError):
        EmpiricalPrediction(ServiceCalibration.fit([2, 3]), 1e308)
