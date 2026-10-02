"""Moment, CDF, endpoint and scaling checks independent of scheduling."""

import numpy as np
import pytest

from most_queue.random.map_ph import PHDistribution, PHParams
from most_queue.random.service_calibration import FAMILIES, ServiceCalibration


@pytest.mark.parametrize("sample", [[], [1], [0, 1], [-1, 2], [1, np.nan], [1, np.inf], [[1, 2]]])
def test_invalid_observations(sample):
    with pytest.raises(ValueError):
        ServiceCalibration.fit(sample)


@pytest.mark.parametrize("u", [[0], [1], [-0.1], [1.1], [np.nan], [np.inf]])
def test_invalid_quantiles(u):
    with pytest.raises(ValueError):
        ServiceCalibration.fit([1, 2]).quantiles("empirical", u)


def test_empirical_is_discrete_not_interpolated():
    model = ServiceCalibration.fit([4, 1, 2, 3])
    np.testing.assert_array_equal(model.quantiles("empirical", [0.01, 0.25, 0.251, 0.99]), [1, 1, 2, 4])
    assert model.samples == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        model.quantiles("unsupported", [0.5])


@pytest.mark.parametrize("family", FAMILIES)
def test_scale_equivariance_and_monotonicity(family):
    sample = [1] * 30 + [100, 200]
    u = np.linspace(0.001, 0.999, 300)
    first = ServiceCalibration.fit(sample).quantiles(family, u)
    second = ServiceCalibration.fit(np.array(sample) * 7).quantiles(family, u)
    np.testing.assert_allclose(second, 7 * first, rtol=1e-12)
    assert np.all(np.diff(first) >= 0)


def test_h2_inverse_cdf_and_exact_moments():
    model = ServiceCalibration.fit([1] * 90 + [100] * 10)
    p = (1 + np.sqrt((model.cv2 - 1) / (model.cv2 + 1))) / 2
    rates = np.array([2 * p, 2 * (1 - p)]) / model.mean
    params = PHParams(alpha=np.array([p, 1 - p]), T=-np.diag(rates))
    moments = PHDistribution.calc_theory_moments(params, 2)
    assert moments[0] == pytest.approx(model.mean)
    assert moments[1] == pytest.approx(model.mean**2 * (1 + model.cv2))
    u = np.array([1e-10, 0.01, 0.5, 0.99, 1 - 1e-10])
    x = model.quantiles("ph", u)
    np.testing.assert_allclose(1 - p * np.exp(-rates[0] * x) - (1 - p) * np.exp(-rates[1] * x), u, atol=1e-15)


def test_low_variance_and_deterministic_boundary_are_explicit():
    model = ServiceCalibration.fit([2, 2, 2])
    assert model.parameters()["ph_phases"] == 64
    assert model.parameters()["ph_cv2"] == 1 / 64
    np.testing.assert_allclose(model.quantiles("lognormal", [0.1, 0.9]), [2, 2])
    assert model.quantiles("ph", [0.1])[0] < 2 < model.quantiles("ph", [0.9])[0]


def test_inverse_integration_recovers_mean_for_nonempirical_laws():
    model = ServiceCalibration.fit([1] * 90 + [100] * 10)
    u = (np.arange(100_000) + 0.5) / 100_000
    for family in ("exponential", "ph", "lognormal"):
        x = model.quantiles(family, u)
        assert x.mean() == pytest.approx(model.mean, rel=0.005)
        if family != "exponential":
            assert x.var() / x.mean() ** 2 == pytest.approx(model.cv2, rel=0.08)
