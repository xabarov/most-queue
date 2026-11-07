"""Independent transform and moment checks of positive service representations."""

import numpy as np
import pytest
from scipy.special import poch

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, hyperexponential
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph, positive_h2_equivalent, transform


@pytest.mark.parametrize("cv", [0.8, 1.2, 0.5])
@pytest.mark.parametrize("j", [2, 4, 8])
def test_positive_gamma_parameters_and_quadrature_moments(cv, j):
    mean, shape = 2.1, 1 / cv**2
    service = gamma_ph(mean, cv, j)
    off = service.generator - np.diag(np.diag(service.generator))
    assert service.alpha.min() >= 0 and abs(service.alpha.sum() - 1) < 1e-13
    assert off.min() >= 0 and service.exit.min() >= -1e-13
    order = min(2 * j - 1, 9)
    exact = [poch(shape, k) * (mean / shape) ** k for k in range(1, order + 1)]
    np.testing.assert_allclose(service.moments(order), exact, rtol=5e-12)


def test_exact_erlang_transform_and_gamma_transform_refinement():
    s_values = np.logspace(-3, 2, 150)
    exact_erlang = (1 + 2.1 * s_values / 4) ** -4
    np.testing.assert_allclose([transform(gamma_ph(2.1, 0.5), s) for s in s_values], exact_erlang, rtol=1e-12)
    errors = []
    for j in [2, 4, 8]:
        shape = 1 / 0.8**2
        exact = (1 + 2.1 * s_values / shape) ** -shape
        errors.append(max(abs(transform(gamma_ph(2.1, 0.8, j), s) - value) for s, value in zip(s_values, exact)))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
@pytest.mark.parametrize("action", ["restart", "remove"])
def test_coxian_equivalence_of_signed_real_h2(mode, action):
    h = H2Distribution.get_params_clx(gamma_moments_by_mean_and_cv(2.1, 0.8))
    signed = hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2])
    positive = positive_h2_equivalent(signed.alpha, -signed.generator.diagonal())
    for s in [0, 0.01, 0.3, 1, 10]:
        assert transform(positive, s) == pytest.approx(transform(signed, s), rel=1e-12)
    options = dict(n=2, lam=0.6, delta=0.3, mode=mode, action=action, levels=70, behind=25)
    a, b = calculate(signed, **options), calculate(positive, **options)
    np.testing.assert_allclose(a["w"], b["w"], rtol=1e-10)
    np.testing.assert_allclose(a["v"], b["v"], rtol=1e-10)
