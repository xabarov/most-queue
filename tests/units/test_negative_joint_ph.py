"""Physical limits and independent reference solvers for the v7 PH equations."""

import numpy as np
import pytest

from works.negative_queues.refinement_20261107.fixed_work_phase import calculate as fixed_stationary
from works.negative_queues.refinement_20261107.fixed_work_phase import service_moments, work_quadrature
from works.negative_queues.refinement_20261107.restart_phase import calculate as h2_reference
from works.negative_queues.revision_v7_20261107.ph_queue import (
    calculate,
    fixed_work,
    hyperexponential,
    occurrence_blocks,
    stationary,
)


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_resampling_agrees_with_separate_h2_implementation(mode):
    y, mu = [0.4, 0.6], [0.8, 1.6]
    ref = h2_reference(3, 1, 0.3, y, mu, mode, levels=70, behind=30, order=2)
    result = calculate(hyperexponential(y, mu), n=3, lam=1, delta=0.3, mode=mode, levels=70, behind=30)
    np.testing.assert_allclose(result["w"], ref["w"], rtol=1e-10)
    np.testing.assert_allclose(result["v"], ref["v"], rtol=1e-10)


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_one_server_retained_class_pollaczek_khinchine(mode):
    mean, cv, lam, delta = 0.7, 0.8, 0.6, 0.02
    work, weight = work_quadrature(mean, cv, 3)
    t = service_moments(work, weight, 4, delta, order=3)
    w1 = lam * t[1] / (2 * (1 - lam * t[0]))
    w2 = lam * t[2] / (3 * (1 - lam * t[0])) + 2 * w1 * w1
    expected = [w1 + t[0], w2 + 2 * w1 * t[0] + t[1]]
    result = calculate(
        fixed_work(cv, 3, 4, mean), n=1, lam=lam, delta=delta, mode=mode, action="fixed", levels=120, behind=20
    )
    np.testing.assert_allclose(result["w"], [w1, w2], rtol=1e-8)
    np.testing.assert_allclose(result["v"], expected, rtol=1e-8)


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_one_server_exponential_removal(mode):
    lam, mu, delta = 0.6, 1, 0.4
    a = mu + delta - lam
    rho = lam / (mu + delta)
    result = calculate(
        hyperexponential([1], [mu]), n=1, lam=lam, delta=delta, mode=mode, action="remove", levels=100, behind=30
    )
    if mode == "rcs":
        np.testing.assert_allclose(result["w"], [rho / a, 2 * rho / a**2], rtol=1e-9)
        np.testing.assert_allclose(result["v"], [1 / a, 2 / a**2], rtol=1e-9)
    assert max(result["little_residuals"]) < 1e-9


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_fixed_work_means_agree_with_stationary_only_engine(mode):
    ref = fixed_stationary(cv=0.8, mode=mode, classes=2, stages=3, levels=100, method="krylov", tolerance=1e-11)
    result = calculate(fixed_work(0.8, 2, 3), delta=0.02, mode=mode, action="fixed", levels=100, behind=30)
    np.testing.assert_allclose(result["w"], ref["w"], rtol=2e-6)
    assert result["v"][0] == pytest.approx(ref["v"][0], rel=2e-6)
    if mode == "disaster":
        np.testing.assert_allclose(result["v"], ref["v"], rtol=2e-6)


def test_clearance_stabilizes_overloaded_queue():
    result = calculate(
        hyperexponential([1], [1]), n=2, lam=4, delta=1, mode="disaster", action="remove", levels=100, behind=30
    )
    assert max(result["little_residuals"]) < 1e-7


def test_unstable_saturated_queue_is_rejected():
    with pytest.raises(ValueError, match="not stable"):
        calculate(hyperexponential([1], [1]), n=2, lam=2.5, delta=0.02, mode="rcs", action="restart")


@pytest.mark.parametrize("lam", [0.5, 3.5])
def test_clearing_resolvent_agrees_with_explicit_physical_generator(lam):
    from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph

    parts = occurrence_blocks(gamma_ph(2.1, 0.8, 2), 2)
    a, _, da = stationary(parts, 2, lam, 0.3, "disaster", "remove", 80)
    b, _, db = stationary(parts, 2, lam, 0.3, "disaster", "remove", 80, clearing_resolvent=False)
    np.testing.assert_allclose(np.concatenate(a), np.concatenate(b), atol=1e-13, rtol=1e-8)
    assert da["balance_residual"] < 1e-12 and db["balance_residual"] < 1e-12
