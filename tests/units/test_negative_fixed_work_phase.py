"""Checks of retained-work semantics and independently known queueing limits."""

import math

import numpy as np
import pytest

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.refinement_20261107.fixed_work_phase import (
    blocks,
    calculate,
    service_moments,
    work_quadrature,
)


@pytest.mark.parametrize("cv", [0.8, 1.2])
def test_quadrature_preserves_the_selected_input_law(cv):
    x, w = work_quadrature(2.1, cv, 4)
    assert min(x) > 0 and min(w) > 0 and sum(w) == pytest.approx(1)
    if cv < 1:
        expected = [2.1**k * math.prod(1 + j * cv**2 for j in range(k)) for k in range(1, 7)]
    else:
        h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(2.1, cv))
        expected = [math.factorial(k) * (h.p1 / h.mu1**k + (1 - h.p1) / h.mu2**k) for k in range(1, 7)]
    np.testing.assert_allclose([sum(w * x**k) for k in range(1, 7)], expected, rtol=1e-10)


def test_reset_retains_work_classes_and_is_simultaneous():
    for mode in ["rcs", "disaster"]:
        states, inside, _, _, _ = blocks(2, 1, 0.4, np.array([1.0, 3.0]), np.array([0.3, 0.7]), 3, mode)
        row = states[2].index((1, 5))
        both = states[2].index((0, 3))
        first = states[2].index((0, 5))
        second = states[2].index((1, 3))
        if mode == "disaster":
            assert inside[2][row, both] == pytest.approx(0.4)
            assert inside[2][row, first] == 0 and inside[2][row, second] == 0
        else:
            assert inside[2][row, both] == 0
            assert inside[2][row, first] == pytest.approx(0.2)
            assert inside[2][row, second] == pytest.approx(0.2)


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_single_server_retained_class_matches_pk(mode):
    lam = 0.2
    result = calculate(cv=1.2, mode=mode, n=1, lam=lam, delta=0.02, classes=3, stages=3, levels=100)
    s = service_moments(np.array(result["work"]), np.array(result["weights"]), 3, 0.02)
    w1 = lam * s[1] / (2 * (1 - lam * s[0]))
    w2 = lam * s[2] / (3 * (1 - lam * s[0])) + 2 * w1 * w1
    np.testing.assert_allclose(result["w"], [w1, w2], rtol=2e-7)
    assert result["v"][0] == pytest.approx(w1 + s[0], rel=2e-7)
    assert result["tail"] < 1e-12
    assert result["embedded_chain_residual_l1"] < 1e-8


def test_erlangization_converges_to_repeating_the_same_deterministic_work():
    b, r = 2.1, 0.2
    exact = np.array([(math.exp(r * b) - 1) / r, 2 * math.exp(r * b) * (math.exp(r * b) - 1 - r * b) / r**2])
    errors = []
    for stages in [4, 16, 64, 256, 1024]:
        moments = service_moments(np.array([b]), np.array([1.0]), stages, r, order=2)
        errors.append(np.max(abs(moments / exact - 1)))
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.001


def test_two_stationary_solvers_agree():
    parameters = dict(cv=1.2, n=2, lam=0.5, classes=2, stages=3, levels=70)
    power = calculate(**parameters, method="power")
    krylov = calculate(**parameters, method="krylov")
    np.testing.assert_allclose(krylov["w"], power["w"], rtol=2e-7)
    np.testing.assert_allclose(krylov["v"], power["v"], rtol=2e-7)
    assert min(krylov["p"]) >= -1e-12
