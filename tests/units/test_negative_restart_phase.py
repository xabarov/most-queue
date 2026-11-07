"""Independent exact limits for the manuscript's resampling phase calculation."""

import math

import numpy as np
import pytest

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.utils.restarts import effective_service_moments
from works.negative_queues.refinement_20261107.diagnose import phase_reference
from works.negative_queues.refinement_20261107.restart_phase import calculate


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_one_server_matches_pollaczek_khinchine(mode):
    b = gamma_moments_by_mean_and_cv(0.7, 1.2)
    h = H2Distribution.get_params_clx(b)
    y, mu = [h.p1, 1 - h.p1], [h.mu1, h.mu2]
    effective = effective_service_moments(b, y, mu, 0.3, num_of_moments=3)
    a = 1 - effective[0]
    w1 = effective[1] / (2 * a)
    w2 = effective[2] / (3 * a) + 2 * w1 * w1
    v = [w1 + effective[0], w2 + 2 * w1 * effective[0] + effective[1]]
    result = calculate(1, 1, 0.3, y, mu, mode, levels=100, behind=100, order=2)
    np.testing.assert_allclose(result["w"], [w1, w2], rtol=1e-9)
    np.testing.assert_allclose(result["v"], v, rtol=1e-9)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_exponential_resampling_is_erlang_c(n, mode):
    mu, lam = 1.0, 0.7 * n
    offered = lam / mu
    rho = offered / n
    tail = offered**n / (math.factorial(n) * (1 - rho))
    erlang_c = tail / (sum(offered**k / math.factorial(k) for k in range(n)) + tail)
    w = [erlang_c * math.factorial(k) / (n * mu - lam) ** k for k in [1, 2]]
    v = [w[0] + 1 / mu, w[1] + 2 * w[0] / mu + 2 / mu**2]
    result = calculate(n, lam, 2, [0.3, 0.7], [mu, mu], mode, levels=100, behind=100, order=2)
    np.testing.assert_allclose(result["w"], w, rtol=1e-8)
    np.testing.assert_allclose(result["v"], v, rtol=1e-8)


@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_zero_resets_recover_h2_queue(mode):
    b = gamma_moments_by_mean_and_cv(2.1, 1.2)
    h = H2Distribution.get_params_clx(b)
    result = calculate(3, 1, 0, [h.p1, 1 - h.p1], [h.mu1, h.mu2], mode, levels=120, behind=120, order=2)
    base = MGnCalc(3, calc_params=TakahashiTakamiParams(N=120, tolerance=1e-12))
    base.set_sources(1)
    base.set_servers(b)
    reference = base.run()
    np.testing.assert_allclose(result["w"], reference.w[:2], rtol=1e-8)
    np.testing.assert_allclose(result["v"], reference.v[:2], rtol=1e-8)


@pytest.mark.parametrize("cv", [0.8, 1.2])
@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_direct_phase_states_and_first_step_waiting(cv, mode):
    b = gamma_moments_by_mean_and_cv(2.1, cv)
    h = H2Distribution.get_params_clx(b)
    y, mu = [h.p1, 1 - h.p1], [h.mu1, h.mu2]
    result = calculate(3, 1, 0.3, y, mu, mode, levels=120, behind=120, order=2)
    reference = phase_reference(cv, mode, levels=120)
    np.testing.assert_allclose(result["w"], reference["w"], atol=5e-9)
    assert result["v"][0] == pytest.approx(reference["v1"], abs=5e-9)
    assert result["balance_residual"] < 1e-12
    assert max(result["little_residuals"]) < 1e-8
    if mode == "disaster":
        # Strong Markov property: the future reset process after service starts
        # is independent of the initial wait, despite dependence among servers.
        # Use the actual phase transform, also for the algebraic CV<1 fit;
        # a Gamma law with matching moments is a different distribution.
        rate = 0.3
        beta = sum(yi * mui / (mui + rate) for yi, mui in zip(y, mu))
        derivative = -sum(yi * mui / (mui + rate) ** 2 for yi, mui in zip(y, mu))
        service = [(1 - beta) / (rate * beta), 2 * (1 - beta + rate * derivative) / (rate**2 * beta**2)]
        np.testing.assert_allclose(
            result["v"],
            [result["w"][0] + service[0], result["w"][1] + 2 * result["w"][0] * service[0] + service[1]],
            rtol=1e-8,
        )
