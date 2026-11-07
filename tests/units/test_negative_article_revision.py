"""Independent regressions for the negative-queues manuscript audit."""

import math

import numpy as np
import pytest

from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import DisasterScenario, NegativeServiceType, QsSimNegatives, RcsScenario
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc
from most_queue.theory.utils.restarts import (
    beff_moments_repeat_without_resampling_from_h2,
    beff_moments_repeat_without_resampling_from_moments,
    beff_pls_from_beta,
    effective_service_moments,
)
from works.negative_queues.revision_20261107.verify_models import stationary


def model(cls, n=3, cv=1.2, delta=0.3, gamma=None):
    c = cls(n, calc_params=TakahashiTakamiParams(N=70, tolerance=1e-12))
    c.set_sources(1.0, delta)
    c.set_servers(gamma_moments_by_mean_and_cv(0.7 * n, cv))
    if gamma is not None:
        c.gamma = gamma
    c.run()
    return c


def waiting_reference(c, states, pi, mode):
    """First-step moment equations, using physical rates and independent weights."""
    n = c.n
    p = np.zeros((n + 1, n + 1), complex)
    rates = np.zeros(n + 1, complex)
    for j in range(n + 1):
        extra = c.l_neg / n if mode == "rcs" else 0
        a = (n - j) * (c.mu[0] + extra)
        b = j * (c.mu[1] + extra)
        rates[j] = a + b
        p[j, j] = (a * c.y[0] + b * c.y[1]) / rates[j]
        if j < n:
            p[j, j + 1] = a * c.y[1] / rates[j]
        if j:
            p[j, j - 1] = b * c.y[0] / rates[j]
    exit_rate = rates + (c.l_neg if mode == "disaster" else 0)
    continuation = rates / exit_rate
    first, second = np.zeros(n + 1, complex), np.zeros(n + 1, complex)
    result = np.zeros(2, complex)
    weights = {s: prob for s, prob in zip(states, pi)}
    for level in range(n, c.N):
        prev = p @ first
        second = 2 / exit_rate**2 + 2 * continuation / exit_rate * prev + continuation * (p @ second)
        first = 1 / exit_rate + continuation * prev
        ys = np.array([weights[(level, j)] for j in range(n + 1)])
        result += [ys @ first, ys @ second]
    return result.real


@pytest.mark.parametrize("n,cv", [(1, 1.2), (3, 1.2), (3, 0.8)])
@pytest.mark.parametrize("mode,cls", [("rcs", MGnNegativeRCSCalc), ("disaster", MGnNegativeDisasterCalc)])
def test_stationary_and_waiting_against_independent_equations(n, cv, mode, cls):
    c = model(cls, n, cv)
    states, pi, p, residual = stationary(n, 1.0, 0.3, c.y, c.mu, mode, c.N)
    assert residual < 1e-12
    np.testing.assert_allclose(c.get_p(), p, atol=3e-10)
    assert c.get_utilization() == pytest.approx(sum(min(k, n) * pk for k, pk in enumerate(p)) / n, abs=1e-9)
    np.testing.assert_allclose(c.get_w(2)[:2], waiting_reference(c, states, pi, mode), rtol=3e-6, atol=2e-7)
    assert abs(c.w[0] - sum(max(k - n, 0) * pk for k, pk in enumerate(p))) < 1e-8
    if mode == "disaster":
        assert abs(c.v[0] - sum(k * pk for k, pk in enumerate(p))) < 1e-8
        q = c._z0_pls(0.3).real
        np.testing.assert_allclose(
            np.array(c.get_v_served()[:2]) * q + np.array(c.get_v_broken()[:2]) * (1 - q), c.v[:2], rtol=2e-7
        )


@pytest.mark.parametrize("gamma", [0.02, 1.0, 1e4])
def test_disaster_auxiliary_rate_and_idempotent_probabilities(gamma):
    c = model(MGnNegativeDisasterCalc, gamma=gamma)
    p = np.array(c.get_p())
    np.testing.assert_allclose(c.get_p(), p, atol=1e-14)
    assert abs(c.w[0] - 0.16462215483024) < 2e-9
    assert abs(c.v[0] - 1.3116947723852) < 2e-9


def test_base_waiting_laplace_order_matches_little():
    c = MGnCalc(3, calc_params=TakahashiTakamiParams(N=90, tolerance=1e-12))
    c.set_sources(1.0)
    c.set_servers(gamma_moments_by_mean_and_cv(2.1, 1.2))
    c.run()
    h = 1e-4
    derivative = (c._calc_w_lst(h) - c._calc_w_lst(-h)) / (2 * h)
    assert abs(-derivative.real - c.w[0]) < 1e-6


def test_exponential_restart_and_moment_existence():
    b = [math.factorial(k) * 2**k for k in range(1, 5)]
    for r in [0.01, 0.2, 2.0]:
        np.testing.assert_allclose(effective_service_moments(b, [0.4, 0.6], [0.5, 0.5], r), b, rtol=1e-9)
    finite = beff_moments_repeat_without_resampling_from_h2([0.4, 0.6], [0.5, 0.5], 0.3, num_of_moments=2)
    assert finite[0] == pytest.approx(5.0)
    assert math.isinf(finite[1])
    assert math.isinf(beff_moments_repeat_without_resampling_from_h2([0.4, 0.6], [0.5, 0.5], 0.6, num_of_moments=1)[0])
    assert beff_pls_from_beta(lambda s: 1 / (1 + s), 0.0, 0.0) == 1.0
    assert beff_moments_repeat_without_resampling_from_moments(2.0, 0.0, 0.0, num_of_moments=3) == [2.0, 4.0, 8.0]


@pytest.mark.parametrize("mode", [NegativeServiceType.RCS, NegativeServiceType.DISASTER])
def test_instant_restart_does_not_recount_waiting(mode):
    sim = QsSimNegatives(
        1,
        mode,
        verbose=False,
        rcs_scenario=RcsScenario.REQUEUE,
        disaster_scenario=DisasterScenario.REQUEUE_ALL,
        seed=123,
    )
    sim.set_positive_sources(0.1, "M")
    sim.set_negative_sources(1.0, "M")
    sim.set_servers(1.0, "M")
    sim.positive_arrival()
    sim.w = [3.0, 9.0, 27.0, 81.0]  # A distinctive existing observation must remain unchanged.
    before = (sim.taked, list(sim.w))
    sim.negative_arrival()
    assert (sim.taked, sim.w) == before
    assert sim.in_sys == 1 and sim.free_channels == 0


@pytest.mark.parametrize("cls", [MGnNegativeRCSCalc, MGnNegativeDisasterCalc])
def test_invalid_restart_model_fails_instead_of_changing_rate(cls):
    c = cls(1, requeue_on_disaster=True, repeat_without_resampling=True)
    c.set_sources(0.5, 2.0)
    c.set_servers(gamma_moments_by_mean_and_cv(1.0, 1.2))
    with pytest.raises(ValueError, match="moments do not exist"):
        c.run()


def test_simulation_rng_is_reproducible_including_rcs_selection():
    outcomes = []
    for _ in range(2):
        sim = QsSimNegatives(3, NegativeServiceType.RCS, verbose=False, seed=4711)
        sim.set_positive_sources(1.0, "M")
        sim.set_negative_sources(0.3, "M")
        sim.set_servers(0.5, "M")
        for _ in range(1000):
            sim.run_one_step()
        outcomes.append((sim.ttek, sim.served, sim.broken, sim.w, sim.v))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("cls", [MGnNegativeRCSCalc, MGnNegativeDisasterCalc])
def test_zero_negative_rate_recovers_base_model(cls):
    c = model(cls, delta=0.0)
    b = MGnCalc(3, calc_params=TakahashiTakamiParams(N=70, tolerance=1e-12))
    b.set_sources(1.0)
    b.set_servers(gamma_moments_by_mean_and_cv(2.1, 1.2))
    r = b.run()
    np.testing.assert_allclose(c.get_p(), r.p, atol=2e-10)
    np.testing.assert_allclose(c.w[:2], r.w[:2], rtol=1e-6)
    np.testing.assert_allclose(c.v[:2], r.v[:2], rtol=1e-6)
    assert c.get_v_broken() == [0.0] * 4
