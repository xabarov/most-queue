"""
Unit tests for M/M/1 with a threshold-controlled service rate
(roadmap item R5; Morrison, Queueing Systems 4:213-235, 1989).

The model and the exact result are Morrison's. These check OUR implementation
against references independent of the code path under test:

1. the ordinary M/M/1, which the model must reproduce exactly whenever the two
   rates coincide or the threshold removes one of them;
2. the published stationary distribution of the same model (Adan & D'Auria,
   SIAM J. Appl. Math. 76(1), 2016, equations (1)-(2));
3. Little's law, ``E[V] = E[N]/lam`` and ``E[W] = E[N_q]/lam`` -- the recursion
   knows nothing about either, so this ties the sojourn moments to the
   stationary distribution through a completely separate route;
4. the same absorbing chain solved as an explicit sparse linear system with the
   number-behind truncated far out, which uses none of the finite-recursion
   machinery and in particular none of the Erlang boundary condition.

Agreement with simulation lives in tests/test_mm1_threshold_rate_vs_sim.py.
"""

import math

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from most_queue.theory.fifo.mm1_threshold_rate import MM1ThresholdRateCalc

LAM = 0.6


def _calc(threshold, mu_low, mu_high, lam=LAM):
    calc = MM1ThresholdRateCalc(threshold=threshold)
    calc.set_sources(lam)
    calc.set_servers(mu_low=mu_low, mu_high=mu_high)
    return calc


def _mm1_moments(lam, mu, num):
    """(sojourn, waiting) raw moments of an ordinary M/M/1."""
    sojourn = [math.factorial(k) / (mu - lam) ** k for k in range(1, num + 1)]
    waiting = [(lam / mu) * m for m in sojourn]
    return sojourn, waiting


def _matrix_conditional_moments(lam, mu_low, mu_high, threshold, found, num, m_max=900):
    """
    The same tagged-customer chain, solved as ``(-A) m_k = k m_{k-1}`` on an
    explicitly truncated state space.

    Deliberately shares nothing with the implementation: it has no Erlang
    boundary at all, it simply blocks arrivals once ``m_max`` customers have
    queued behind the tagged one and relies on that being far enough out.
    """
    levels = found + 1
    size = levels * (m_max + 1)

    def idx(r, m):
        return r * (m_max + 1) + m

    rows, cols, vals = [], [], []
    diag = np.zeros(size)
    for r in range(levels):
        for m in range(m_max + 1):
            src = idx(r, m)
            total = 0.0
            if m < m_max:
                rows.append(src)
                cols.append(idx(r, m + 1))
                vals.append(lam)
                total += lam
            rate = mu_low if r + 1 + m <= threshold else mu_high
            if r >= 1:
                rows.append(src)
                cols.append(idx(r - 1, m))
                vals.append(rate)
            total += rate  # at r == 0 this rate is the absorption
            diag[src] = total
    gen = sp.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsc() - sp.diags(diag)

    vec = np.ones(size)
    out = []
    for k in range(1, num + 1):
        vec = spla.spsolve((-gen).tocsc(), k * vec)
        out.append(float(vec[idx(found, 0)]))
    return out


@pytest.mark.parametrize("threshold", [0, 1, 3, 10, 40])
@pytest.mark.parametrize("mu", [1.0, 2.5])
def test_equal_rates_reduce_to_an_ordinary_mm1(threshold, mu):
    """Whatever the threshold, equal rates must give the textbook M/M/1."""
    calc = _calc(threshold, mu, mu)
    sojourn, waiting = _mm1_moments(LAM, mu, 4)
    assert np.allclose(calc.get_v(4), sojourn, rtol=1e-12)
    assert np.allclose(calc.get_w(4), waiting, rtol=1e-12)


@pytest.mark.parametrize("mu_low", [0.3, 1.0, 5.0])
def test_zero_threshold_removes_the_low_rate_entirely(mu_low):
    """
    ``K = 0`` means the high rate applies in every non-empty state, so the low
    rate cannot influence anything. It also exercises the Erlang boundary in
    isolation: the conditional sojourn must be exactly ``Erlang(n+1, mu_high)``.
    """
    mu_high = 1.4
    calc = _calc(0, mu_low, mu_high)
    sojourn, waiting = _mm1_moments(LAM, mu_high, 4)
    assert np.allclose(calc.get_v(4), sojourn, rtol=1e-12)
    assert np.allclose(calc.get_w(4), waiting, rtol=1e-12)
    for found in (0, 2, 5):
        erlang = [math.factorial(found + k) / math.factorial(found) / mu_high**k for k in range(1, 5)]
        assert np.allclose(calc.get_conditional_sojourn_moments(found, 4), erlang, rtol=1e-12)


def test_huge_threshold_behaves_like_the_low_rate_mm1():
    """With the threshold far above anything reachable, only the low rate is seen."""
    mu_low = 1.0
    calc = _calc(400, mu_low, 5.0)
    sojourn, _ = _mm1_moments(LAM, mu_low, 2)
    assert np.allclose(calc.get_v(2), sojourn, rtol=1e-6)


@pytest.mark.parametrize("threshold, mu_low, mu_high", [(2, 0.7, 1.4), (5, 1.5, 0.9), (1, 0.4, 1.2), (8, 0.4, 1.2)])
def test_stationary_distribution_matches_the_published_formula(threshold, mu_low, mu_high):
    """
    Adan & D'Auria (2016), equations (1)-(2) for the same model:
    ``pi_n = (lam/mu0)^n pi_0`` up to the threshold and
    ``(mu1/mu0)^K (lam/mu1)^n pi_0`` above it.
    """
    calc = _calc(threshold, mu_low, mu_high)
    pi0 = 1.0 / (
        sum((LAM / mu_low) ** n for n in range(threshold + 1)) + (LAM / (mu_high - LAM)) * (LAM / mu_low) ** threshold
    )
    probs = calc.get_p(threshold + 15)
    for n, value in enumerate(probs):
        expected = (
            (LAM / mu_low) ** n * pi0
            if n <= threshold
            else (mu_high / mu_low) ** threshold * (LAM / mu_high) ** n * pi0
        )
        assert np.isclose(value, expected, rtol=1e-12), n
    assert np.isclose(sum(calc.get_p(4000)), 1.0, atol=1e-9)


@pytest.mark.parametrize("threshold, mu_low, mu_high", [(2, 0.7, 1.4), (5, 1.5, 0.9), (8, 0.4, 1.2), (1, 2.0, 1.1)])
def test_little_law_ties_the_moments_to_the_stationary_distribution(threshold, mu_low, mu_high):
    """
    ``E[V] = E[N]/lam`` and ``E[W] = E[N_q]/lam``. The recursion never sees the
    stationary distribution except as the arrival-state weights, and never uses
    Little's law, so this is a genuine cross-check rather than an identity.
    """
    calc = _calc(threshold, mu_low, mu_high)
    probs = calc.get_p(6000)
    mean_in_system = sum(n * p for n, p in enumerate(probs))
    mean_in_queue = sum(max(n - 1, 0) * p for n, p in enumerate(probs))
    assert np.isclose(calc.get_v(1)[0], mean_in_system / LAM, rtol=1e-9)
    assert np.isclose(calc.get_w(1)[0], mean_in_queue / LAM, rtol=1e-9)
    assert np.isclose(calc.get_n_mean(), mean_in_system, rtol=1e-9)


@pytest.mark.parametrize("found", [0, 1, 3, 7])
@pytest.mark.parametrize("threshold, mu_low, mu_high", [(3, 0.7, 1.4), (2, 1.6, 0.8)])
def test_conditional_moments_match_an_explicit_matrix_solve(found, threshold, mu_low, mu_high):
    """
    The sharpest check available: the same absorbing chain, solved as a sparse
    linear system with no Erlang boundary condition at all. If the boundary were
    placed wrongly this would diverge; it agrees to machine precision.
    """
    calc = _calc(threshold, mu_low, mu_high)
    reference = _matrix_conditional_moments(LAM, mu_low, mu_high, threshold, found, 4)
    assert np.allclose(calc.get_conditional_sojourn_moments(found, 4), reference, rtol=1e-10)


def test_wait_probability_is_the_pasta_busy_probability():
    calc = _calc(3, 0.7, 1.4)
    assert np.isclose(calc.get_wait_prob(), 1.0 - calc.get_p(2)[0], rtol=1e-12)
    assert np.isclose(calc.get_conditional_waiting_moments(0, 2)[0], 0.0, atol=1e-15)


@pytest.mark.parametrize("threshold, mu_low, mu_high", [(3, 0.7, 1.4), (4, 1.6, 0.8)])
def test_service_time_lies_strictly_between_the_two_rates(threshold, mu_low, mu_high):
    """
    A customer may be served partly below and partly above the threshold, so the
    realised mean service time is neither ``1/mu_low`` nor ``1/mu_high``.
    """
    calc = _calc(threshold, mu_low, mu_high)
    low, high = sorted((1.0 / mu_low, 1.0 / mu_high))
    assert low < calc.get_service_time_mean() < high


def test_a_faster_high_rate_helps_and_a_higher_threshold_hurts():
    """
    Monotonicity in both controls, when the high rate is the faster one: raising
    it helps; letting the server stay slow for longer (bigger K) hurts.
    """
    previous = None
    for mu_high in (0.9, 1.2, 1.8, 3.0):
        value = _calc(3, 0.7, mu_high).get_v(1)[0]
        if previous is not None:
            assert value < previous
        previous = value

    previous = None
    for threshold in (0, 1, 3, 8, 20):
        value = _calc(threshold, 0.7, 1.8).get_v(1)[0]
        if previous is not None:
            assert value > previous
        previous = value


def test_stability_depends_on_the_high_rate_alone():
    """
    A low rate below the arrival rate is perfectly fine -- the queue grows until
    it crosses the threshold and is then drained faster. A high rate below the
    arrival rate is not.
    """
    calc = _calc(5, 0.2, 1.5, lam=0.9)
    assert calc.get_v(1)[0] > 0
    with pytest.raises(ValueError, match="unstable"):
        _calc(5, 5.0, 0.8, lam=0.9).get_v(1)


def test_moments_are_a_valid_sequence():
    calc = _calc(4, 0.6, 1.5)
    sojourn = calc.get_v(4)
    waiting = calc.get_w(4)
    assert all(x > 0 for x in sojourn)
    assert sojourn[1] > sojourn[0] ** 2  # positive variance
    assert waiting[1] > waiting[0] ** 2
    assert all(v > w for v, w in zip(sojourn, waiting))


def test_run_reports_consistent_metrics():
    calc = _calc(3, 0.7, 1.4)
    res = calc.run(3)
    assert np.allclose(res.v, calc.get_v(3))
    assert np.allclose(res.w, calc.get_w(3))
    assert 0.0 < res.utilization < 1.0
    assert np.isclose(sum(res.p), 1.0, atol=1e-6)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        MM1ThresholdRateCalc(threshold=-1)
    calc = MM1ThresholdRateCalc(threshold=2)
    with pytest.raises(ValueError, match="positive"):
        calc.set_sources(0.0)
    calc.set_sources(LAM)
    with pytest.raises(ValueError, match="positive"):
        calc.set_servers(mu_low=-1.0, mu_high=1.0)
    calc.set_servers(mu_low=1.0, mu_high=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        calc.get_conditional_sojourn_moments(-1)
    with pytest.raises(ValueError, match="beyond the computed range"):
        calc.get_conditional_sojourn_moments(10**6)
