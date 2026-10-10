"""
Unit tests for GI/M/2 with a busy-server-dependent service rate
(roadmap item R4; Bhat, AISM 18:211-221, 1966).

The model and solution are Bhat's; these check OUR steady-state limit of his
(45)-(47) against four references it cannot have been fitted to:

1. his own published table (56) -- Poisson arrivals at sigma = 1/4, 1/2, 1;
2. the elementary M/M/2 birth-death chain, for ARBITRARY sigma (including
   sigma > 1, which the table does not cover);
3. an exact CTMC, available because Erlang interarrivals make the whole system
   a finite-phase Markov chain -- this is the sharpest reference of the four and
   pins both the time-stationary and the arrival-observed distributions to
   machine precision for a genuinely state-dependent rate;
4. the library's own GiMn at the two degenerate ratios.

Agreement with simulation lives in tests/test_gi_m2_state_dependent_vs_sim.py.
"""

import math

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import simpson

from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.params import ErlangParams, GammaParams
from most_queue.theory.fifo.gi_m2_state_dependent import GiM2StateDependentCalc
from most_queue.theory.fifo.gi_m_n import GiMn

ARRIVAL_RATE = 1.2
MU = 1.0


def _calc(arrival_params, mu=MU, mu_single=None):
    calc = GiM2StateDependentCalc()
    calc.set_sources_params(arrival_params)
    calc.set_servers(mu=mu, mu_single=mu_single)
    return calc


def _poisson(rate=ARRIVAL_RATE):
    return GammaParams(mu=rate, alpha=1.0)


def _exact_ctmc(r, theta, mu, mu_single, nmax=3000):
    """
    Exact stationary solution for Erlang(r) interarrivals.

    With phase-type arrivals the system is an ordinary CTMC on
    ``(number in system, arrival phase)``, so it can be solved directly. An
    arrival occurs on leaving the last phase, which is what makes the
    arrival-observed distribution readable off the same solution.
    """
    size = (nmax + 1) * r
    rows, cols, vals = [], [], []
    for n in range(nmax + 1):
        for k in range(r):
            src = n * r + k
            if k < r - 1:
                dst, rate = n * r + k + 1, theta
            elif n < nmax:
                dst, rate = (n + 1) * r, theta
            else:
                dst = rate = None
            if dst is not None:
                rows.append(src)
                cols.append(dst)
                vals.append(rate)
            if n >= 1:
                rows.append(src)
                cols.append((n - 1) * r + k)
                vals.append(mu_single if n == 1 else 2 * mu)
    q = sp.coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsr()
    q = q - sp.diags(np.asarray(q.sum(axis=1)).ravel())
    lhs = q.T.tolil()
    lhs[0, :] = 1.0
    rhs = np.zeros(size)
    rhs[0] = 1.0
    pi = spla.spsolve(lhs.tocsr(), rhs)
    grid = pi.reshape(nmax + 1, r)
    arrival_seen = grid[:, r - 1]
    return grid.sum(axis=1), arrival_seen / arrival_seen.sum()


@pytest.mark.parametrize(
    "sigma, p0_formula, eq_formula",
    [
        (0.25, lambda r: (1 - r) / (1 + 3 * r), lambda r: 4 * r / ((1 - r) * (1 + 3 * r))),
        (0.5, lambda r: (1 - r) / (1 + r), lambda r: 2 * r / ((1 - r) * (1 + r))),
        (1.0, lambda r: 1 - r, lambda r: r / (1 - r)),
    ],
)
def test_matches_bhats_published_table(sigma, p0_formula, eq_formula):
    """Table (56) of the paper: P_0 and E[Q] for Poisson input at three ratios."""
    calc = _calc(_poisson(), mu_single=2 * MU * sigma)
    rho = ARRIVAL_RATE / (2 * MU)
    assert np.isclose(calc.get_p(3)[0], p0_formula(rho), rtol=1e-12)
    assert np.isclose(calc.get_q(), eq_formula(rho), rtol=1e-9)


@pytest.mark.parametrize("sigma", [0.2, 0.5, 0.8, 1.0, 1.7, 3.0])
def test_poisson_case_matches_the_elementary_birth_death_chain(sigma):
    """
    With Poisson input the system is a plain birth-death chain and the answer is
    elementary: ``P_0 = sigma(1-rho)/(sigma(1-rho)+rho)``, ``P_j = rho^j P_0/sigma``.
    Bhat's table covers three ratios; this covers arbitrary ones, including the
    strongly accelerated ``sigma > 1`` the table says nothing about.
    """
    calc = _calc(_poisson(), mu_single=2 * MU * sigma)
    rho = ARRIVAL_RATE / (2 * MU)
    p0 = sigma * (1 - rho) / (sigma * (1 - rho) + rho)
    expected = [p0] + [rho**j * p0 / sigma for j in range(1, 10)]
    assert np.allclose(calc.get_p(10), expected, rtol=1e-10)


@pytest.mark.parametrize("r, mu_single", [(4, 1.6), (4, 0.6), (3, 1.4), (2, 1.0), (5, 2.0)])
def test_matches_an_exact_ctmc_for_erlang_arrivals(r, mu_single):
    """
    Erlang interarrivals turn the system into a finite-phase CTMC that can be
    solved exactly, independently of Bhat's transforms. Both the time-stationary
    and the arrival-observed distributions must match -- and they are genuinely
    different, since PASTA does not hold for renewal input.
    """
    theta = r * ARRIVAL_RATE
    calc = _calc(ErlangParams(r=r, mu=theta), mu_single=mu_single)
    p_exact, pi_exact = _exact_ctmc(r, theta, MU, mu_single)

    assert np.allclose(calc.get_p(10), p_exact[:10], atol=1e-12)
    assert np.allclose(calc.get_pi(10), pi_exact[:10], atol=1e-12)
    assert np.isclose(calc.get_q(), float(sum(j * p_exact[j] for j in range(len(p_exact)))), rtol=1e-9)
    # ...and the two distributions really do differ, so the test above has teeth.
    assert not np.allclose(calc.get_p(6), calc.get_pi(6), atol=1e-3)


@pytest.mark.parametrize("cv", [0.5, 1.0, 1.5])
def test_equal_rates_reduce_to_the_ordinary_gi_m_2(cv):
    """mu_single == mu is an ordinary GI/M/2 -- states AND waiting times."""
    mean_a = 1.0 / ARRIVAL_RATE
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean_a, cv)
    moments = GammaDistribution.calc_theory_moments(gamma_params, 4)

    calc = GiM2StateDependentCalc()
    calc.set_sources(moments)
    calc.set_servers(mu=MU)

    ref = GiMn(n=2)
    ref.set_sources(moments)
    ref.set_servers(MU)

    assert np.allclose(calc.get_p(12), ref.get_p()[:12], atol=1e-8)
    assert np.allclose(calc.get_w(2), ref.get_w(2), rtol=1e-5)


@pytest.mark.parametrize("cv", [0.5, 1.0, 1.5])
def test_double_single_rate_matches_gi_m_1_in_STATES_but_not_in_waiting(cv):
    """
    ``mu_single == 2*mu`` leaves the total drain rate at ``2*mu`` in every state,
    so the OCCUPANCY process is exactly a GI/M/1 of rate ``2*mu``.

    The waiting times are a different matter and must NOT agree: there are still
    two servers here, so a customer that finds one job in system goes straight to
    the free one, whereas in a real GI/M/1 it would queue. Same occupancy
    process, different per-customer experience -- asserting the difference keeps
    the reduction honest instead of over-claiming it.
    """
    mean_a = 1.0 / ARRIVAL_RATE
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean_a, cv)
    moments = GammaDistribution.calc_theory_moments(gamma_params, 4)

    calc = GiM2StateDependentCalc()
    calc.set_sources(moments)
    calc.set_servers(mu=MU, mu_single=2 * MU)

    ref = GiMn(n=1)
    ref.set_sources(moments)
    ref.set_servers(2 * MU)

    assert np.allclose(calc.get_p(12), ref.get_p()[:12], atol=1e-8)
    assert calc.get_w(1)[0] < ref.get_w(1)[0] * 0.9


@pytest.mark.parametrize("mu_single", [0.4, 1.0, 1.9])
def test_distributions_are_proper(mu_single):
    calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=mu_single)
    p, pi = calc.get_p(600), calc.get_pi(600)
    assert all(x >= -1e-15 for x in p) and all(x >= -1e-15 for x in pi)
    assert np.isclose(sum(p), 1.0, atol=1e-9)
    assert np.isclose(sum(pi), 1.0, atol=1e-9)


@pytest.mark.parametrize("mu_single", [0.4, 1.0, 1.9])
def test_waiting_time_is_an_atom_plus_one_exponential(mu_single):
    """
    The positive part of the wait is ``Exp(2 mu (1 - gamma))``: the moments, the
    tail and the integral of the tail must all agree with that single statement.
    """
    calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=mu_single)
    wait_prob = calc.get_wait_prob()
    moments = calc.get_w(4)
    rate = -math.log(calc.get_w_tail(2.0) / wait_prob) / 2.0  # read the rate off the tail

    for k, moment in enumerate(moments, start=1):
        assert np.isclose(moment, wait_prob * math.factorial(k) / rate**k, rtol=1e-9)
    assert np.isclose(calc.get_w_tail(0.0), wait_prob, rtol=1e-12)
    # Simpson, not trapezoid: the tail is a convex exponential, where the
    # trapezoid rule's O(dt^2) error alone is ~1e-5 on any sane grid.
    grid = np.linspace(0.0, 400.0, 40001)
    assert np.isclose(simpson([calc.get_w_tail(float(t)) for t in grid], x=grid), moments[0], rtol=1e-8)


def test_wait_probability_equals_the_arrival_tail():
    """P(W > 0) must be exactly the probability an arrival finds two busy servers."""
    calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=1.6)
    assert np.isclose(calc.get_wait_prob(), sum(calc.get_pi(600)[2:]), atol=1e-9)


def test_the_exponential_rate_does_not_depend_on_the_single_server_rate():
    """
    The structural observation in the module docstring: ``gamma`` solves
    ``z = psi(2 mu (1-z))``, which never mentions ``mu_single``. So changing
    ``mu_single`` changes how OFTEN a customer waits but not the decay rate of
    how long. Stated as a test because it is the one claim here that is ours
    rather than the paper's.
    """
    rates, decays = [], []
    for mu_single in (0.4, 1.0, 1.6):
        calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=mu_single)
        rates.append(calc.get_wait_prob())
        decays.append(-math.log(calc.get_w_tail(5.0) / calc.get_wait_prob()) / 5.0)
    assert np.allclose(decays, decays[0], rtol=1e-10)  # same decay
    assert rates[0] > rates[1] > rates[2]  # a slower lone server makes waiting likelier


def test_little_law_and_the_service_time_split():
    """
    E[V] = E[Q] * alpha, and E[V] - E[W] is the mean time actually in service.
    When the rate does not depend on the state that must be exactly 1/mu -- a
    sharp check on the Little's-law path, which shares no code with get_w.
    """
    mean_a = 1.0 / ARRIVAL_RATE
    calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=MU)
    assert np.isclose(calc.get_v()[0], calc.get_q() * mean_a, rtol=1e-12)
    assert np.isclose(calc.get_service_time_mean(), 1.0 / MU, rtol=1e-6)


@pytest.mark.parametrize("mu_single", [0.5, 1.8])
def test_service_time_lies_between_the_two_rates(mu_single):
    """A customer is served partly alone and partly alongside, so E[S] is between."""
    calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=mu_single)
    lo, hi = sorted((1.0 / MU, 1.0 / mu_single))
    assert lo < calc.get_service_time_mean() < hi


def test_a_faster_lone_server_helps_every_metric():
    prev_q = prev_w = None
    for mu_single in (0.5, 1.0, 1.5, 2.5):
        calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=mu_single)
        if prev_q is not None:
            assert calc.get_q() < prev_q
            assert calc.get_w(1)[0] < prev_w
        prev_q, prev_w = calc.get_q(), calc.get_w(1)[0]


def test_run_reports_consistent_metrics():
    calc = _calc(ErlangParams(r=3, mu=3 * ARRIVAL_RATE), mu_single=1.6)
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w(1)[0])
    assert np.isclose(res.v[0], calc.get_q() / ARRIVAL_RATE, rtol=1e-9)
    assert 0.0 < res.utilization < 1.0
    assert len(res.pi) == len(res.p)


def test_invalid_parameters_are_rejected():
    calc = GiM2StateDependentCalc()
    with pytest.raises(ValueError, match="positive mean"):
        calc.set_sources([])
    calc.set_sources_params(_poisson())
    with pytest.raises(ValueError, match="mu must be positive"):
        calc.set_servers(mu=0.0)
    with pytest.raises(ValueError, match="mu_single must be positive"):
        calc.set_servers(mu=1.0, mu_single=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        _calc(_poisson(), mu_single=1.0).get_w(0)
    with pytest.raises(ValueError, match="non-negative"):
        _calc(_poisson(), mu_single=1.0).get_w_tail(-1.0)


def test_instability_is_decided_by_the_both_busy_rate_alone():
    """
    A huge ``mu_single`` cannot rescue an overloaded system: once the queue is
    long both servers are busy and the drain rate is ``2*mu`` regardless.
    """
    calc = _calc(_poisson(rate=2.5), mu=1.0, mu_single=50.0)
    with pytest.raises(ValueError, match="unstable"):
        calc.get_p()
