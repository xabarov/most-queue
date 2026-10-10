"""
Unit tests for the higher conditional sojourn-time moments in M/G/1-PS
(roadmap item R3; Yashkov, arXiv:math/0512281, 2005).

The result is Yashkov's; these tests check OUR implementation of it, against
references that are independent of the code path under test:

1. the insensitivity theorem -- ``E[V(x)] = x/(1-rho)`` exactly, whatever the
   service-time shape. The implementation never uses this fact, it comes out of
   the Theorem 3.2 recursion, so reproducing it to machine precision is a
   strong check on the whole chain;
2. the M/M/1-PS variance in closed form, obtained by evaluating Yashkov's
   integral (3.10) by hand for exponential service;
3. his own small-job asymptotic ``Var[V(u)] ~ u^2 rho/(1-rho)^2`` as ``u -> 0``;
4. equation (3.10) itself, integrated numerically from the waiting-time
   distribution -- a different route to the second moment than the recursion;
5. Theorem 3.3 for ``K`` permanent jobs.

Agreement with simulation lives in tests/test_mg1_ps_sojourn_vs_sim.py.
"""

import math

import numpy as np
import pytest
import scipy.linalg as sla

from most_queue.random.utils.params import ErlangParams, GammaParams, H2Params
from most_queue.theory.fifo._ps_sojourn import equilibrium_ph, ph_mean, ph_representation, waiting_time_ph
from most_queue.theory.fifo.mg1_ps import MG1PSCalc

EXP1 = H2Params(mu1=1.0, mu2=9.0, p1=1.0)  # degenerate H2 == Exp(1)
ERL3 = ErlangParams(r=3, mu=3.0)  # mean 1, cv = 1/sqrt(3)
H2CV2 = H2Params(mu1=0.5, mu2=3.0, p1=0.4)  # mean 1, cv > 1
SHAPES = [("exp", EXP1), ("erlang", ERL3), ("h2", H2CV2)]


def _calc(lam, params):
    calc = MG1PSCalc()
    calc.set_sources(lam)
    calc.set_servers_params(params)
    return calc


@pytest.mark.parametrize("name, params", SHAPES)
@pytest.mark.parametrize("lam", [0.2, 0.6, 0.9])
def test_first_moment_is_the_insensitive_closed_form(name, params, lam):
    """
    E[V(x)] = x/(1-rho) for every service-time shape and every load.

    This is the strongest single check available: the recursion computes the
    first moment the same hard way it computes the rest, through xi_1 and the
    phase-type waiting time, and it has to land exactly on a value it was never
    told.
    """
    calc = _calc(lam, params)
    rho = lam * calc.b[0]
    for x in (0.01, 0.5, 2.0, 25.0):
        assert np.isclose(calc.get_conditional_sojourn_moments(x, 4)[0], x / (1 - rho), rtol=1e-11)
    assert name  # keep the parametrize id meaningful


@pytest.mark.parametrize("lam, mu", [(0.3, 1.0), (0.6, 1.0), (0.85, 1.2)])
def test_mm1_variance_matches_the_closed_form(lam, mu):
    """
    For exponential service, Yashkov's (3.10) integrates in closed form:
    ``Var = 2 rho/(1-rho)^2 [u/th - (1 - e^{-th u})/th^2]``, ``th = mu(1-rho)``.
    """
    calc = _calc(lam, H2Params(mu1=mu, mu2=9.0, p1=1.0))
    rho = lam / mu
    theta = mu * (1 - rho)
    for x in (0.05, 0.3, 1.0, 3.0, 10.0):
        expected = (2 * rho / (1 - rho) ** 2) * (x / theta - (1 - math.exp(-theta * x)) / theta**2)
        assert np.isclose(calc.get_conditional_sojourn_var(x), expected, rtol=1e-9)


@pytest.mark.parametrize("name, params", SHAPES)
def test_small_job_asymptotic_is_insensitive(name, params):
    """
    Yashkov's ``Var[V(u)] ~ u^2 rho/(1-rho)^2`` as ``u -> 0``. Note it carries no
    trace of the service-time shape -- for vanishingly small jobs the variance
    becomes insensitive too, which the three shapes here must all reproduce.
    """
    lam = 0.6
    calc = _calc(lam, params)
    rho = lam * calc.b[0]
    for x in (1e-3, 1e-4):
        assert np.isclose(calc.get_conditional_sojourn_var(x), x**2 * rho / (1 - rho) ** 2, rtol=5e-4)
    assert name


@pytest.mark.parametrize("name, params", SHAPES)
def test_variance_matches_equation_3_10_by_quadrature(name, params):
    """
    The recursion versus ``2/(1-rho)^2 int_0^u (u-y)(1-W(y))dy`` evaluated
    directly from the waiting-time distribution. Same theorem, different
    arithmetic: no moment recursion, no series, just one integral.
    """
    lam = 0.6
    calc = _calc(lam, params)
    rho = lam * calc.b[0]
    alpha, t_mat = ph_representation(params)
    pi_eq = equilibrium_ph(alpha, t_mat)
    s_mat = waiting_time_ph(pi_eq, t_mat, rho)
    ones = np.ones(len(pi_eq))

    for x in (0.3, 1.0, 3.0):
        grid = np.linspace(0.0, x, 4001)
        tail = np.array([rho * float(pi_eq @ sla.expm(s_mat * y) @ ones) for y in grid])
        quad = (2 / (1 - rho) ** 2) * np.trapezoid((x - grid) * tail, grid)
        assert np.isclose(calc.get_conditional_sojourn_var(x), quad, rtol=1e-5)
    assert name


def test_variance_grows_with_the_variability_of_the_service_time():
    """
    The mean is insensitive, the variance is not -- and it must be ordered by
    how variable the service time is. This is the whole point of the result:
    an Erlang workload and a hyperexponential one with the same mean give the
    same E[V(x)] but different risk.
    """
    lam = 0.6
    variances = [_calc(lam, p).get_conditional_sojourn_var(2.0) for _, p in SHAPES]
    assert variances[1] < variances[0] < variances[2]  # erlang < exp < h2


@pytest.mark.parametrize("permanent", [1, 2, 5])
def test_permanent_jobs_scale_mean_and_variance_linearly(permanent):
    """
    Theorem 3.3: ``E[V_K] = (K+1)u/(1-rho)`` and ``Var[V_K] = (K+1)Var[V]``.
    Both follow from the transform identity ``v_K = v^(K+1)``, i.e. the sojourn
    time with K permanent jobs is a sum of K+1 independent copies -- which the
    implementation realises as a power of the series, so getting both of these
    right is a check that the power is taken in the right place.
    """
    calc = _calc(0.6, ERL3)
    x = 2.0
    base = calc.get_conditional_sojourn_moments(x, 2)
    base_var = base[1] - base[0] ** 2
    with_perm = calc.get_conditional_sojourn_moments(x, 2, permanent_jobs=permanent)
    assert np.isclose(with_perm[0], (permanent + 1) * x / (1 - 0.6), rtol=1e-11)
    assert np.isclose(with_perm[1] - with_perm[0] ** 2, (permanent + 1) * base_var, rtol=1e-9)


def test_third_moment_is_consistent_with_a_sum_of_independent_copies():
    """
    The permanent-jobs identity gives a free check on the THIRD moment too,
    which none of the published formulas cover: for a sum of two i.i.d. copies,
    ``E[(X+Y)^3] = 2 E[X^3] + 6 E[X^2] E[X]``.
    """
    calc = _calc(0.6, H2CV2)
    x = 1.5
    m = calc.get_conditional_sojourn_moments(x, 3)
    m_perm = calc.get_conditional_sojourn_moments(x, 3, permanent_jobs=1)
    assert np.isclose(m_perm[2], 2 * m[2] + 6 * m[1] * m[0], rtol=1e-9)


def test_zero_load_limit_is_deterministic():
    """As rho -> 0 the job runs alone: V(x) -> x, so every central moment vanishes."""
    calc = _calc(1e-7, ERL3)
    assert np.isclose(calc.get_conditional_sojourn_moments(2.0, 2)[0], 2.0, rtol=1e-6)
    assert calc.get_conditional_sojourn_var(2.0) < 1e-6


def test_zero_size_job_leaves_immediately():
    calc = _calc(0.6, ERL3)
    assert calc.get_conditional_sojourn_moments(0.0, 3) == [0.0, 0.0, 0.0]


def test_unconditional_first_moment_recovers_the_closed_form():
    """
    ``int v_1(u) dB(u)`` must come back as ``b1/(1-rho)``. The conditional
    moments are exact, so any discrepancy here is quadrature error -- this is
    the accuracy check on the only numerical step in the class.
    """
    for _, params in SHAPES:
        calc = _calc(0.6, params)
        assert np.isclose(calc.get_v_moments(3)[0], calc.b[0] / (1 - 0.6 * calc.b[0]), rtol=1e-5)


def test_get_v_keeps_the_exact_first_moment_and_extends():
    calc = _calc(0.6, ERL3)
    assert calc.get_v(1) == [1.0 / (1 - 0.6)]
    extended = calc.get_v(3)
    assert np.isclose(extended[0], 1.0 / (1 - 0.6), rtol=1e-12)
    assert len(extended) == 3
    assert extended[1] > extended[0] ** 2  # positive variance


def test_moments_only_need_the_mean_unless_you_ask_for_more():
    """set_servers (moments only) must keep every insensitive result working."""
    calc = MG1PSCalc()
    calc.set_sources(0.6)
    calc.set_servers([1.0, 2.0, 6.0])
    assert np.isclose(calc.get_v()[0], 1.0 / 0.4)
    assert np.isclose(calc.get_conditional_sojourn_mean(2.0), 5.0)
    assert np.isclose(calc.get_p()[0], 0.4)
    with pytest.raises(ValueError, match="SHAPE"):
        calc.get_conditional_sojourn_var(2.0)


def test_fitting_from_moments_dispatches_on_cv():
    """cv <= 1 -> Erlang, cv > 1 -> H2, same convention as the rest of the library."""
    low_cv = MG1PSCalc()
    low_cv.set_sources(0.6)
    low_cv.set_servers_from_moments([1.0, 1.0 + 1.0 / 3.0])  # cv = 1/sqrt(3)
    assert isinstance(low_cv.service_params, ErlangParams)

    high_cv = MG1PSCalc()
    high_cv.set_sources(0.6)
    high_cv.set_servers_from_moments([1.0, 5.0, 60.0])  # cv = 2
    assert isinstance(high_cv.service_params, H2Params)

    with pytest.raises(ValueError, match="cv > 1"):
        high_cv.set_servers_from_moments([1.0, 5.0, 60.0], family="erlang")
    with pytest.raises(ValueError, match="cv < 1"):
        low_cv.set_servers_from_moments([1.0, 1.2, 2.0], family="h2")


def test_fitted_erlang_reproduces_the_explicit_erlang():
    """Fitting from the moments of Erlang(3,3) must give Erlang(3,3) back."""
    fitted = MG1PSCalc()
    fitted.set_sources(0.6)
    fitted.set_servers_from_moments([1.0, 4.0 / 3.0])
    direct = _calc(0.6, ERL3)
    assert np.allclose(
        fitted.get_conditional_sojourn_moments(2.0, 3), direct.get_conditional_sojourn_moments(2.0, 3), rtol=1e-9
    )


def test_integer_shape_gamma_is_accepted_and_non_integer_is_refused():
    calc = _calc(0.6, GammaParams(mu=3.0, alpha=3.0))
    assert np.allclose(
        calc.get_conditional_sojourn_moments(2.0, 3), _calc(0.6, ERL3).get_conditional_sojourn_moments(2.0, 3)
    )
    with pytest.raises(ValueError, match="not phase-type"):
        _calc(0.6, GammaParams(mu=3.0, alpha=2.5))


def test_waiting_time_representation_matches_pollaczek_khinchine():
    """
    The phase-type waiting time the recursion runs on must have the
    Pollaczek-Khinchine mean ``lam b2 / (2(1-rho))``. If this is wrong every
    moment above the first is wrong, and nothing else in the chain would say so.
    """
    lam = 0.6
    for _, params in SHAPES:
        alpha, t_mat = ph_representation(params)
        b1 = ph_mean(alpha, t_mat)
        rho = lam * b1
        inv = np.linalg.inv(-t_mat)
        b2 = 2 * float(alpha @ inv @ inv @ np.ones(len(alpha)))
        pi_eq = equilibrium_ph(alpha, t_mat)
        s_mat = waiting_time_ph(pi_eq, t_mat, rho)
        mean_ph = rho * float(pi_eq @ np.linalg.solve(-s_mat, np.ones(len(pi_eq))))
        assert np.isclose(mean_ph, lam * b2 / (2 * (1 - rho)), rtol=1e-12)


def test_invalid_inputs_are_rejected():
    calc = _calc(0.6, ERL3)
    with pytest.raises(ValueError, match="non-negative"):
        calc.get_conditional_sojourn_moments(-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        calc.get_conditional_sojourn_moments(1.0, 0)
    unstable = MG1PSCalc()
    unstable.set_sources(2.0)
    unstable.set_servers_params(ERL3)
    with pytest.raises(ValueError, match="unstable"):
        unstable.get_conditional_sojourn_var(1.0)
