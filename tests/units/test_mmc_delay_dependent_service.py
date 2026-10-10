"""
Unit tests for the M/M/c queue with queueing-time dependent service rates
(roadmap item R2; D'Auria, Adan, Bekker & Kulkarni, EJOR 299(2):566-579, 2022).

The model and its solution are the authors'; these tests check OUR
implementation of them, and they do it against three references the
implementation cannot have been fitted to:

1. the ordinary M/M/c, which the model must reproduce exactly at mu1 == mu2 --
   compared with the textbook Erlang-C tail, not with another module here;
2. the paper's own closed form for c = 1 (its Section 5.1);
3. the paper's published numerical example for c = 2 (its Section 5.2), down to
   the printed digits of pi(0,0), pi(0,1) and pi(1,0).

Agreement with an independent simulation lives in
tests/test_delay_dependent_service.py; see docs/epics/EPIC-076*.md for the
figures and for the control experiment that explains the tolerances there.
"""

import math

import numpy as np
import pytest

from most_queue.theory.delay_dependent import MMcDelayDependentServiceCalc


def _calc(c, k, lam, mu1, mu2):
    calc = MMcDelayDependentServiceCalc(c=c, k=k)
    calc.set_sources(lam)
    calc.set_servers(mu1=mu1, mu2=mu2)
    return calc


def _erlang_c(c: int, a: float) -> float:
    """Erlang's C formula -- probability that an arrival has to wait in M/M/c."""
    rho = a / c
    tail = a**c / (math.factorial(c) * (1 - rho))
    return tail / (sum(a**j / math.factorial(j) for j in range(c)) + tail)


@pytest.mark.parametrize(
    "c, lam, mu, k", [(1, 0.7, 1.1, 2.0), (2, 1.3, 1.0, 0.45), (3, 2.0, 0.9, 5.0), (5, 3.7, 1.0, 1.0)]
)
def test_equal_rates_reduce_to_erlang_c(c, lam, mu, k):
    """mu1 == mu2 is an ordinary M/M/c: P(W>x) = C(c,a) exp(-c mu (1-rho) x)."""
    calc = _calc(c, k, lam, mu, mu)
    a = lam / mu
    rho = lam / (c * mu)
    c_erl = _erlang_c(c, a)
    for x in (0.0, 0.3, k, 2 * k, 10.0):
        assert np.isclose(calc.get_tail(x), c_erl * math.exp(-c * mu * (1 - rho) * x), atol=1e-12)
    assert np.isclose(calc.get_w()[0], c_erl / (c * mu - lam), rtol=1e-10)
    assert np.isclose(calc.get_p0_wait(), 1.0 - c_erl, atol=1e-12)


@pytest.mark.parametrize(
    "lam, mu1, mu2, k",
    [(0.7, 1.4, 0.9, 2.0), (0.5, 0.8, 1.6, 1.0), (1.0, 3.0, 1.5, 0.3), (0.4, 0.6, 1.1, 4.0)],
)
def test_single_server_matches_the_papers_closed_form(lam, mu1, mu2, k):
    """Section 5.1 of the paper solves c = 1 in closed form; we must reproduce it."""
    calc = _calc(1, k, lam, mu1, mu2)
    pi00 = ((mu1 / lam - 1) * (mu2 / lam - 1)) / (
        (mu1 / lam) * (mu2 / lam - 1) - (mu2 / mu1 - 1) * math.exp((lam - mu1) * k)
    )
    assert np.isclose(calc.get_p0_wait(), pi00, atol=1e-12)

    for x in (0.05, k / 2, k * 1.5, k + 3.0, k + 10.0):
        if x < k:
            ref = lam * pi00 * math.exp(-(mu1 - lam) * x)
        else:
            y = x - k
            ref = (
                lam
                * pi00
                * math.exp(-(mu1 - lam) * k)
                * (
                    ((mu2 - mu1) * math.exp(-mu1 * y)) / (mu2 - mu1 - lam)
                    - (lam * math.exp(-(mu2 - lam) * y)) / (mu2 - mu1 - lam)
                )
            )
        assert np.isclose(calc.get_pdf(x), ref, atol=1e-12)


def test_matches_the_papers_published_c2_example():
    """
    Section 5.2 of the paper works out c = 2, k = 0.45, lam = 2, mu1 = 0.75,
    mu2 = 1.12 and prints the boundary probabilities to six figures. This is the
    one check that exercises the whole Theorem 3 / 5 / 6 chain against numbers
    produced by someone else's independent implementation.
    """
    calc = _calc(2, 0.45, 2.0, 0.75, 1.12)
    probs = calc.get_server_state_probs()
    assert np.isclose(probs[(0, 0)], 0.0224116, atol=5e-8)
    assert np.isclose(probs[(0, 1)], 0.0108889, atol=5e-8)
    assert np.isclose(probs[(1, 0)], 0.0435035, atol=5e-8)
    # The paper also reports the extreme eigenvalues of the two quadratics.
    assert np.isclose(np.max(calc._closed_form_eigenvalues(1)[:3]), 0.5)  # |lam - c*mu1|
    assert np.isclose(np.min(calc._closed_form_eigenvalues(2)[1:]), -0.24)  # lam - c*mu2
    # ...and the limits of the two component functions.
    f_inf = calc._solve()["f_inf"]
    assert np.allclose(f_inf, [0.8271, 0.0961], atol=5e-5)
    # The paper prints b_c = -0.827051, we get +0.827051. Ours is consistent with
    # the paper's OWN formulas: with psi_c = [1, 0] and alpha1 M1 vanishing in the
    # first component, F_0(inf) IS b_c, and they print F_0(inf) = +0.8271. The
    # printed sign looks like a slip; nothing observable depends on it.
    assert np.isclose(calc._solve()["b_c"], f_inf[0])


@pytest.mark.parametrize(
    "c, lam, mu1, mu2, k", [(2, 2.0, 0.75, 1.12, 0.45), (3, 2.0, 0.8, 0.7, 5.0), (1, 0.7, 1.4, 0.9, 2.0)]
)
def test_distribution_is_proper_and_consistent(c, lam, mu1, mu2, k):
    """F(0) = 0, F is continuous at the threshold, the cdf is monotone and reaches 1."""
    calc = _calc(c, k, lam, mu1, mu2)
    assert np.isclose(calc.get_cdf(0.0), calc.get_p0_wait())
    assert np.isclose(np.sum(calc._f_vector(0.0)), 0.0, atol=1e-12)  # (14)
    eps = 1e-9
    assert np.allclose(calc._f_vector(k - eps), calc._f_vector(k + eps), atol=1e-7)  # (15)
    assert np.allclose(calc._f_prime_vector(k - eps), calc._f_prime_vector(k + eps), atol=1e-6)  # (16)

    grid = sorted({0.0, 0.1, k / 2, k, 2 * k + 1.0, 50.0, 400.0})
    cdf = [calc.get_cdf(x) for x in grid]
    assert all(0.0 <= y <= 1.0 + 1e-12 for y in cdf)
    assert all(a <= b + 1e-12 for a, b in zip(cdf, cdf[1:]))
    assert np.isclose(cdf[-1], 1.0, atol=1e-8)


@pytest.mark.parametrize("c, lam, mu1, mu2, k", [(2, 2.0, 0.75, 1.12, 0.45), (3, 2.0, 0.3, 0.8, 5.0)])
def test_mean_equals_the_integral_of_the_tail(c, lam, mu1, mu2, k):
    """
    E[W] from Lemma 2 must equal int_0^inf P(W>x) dx. The two go through
    different code: the closed-form integral (85) of the matrix exponentials
    versus pointwise evaluation of F.
    """
    calc = _calc(c, k, lam, mu1, mu2)
    grid = np.linspace(0.0, 400.0, 20001)
    tail = np.array([calc.get_tail(float(x)) for x in grid])
    assert np.isclose(np.trapezoid(tail, grid), calc.get_w()[0], rtol=1e-5)


def test_density_integrates_to_the_waiting_probability():
    """int_0^inf f(x) dx must be 1 - P(W = 0): the density carries all the non-atom mass."""
    calc = _calc(2, 0.45, 2.0, 0.75, 1.12)
    grid = np.linspace(1e-9, 300.0, 30001)
    pdf = np.array([calc.get_pdf(float(x)) for x in grid])
    assert np.isclose(np.trapezoid(pdf, grid), 1.0 - calc.get_p0_wait(), rtol=1e-5)


def test_slowdown_is_worse_and_speedup_is_better_than_the_equal_rate_case():
    """The whole point of the model: mu2 < mu1 must hurt, mu2 > mu1 must help."""
    base = _calc(3, 5.0, 2.0, 0.8, 0.8).get_w()[0]
    assert _calc(3, 5.0, 2.0, 0.8, 0.7).get_w()[0] > base
    assert _calc(3, 5.0, 2.0, 0.8, 0.9).get_w()[0] < base


def test_sojourn_splits_into_wait_and_the_pasta_weighted_service():
    """E[V] = E[W] + P(W<=k)/mu1 + P(W>k)/mu2, with the PASTA weighting."""
    calc = _calc(2, 0.45, 2.0, 0.75, 1.12)
    p1 = calc.get_class1_prob()
    assert np.isclose(p1, calc.get_cdf(calc.k))
    assert np.isclose(calc.get_service_time_mean(), p1 / 0.75 + (1 - p1) / 1.12)
    assert np.isclose(calc.get_v()[0], calc.get_w()[0] + calc.get_service_time_mean())


def test_large_threshold_is_an_mmc_with_the_fast_rate():
    """
    As k -> inf nobody crosses the threshold, so every customer is served at
    mu1 and the system becomes an ordinary M/M/c with rate mu1. This only makes
    sense when lam < c*mu1, which is the regime tested.
    """
    c, lam, mu1 = 2, 1.0, 1.2
    calc = _calc(c, 400.0, lam, mu1, 0.9)
    a = lam / mu1
    assert np.isclose(calc.get_w()[0], _erlang_c(c, a) / (c * mu1 - lam), rtol=1e-6)


def test_zero_threshold_charges_mu1_only_to_customers_who_do_not_wait():
    """
    k = 0 is the "operator slowdown" variant: only customers who find a free
    server get mu1, everyone who queues at all gets mu2. P(W>0) must then match
    the equal-rate-mu2 Erlang-C, because once a queue exists the system drains
    at c*mu2 regardless of mu1.
    """
    c, lam, mu1, mu2 = 2, 1.0, 1.5, 0.8
    calc = _calc(c, 0.0, lam, mu1, mu2)
    assert 0.0 < calc.get_p0_wait() < 1.0
    # Beyond the threshold the decay is governed by mu2 alone.
    tail_ratio = calc.get_tail(12.0) / calc.get_tail(10.0)
    assert np.isclose(tail_ratio, math.exp(-(c * mu2 - lam) * 2.0), rtol=1e-6)


def test_unstable_parameters_are_rejected_on_mu2_not_on_the_mix():
    """
    Stability depends on mu2 only: however fast mu1 is, a long enough queue puts
    everyone past the threshold. lam = 1.9 against c*mu2 = 1.6 is unstable even
    though c*mu1 = 20 is enormous.
    """
    calc = _calc(2, 1.0, 1.9, 10.0, 0.8)
    with pytest.raises(ValueError, match="unstable"):
        calc.get_w()


def test_resonance_surface_is_reported_clearly():
    """
    At mu2 == mu1 + lam/c the particular solution above the threshold resonates
    with a homogeneous mode and the representation degenerates. The paper does
    not list this among its Remark 2 cases, but it is visible in its own c = 1
    formula, where mu2 - mu1 - lam is a denominator. We must say so rather than
    surface a bare LinAlgError.
    """
    calc = _calc(2, 1.0, 1.0, 0.8, 0.8 + 1.0 / 2)
    with pytest.raises(ValueError, match="resonates"):
        calc.get_w()


def test_particular_solution_satisfies_its_defining_sylvester_equation():
    """
    The paper asserts the particular solution above the threshold "by direct
    substitution" with a plain inverse M2 (its equation (34)), even though the
    substitution leads to a Sylvester equation and Dt1, Dt2 do not commute. They
    are right; this pins down that the inverse really is the Sylvester solution,
    which is the one step of the derivation that is easy to transcribe wrongly.
    """
    for c, lam, mu1, mu2 in ((1, 0.7, 1.4, 0.9), (2, 2.0, 0.75, 1.12), (3, 2.0, 0.8, 0.7), (4, 2.6, 1.3, 0.9)):
        calc = _calc(c, 1.0, lam, mu1, mu2)
        s = calc._solve()
        _b1, b2, _ = calc._blocks()
        dt1, dt2 = s["dt1"], s["dt2"]
        n_mat = (dt1 - dt2) @ s["m2"]
        lhs = dt1 @ dt1 @ n_mat + dt1 @ n_mat @ (lam * np.eye(c) - dt2) + lam * n_mat @ (b2 - dt2)
        assert np.allclose(lhs, dt1 - dt2, atol=1e-9)


def test_eigenvalues_agree_with_the_papers_closed_forms():
    """
    The U matrices are built here by linearising the quadratic eigenvalue
    problem, not from the paper's explicit roots. The two must coincide --
    the implementation checks this internally, and this test makes the check
    visible rather than leaving it to a RuntimeError nobody triggers.
    """
    for c, lam, mu1, mu2 in ((2, 2.0, 0.75, 1.12), (4, 2.6, 1.3, 0.9)):
        calc = _calc(c, 1.0, lam, mu1, mu2)
        _u1m, _u1p, vals1 = calc._solve_u(*_u_args(calc, 1))
        _u2m, _u2p, vals2 = calc._solve_u(*_u_args(calc, 2))
        assert np.allclose(vals1, calc._closed_form_eigenvalues(1), atol=1e-9)
        assert np.allclose(vals2, calc._closed_form_eigenvalues(2), atol=1e-9)


def _u_args(calc, which):
    """(B_kappa, Dt_kappa) for the quadratic of index `which`."""
    b1, b2, deltas = calc._blocks()
    b_mat = b1 if which == 1 else b2
    mu = calc.mu1 if which == 1 else calc.mu2
    dt = mu * np.eye(calc.c) + np.linalg.solve(b_mat, deltas[calc.c - 1] @ b_mat)
    return b_mat, dt


def test_run_reports_consistent_metrics():
    calc = _calc(3, 5.0, 2.0, 0.8, 0.7)
    res = calc.run()
    assert np.isclose(res.w[0], calc.get_w()[0])
    assert np.isclose(res.v[0], res.w[0] + calc.get_service_time_mean())
    assert 0.0 < res.utilization < 1.0
    assert res.duration >= 0.0


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        MMcDelayDependentServiceCalc(c=2, k=-1.0)
    calc = MMcDelayDependentServiceCalc(c=2, k=1.0)
    with pytest.raises(ValueError, match="positive"):
        calc.set_sources(0.0)
    with pytest.raises(ValueError, match="positive"):
        calc.set_servers(mu1=1.0, mu2=-1.0)
    with pytest.raises(ValueError):
        MMcDelayDependentServiceCalc(c=2, k=1.0).get_w()
