"""
Unit tests for exact M/M/1 with deadline-aware admission control
(most_queue.theory.admission_control.mm1_deadline_admission, EPIC-034).
"""

import numpy as np
import pytest

from most_queue.theory.admission_control.mm1_deadline_admission import MM1DeadlineAdmissionControlCalc
from most_queue.theory.fifo.mg1 import MG1Calc


def test_series_converges_with_more_terms():
    """Residual from doubling num_terms must be negligible (series truncation check)."""
    calc1 = MM1DeadlineAdmissionControlCalc(num_terms=80)
    calc1.set_sources(1.2)
    calc1.set_servers(1.0)
    calc1.set_deadline(0.5)
    res1 = calc1.run()

    calc2 = MM1DeadlineAdmissionControlCalc(num_terms=160)
    calc2.set_sources(1.2)
    calc2.set_servers(1.0)
    calc2.set_deadline(0.5)
    res2 = calc2.run()

    assert np.allclose(res1.v, res2.v, rtol=1e-8)
    assert np.isclose(res1.loss_prob, res2.loss_prob, rtol=1e-8)


def test_stricter_deadline_does_not_decrease_loss():
    """
    Larger theta (stricter/shorter deadline) must not decrease loss_prob.
    Note: loss_prob does NOT approach 1 as theta -> inf here -- arrivals
    that happen to land exactly while the system is idle (P(U=0)=pi0>0,
    an atom, not vanishing) are still admitted, since any D>0 exceeds
    U=0. Verified against an independent DES before relying on this in a
    test (0.545 theory vs 0.544 DES at theta=1000, not 1.0 -- an easy
    intuition trap, see docs/research/llm-serving-deadline-admission-control-2026.md).
    """
    prev_loss = None
    for theta in (0.5, 5.0, 50.0, 1000.0):
        calc = MM1DeadlineAdmissionControlCalc()
        calc.set_sources(1.2)
        calc.set_servers(1.0)
        calc.set_deadline(theta)
        loss = calc.get_loss_prob()
        if prev_loss is not None:
            assert loss >= prev_loss - 1e-9
        prev_loss = loss


def test_small_deadline_rate_theta_reduces_to_plain_mm1():
    """theta -> 0 (deadline almost never binds) -> loss_prob -> 0, behaves like plain M/M/1."""
    lam, mu = 0.6, 1.0
    calc = MM1DeadlineAdmissionControlCalc()
    calc.set_sources(lam)
    calc.set_servers(mu)
    calc.set_deadline(1e-4)
    res = calc.run()

    mg1 = MG1Calc()
    mg1.set_sources(lam)
    mg1.set_servers([1 / mu, 2 / mu**2, 6 / mu**3, 24 / mu**4])
    ref = mg1.run()

    assert res.loss_prob < 1e-3
    assert np.isclose(res.v[0], ref.v[0], rtol=1e-3)


def test_more_generous_deadline_reduces_loss_but_not_wait():
    """
    Larger deadline mean (smaller theta) must not increase loss_prob -- but
    E[W|admitted] moves the OTHER way (increases): a strict deadline only
    admits arrivals lucky enough to see a nearly-empty system (low wait); a
    lenient deadline admits a much less selective, higher-wait population
    too. Confirmed against independent DES before encoding this direction.
    """
    prev_loss = prev_w = None
    for theta in (2.0, 1.0, 0.5, 0.2):
        calc = MM1DeadlineAdmissionControlCalc()
        calc.set_sources(1.2)
        calc.set_servers(1.0)
        calc.set_deadline(theta)
        res = calc.run()
        if prev_loss is not None:
            assert res.loss_prob <= prev_loss + 1e-9
            assert res.w[0] >= prev_w - 1e-9
        prev_loss, prev_w = res.loss_prob, res.w[0]


def test_sojourn_equals_wait_plus_service():
    calc = MM1DeadlineAdmissionControlCalc()
    calc.set_sources(1.2)
    calc.set_servers(1.0)
    calc.set_deadline(0.5)
    res = calc.run()
    assert np.isclose(res.v[0], res.w[0] + 1.0 / calc.mu, atol=1e-9)


def test_u_moments_admitted_reduce_wait():
    """get_u_moments_admitted() must equal run()'s w exactly."""
    calc = MM1DeadlineAdmissionControlCalc()
    calc.set_sources(1.2)
    calc.set_servers(1.0)
    calc.set_deadline(0.5)
    res = calc.run()
    assert np.allclose(res.w, calc.get_u_moments_admitted())


def test_invalid_params_rejected():
    calc = MM1DeadlineAdmissionControlCalc()
    with pytest.raises(ValueError):
        calc.set_sources(-1.0)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers(0.0)
    calc.set_servers(1.0)
    with pytest.raises(ValueError):
        calc.set_deadline(-0.5)
    with pytest.raises(ValueError):
        MM1DeadlineAdmissionControlCalc(num_terms=1)


def test_run_before_deadline_set_raises():
    calc = MM1DeadlineAdmissionControlCalc()
    calc.set_sources(1.0)
    calc.set_servers(1.0)
    with pytest.raises(RuntimeError):
        calc.run()


if __name__ == "__main__":
    test_series_converges_with_more_terms()
    test_stricter_deadline_does_not_decrease_loss()
    test_small_deadline_rate_theta_reduces_to_plain_mm1()
    test_more_generous_deadline_reduces_loss_but_not_wait()
    test_sojourn_equals_wait_plus_service()
    test_u_moments_admitted_reduce_wait()
    test_invalid_params_rejected()
    test_run_before_deadline_set_raises()
    print("all admission-control tests passed")
