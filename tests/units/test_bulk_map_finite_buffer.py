"""
Unit tests for the exact MAP/PH^(a,b)/1/N bulk-service queue.

The construction is a tagged-customer absorbing chain, so the tests lean on
things that chain cannot arrange for itself: closed forms in the degenerate
corners, Little's law, the identity between the CDF and the moments, and
agreement with a completely separate implementation of the Poisson case that
already lives in this library.
"""

import math

import numpy as np
import pytest

from most_queue.random.map_ph import MAPParams, PHParams
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.map_ph_finite_buffer import BulkServiceMapPhCalc

_TOL = 1e-9


def _poisson(a=3, b=6, capacity=40, rate=6.7, mu=1.7):
    calc = BulkServiceMapPhCalc(a=a, b=b, capacity=capacity)
    calc.set_poisson_sources(rate).set_exponential_servers(mu)
    return calc


def _bursty_map():
    """A genuinely correlated two-phase MAP: a fast burst state and a quiet one."""
    return MAPParams(
        D0=np.array([[-9.0, 0.6], [0.4, -1.4]]),
        D1=np.array([[8.0, 0.4], [0.2, 0.8]]),
    )


def _two_phase_ph():
    return PHParams(alpha=np.array([0.3, 0.7]), T=np.array([[-2.0, 1.0], [0.0, -5.0]]))


# --------------------------------------------------------------------------
# configuration and validation
# --------------------------------------------------------------------------


def test_batch_rule_must_make_sense():
    with pytest.raises(ValueError, match="a must be at least 1"):
        BulkServiceMapPhCalc(a=0, b=3, capacity=10)
    with pytest.raises(ValueError, match="b must be at least a"):
        BulkServiceMapPhCalc(a=5, b=3, capacity=10)


def test_buffer_must_hold_a_full_batch_trigger():
    """With N < a the server could never accumulate a batch at all."""
    with pytest.raises(ValueError, match="capacity"):
        BulkServiceMapPhCalc(a=5, b=5, capacity=4)


def test_map_matrices_are_checked():
    calc = BulkServiceMapPhCalc(a=1, b=1, capacity=5)
    with pytest.raises(ValueError, match="square and of equal shape"):
        calc.set_sources(MAPParams(D0=np.zeros((2, 2)), D1=np.zeros((3, 3))))
    with pytest.raises(ValueError, match="non-negative"):
        calc.set_sources(MAPParams(D0=np.array([[-1.0]]), D1=np.array([[-1.0]])))
    with pytest.raises(ValueError, match="sum to zero"):
        calc.set_sources(MAPParams(D0=np.array([[-1.0]]), D1=np.array([[2.0]])))


def test_service_law_is_checked():
    calc = BulkServiceMapPhCalc(a=1, b=1, capacity=5)
    with pytest.raises(ValueError, match="does not match"):
        calc.set_servers(PHParams(alpha=np.array([1.0]), T=np.eye(2) * -1))
    with pytest.raises(ValueError, match="probability vector"):
        calc.set_servers(PHParams(alpha=np.array([0.5]), T=np.array([[-1.0]])))
    with pytest.raises(ValueError, match="never completes"):
        calc.set_servers(PHParams(alpha=np.array([1.0]), T=np.array([[0.0]])))


def test_rates_must_be_positive():
    calc = BulkServiceMapPhCalc(a=1, b=1, capacity=5)
    with pytest.raises(ValueError):
        calc.set_poisson_sources(0.0)
    with pytest.raises(ValueError):
        calc.set_exponential_servers(-1.0)


def test_using_the_calculator_before_configuring_it_is_refused():
    with pytest.raises(ValueError, match="set_sources"):
        BulkServiceMapPhCalc(a=1, b=1, capacity=5).get_w(1)
    calc = BulkServiceMapPhCalc(a=1, b=1, capacity=5).set_poisson_sources(1.0)
    with pytest.raises(ValueError, match="set_servers"):
        calc.get_w(1)


def test_moment_order_must_be_positive():
    with pytest.raises(ValueError, match="num"):
        _poisson().get_w(0)


def test_negative_times_are_refused():
    calc = _poisson()
    with pytest.raises(ValueError, match="non-negative"):
        calc.get_w_cdf(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        calc.get_v_cdf(-1.0)


# --------------------------------------------------------------------------
# degenerate corners with known answers
# --------------------------------------------------------------------------


def test_single_phase_map_is_exactly_poisson():
    """A one-phase MAP IS a Poisson process, so the two routes must coincide."""
    poisson = _poisson(capacity=60)
    as_map = BulkServiceMapPhCalc(a=3, b=6, capacity=60)
    as_map.set_sources(MAPParams(D0=np.array([[-6.7]]), D1=np.array([[6.7]])))
    as_map.set_exponential_servers(1.7)
    assert as_map.get_w(3) == pytest.approx(poisson.get_w(3), rel=1e-12)
    assert as_map.get_w_cdf(1.5) == pytest.approx(poisson.get_w_cdf(1.5), rel=1e-12)


@pytest.mark.parametrize("rate,mu", [(2.0, 3.0), (1.0, 4.0), (5.0, 6.0)])
def test_unit_batches_reduce_to_m_m_1(rate, mu):
    """
    With a = b = 1 the bulk rule is no rule at all: serve one at a time, start
    as soon as anyone is there. With a buffer wide enough to never bind, that
    is plain M/M/1.
    """
    calc = BulkServiceMapPhCalc(a=1, b=1, capacity=400)
    calc.set_poisson_sources(rate).set_exponential_servers(mu)
    assert calc.get_w(1)[0] == pytest.approx(rate / (mu * (mu - rate)), rel=1e-9)
    assert calc.get_v(1)[0] == pytest.approx(1.0 / (mu - rate), rel=1e-9)
    assert calc.run().waiting_atom == pytest.approx(1.0 - rate / mu, rel=1e-9)


def test_an_arrival_that_completes_the_batch_waits_no_time():
    """
    The atom at zero is exactly the chance of arriving to an idle server with
    a - 1 others already waiting.
    """
    calc = _poisson(a=3, b=6, capacity=30)
    probs = calc.get_p()
    # The waiting time is conditional on being admitted, so the atom is that
    # probability renormalised by the customers who actually get in.
    expected = probs["idle"][calc.a - 1] / (1.0 - calc.get_loss_probability())
    assert calc.run().waiting_atom == pytest.approx(expected, rel=1e-9)
    assert calc.get_w_cdf(0.0) == pytest.approx(expected, rel=1e-9)


def test_sojourn_cdf_starts_at_zero():
    """Nobody leaves instantly: the sojourn time always contains a service."""
    assert _poisson().get_v_cdf(0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# structural identities
# --------------------------------------------------------------------------


def test_generator_rows_sum_to_zero_and_the_stationary_law_is_a_distribution():
    calc = BulkServiceMapPhCalc(a=2, b=5, capacity=12)
    calc.set_sources(_bursty_map()).set_servers(_two_phase_ph())
    gen = calc._build_generator()
    assert np.max(np.abs(gen.sum(axis=1))) < 1e-10
    pi = calc._stationary()
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= -_TOL)
    assert np.max(np.abs(pi @ gen)) < 1e-9


def test_tagged_chain_is_a_proper_sub_generator():
    """
    Rows may lose mass -- that is absorption -- but never gain it, and the
    off-diagonal entries stay non-negative.
    """
    calc = BulkServiceMapPhCalc(a=3, b=5, capacity=14)
    calc.set_sources(_bursty_map()).set_servers(_two_phase_ph())
    sub = calc._build_tagged()
    off = sub - np.diag(np.diag(sub))
    assert np.all(off >= -_TOL)
    assert np.all(np.diag(sub) <= _TOL)
    assert np.all(sub.sum(axis=1) <= 1e-10)
    # absorption has to be possible from somewhere, else nobody is ever served
    assert np.min(sub.sum(axis=1)) < -_TOL


def test_state_probabilities_form_a_distribution():
    calc = BulkServiceMapPhCalc(a=3, b=6, capacity=20)
    calc.set_sources(_bursty_map()).set_servers(_two_phase_ph())
    probs = calc.get_p()
    assert len(probs["idle"]) == calc.a
    assert len(probs["busy"]) == calc.capacity + 1
    assert sum(probs["idle"]) + sum(probs["busy"]) == pytest.approx(1.0)
    assert all(p >= -_TOL for p in probs["idle"] + probs["busy"])


@pytest.mark.parametrize("a,b,cap,rate,mu", [(3, 6, 30, 6.7, 1.7), (2, 5, 15, 4.0, 1.0), (5, 5, 20, 6.0, 1.6)])
def test_little_law_holds(a, b, cap, rate, mu):
    """``Lq = lambda' E(W)`` with the EFFECTIVE rate, blocking taken out."""
    calc = BulkServiceMapPhCalc(a=a, b=b, capacity=cap)
    calc.set_poisson_sources(rate).set_exponential_servers(mu)
    res = calc.run(num=1)
    assert res.queue_mean == pytest.approx(res.effective_rate * res.w[0], rel=1e-9)


def test_little_law_holds_for_a_correlated_map_too():
    calc = BulkServiceMapPhCalc(a=3, b=8, capacity=25)
    calc.set_sources(_bursty_map()).set_servers(_two_phase_ph())
    res = calc.run(num=1)
    assert res.queue_mean == pytest.approx(res.effective_rate * res.w[0], rel=1e-9)


def test_loss_probability_is_the_full_buffer_probability_under_pasta():
    calc = BulkServiceMapPhCalc(a=2, b=4, capacity=8)
    calc.set_poisson_sources(5.0).set_exponential_servers(1.0)
    assert calc.get_loss_probability() == pytest.approx(calc.get_p()["busy"][-1], rel=1e-10)


def test_sojourn_is_the_wait_plus_one_whole_batch_service():
    calc = _poisson(capacity=30)
    waiting = calc.get_w(2)
    service = calc.get_service_moments(2)
    sojourn = calc.get_v(2)
    assert sojourn[0] == pytest.approx(waiting[0] + service[0], rel=1e-12)
    second = waiting[1] + 2 * waiting[0] * service[0] + service[1]
    assert sojourn[1] == pytest.approx(second, rel=1e-12)


def test_service_moments_match_the_phase_type_formula():
    ph = _two_phase_ph()
    calc = BulkServiceMapPhCalc(a=1, b=1, capacity=5)
    calc.set_poisson_sources(1.0).set_servers(ph)
    alpha, mat = np.asarray(ph.alpha), np.asarray(ph.T)
    expected = []
    for k in range(1, 4):
        expected.append(float(math.factorial(k) * alpha @ np.linalg.matrix_power(np.linalg.inv(-mat), k) @ np.ones(2)))
    assert calc.get_service_moments(3) == pytest.approx(expected, rel=1e-10)


# --------------------------------------------------------------------------
# the CDF against the moments, and against a second implementation
# --------------------------------------------------------------------------


def test_cdf_is_a_distribution_function():
    calc = _poisson(capacity=30)
    times = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0])
    values = calc.get_w_cdf(times)
    assert np.all(np.diff(values) >= -1e-12)
    assert np.all((values >= -_TOL) & (values <= 1.0 + _TOL))
    assert values[-1] == pytest.approx(1.0, abs=1e-6)
    assert calc.get_w_tail(1.0) == pytest.approx(1.0 - calc.get_w_cdf(1.0))


def test_array_and_scalar_calls_agree():
    calc = _poisson(capacity=25)
    times = [0.3, 1.1, 2.7]
    assert calc.get_w_cdf(np.array(times)) == pytest.approx([calc.get_w_cdf(t) for t in times])
    assert calc.get_v_cdf(np.array(times)) == pytest.approx([calc.get_v_cdf(t) for t in times])


def test_moments_agree_with_the_integrated_tail():
    """``E(W) = int P(W > t) dt`` and ``E(W^2) = 2 int t P(W > t) dt``."""
    calc = _poisson(capacity=30)
    times = np.linspace(0.0, 40.0, 1601)
    tail = np.asarray(calc.get_w_tail(times))
    moments = calc.get_w(2)
    assert np.trapezoid(tail, times) == pytest.approx(moments[0], rel=2e-3)
    assert 2 * np.trapezoid(times * tail, times) == pytest.approx(moments[1], rel=2e-3)


def test_matches_the_libraries_own_infinite_buffer_solver():
    """
    A wide buffer is effectively no buffer limit, and
    :class:`BulkServiceMM1Calc` reaches the same quantity by an entirely
    different construction, so the two must land on the same numbers.
    """
    exact = BulkServiceMapPhCalc(a=3, b=6, capacity=200)
    exact.set_poisson_sources(6.7).set_exponential_servers(1.7)
    reference = BulkServiceMM1Calc(a=3, b=6, queue_truncation=400)
    reference.set_sources(6.7)
    reference.set_servers(1.7)
    assert exact.get_w(2) == pytest.approx(reference.get_w(2), rel=1e-6)
    assert exact.get_loss_probability() < 1e-9


def test_a_tighter_buffer_shortens_the_wait_and_raises_the_loss():
    previous_wait, previous_loss = None, None
    for capacity in (6, 10, 20, 50):
        calc = BulkServiceMapPhCalc(a=3, b=6, capacity=capacity)
        calc.set_poisson_sources(10.0).set_exponential_servers(1.5)
        wait = calc.get_w(1)[0]
        loss = calc.get_loss_probability()
        if previous_wait is not None:
            assert wait > previous_wait  # more room to queue means longer queues
            assert loss < previous_loss
        previous_wait, previous_loss = wait, loss


# --------------------------------------------------------------------------
# what correlated arrivals change
# --------------------------------------------------------------------------


def test_arrivals_do_not_see_the_time_stationary_state_under_a_map():
    """
    PASTA is a property of Poisson input, not of queues. Under a bursty MAP an
    arrival is far more likely to land while the server is already busy, and
    taking the time-stationary law instead would be a silent modelling error
    rather than an approximation.
    """
    calc = BulkServiceMapPhCalc(a=3, b=8, capacity=25)
    calc.set_sources(_bursty_map()).set_servers(_two_phase_ph())
    pi = calc._stationary()
    seen = calc._arrival_epoch_weights().sum(axis=1)
    assert 0.5 * np.abs(pi - seen).sum() > 0.1  # the two laws are far apart

    busy_in_time = 1.0 - sum(calc.get_p()["idle"])
    m_a, m_s = calc._dims
    busy_on_arrival = sum(
        seen[calc._busy_index(n, j, m)] for n in range(calc.capacity + 1) for j in range(m_s) for m in range(m_a)
    )
    assert busy_on_arrival > busy_in_time + 0.05


def test_poisson_input_does_satisfy_pasta():
    """The same check the other way round, as a control."""
    calc = _poisson(capacity=20)
    pi = calc._stationary()
    seen = calc._arrival_epoch_weights().sum(axis=1)
    assert np.allclose(pi, seen, atol=1e-12)


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def test_results_are_self_consistent():
    calc = BulkServiceMapPhCalc(a=3, b=8, capacity=25)
    calc.set_sources(_bursty_map()).set_servers(_two_phase_ph())
    res = calc.run(num=2)
    assert res.states == calc._system_size()
    assert res.tagged_states > 0
    assert 0.0 <= res.loss_probability <= 1.0
    assert res.effective_rate == pytest.approx(calc.get_arrival_rate() * (1 - res.loss_probability))
    assert res.v[0] == pytest.approx(res.w[0] + calc.get_service_moments(1)[0], rel=1e-12)
    assert res.utilization == pytest.approx(res.server_busy_probability)
    assert res.duration >= 0.0


def test_reconfiguring_invalidates_the_cached_chain():
    calc = _poisson(capacity=20, rate=6.7)
    first = calc.get_w(1)[0]
    calc.set_poisson_sources(3.0)
    second = calc.get_w(1)[0]
    assert second < first  # a lighter load cannot wait longer
    calc.set_poisson_sources(6.7)
    assert calc.get_w(1)[0] == pytest.approx(first, rel=1e-12)


def test_repr_mentions_the_rule():
    assert "a=3" in repr(_poisson())
