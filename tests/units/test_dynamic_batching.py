"""
Unit tests for dynamic batching with an unbounded batch size (Inoue 2021).

The numerical route is an embedded chain, so the tests lean on things that chain
cannot arrange for itself: a closed form in the corner where the chain becomes
i.i.d., the paper's own identities, and the behaviour of the truncation.

Everything here is deliberately small. The chain is cheap at the parameters the
model is actually for -- a few hundred states, milliseconds -- and the one way
to make it expensive is a high-variance service law, which is exercised only
through the guard that refuses it.
"""

import numpy as np
import pytest

from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc

_ALPHA, _TAU0 = 0.0025, 0.02  # illustrative seconds, not a GPU calibration


def _calc(load=0.5, alpha=_ALPHA, tau0=_TAU0, **kwargs):
    calc = DynamicBatchingCalc(**kwargs)
    calc.set_sources(load / alpha if alpha > 0 else 1.0)
    calc.set_deterministic_service(alpha, tau0)
    return calc


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_arrival_rate_must_be_positive():
    with pytest.raises(ValueError, match="arrival rate"):
        DynamicBatchingCalc().set_sources(0.0)


def test_service_parameters_are_checked():
    calc = DynamicBatchingCalc().set_sources(10.0)
    with pytest.raises(ValueError, match="non-negative"):
        calc.set_deterministic_service(-1.0, 0.1)
    with pytest.raises(ValueError, match="zero time"):
        calc.set_deterministic_service(0.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        calc.set_erlang_service(0.1, 0.1, 0)


def test_unconfigured_use_is_refused():
    with pytest.raises(ValueError, match="set_sources"):
        DynamicBatchingCalc().get_waiting_mean()
    with pytest.raises(ValueError, match="service law"):
        DynamicBatchingCalc().set_sources(1.0).get_waiting_mean()


def test_construction_arguments_are_checked():
    with pytest.raises(ValueError, match="max_batch"):
        DynamicBatchingCalc(max_batch=0)
    with pytest.raises(ValueError, match="tolerance"):
        DynamicBatchingCalc(tolerance=0.0)


def test_instability_is_reported_with_the_right_criterion():
    """
    Stability is ``lambda * alpha < 1`` and does not involve ``tau0`` -- a
    growing batch amortises the fixed overhead away. The message says so,
    because the natural guess is that a large ``tau0`` destabilises things.
    """
    calc = DynamicBatchingCalc()
    calc.set_sources(500.0).set_deterministic_service(alpha=_ALPHA, tau0=_TAU0)
    assert calc.load == pytest.approx(1.25)
    with pytest.raises(ValueError, match="unstable"):
        calc.get_waiting_mean()

    # A ten-times larger fixed overhead, same alpha: still stable.
    stable = DynamicBatchingCalc()
    stable.set_sources(200.0).set_deterministic_service(alpha=_ALPHA, tau0=10 * _TAU0)
    assert stable.load == pytest.approx(0.5)
    assert stable.get_waiting_mean() > 0.0


def test_the_chain_grows_with_the_arrivals_during_the_fixed_overhead():
    """
    At fixed marginal load, more arrivals during the overhead also increase
    the required state space. Load and service variability matter as well.
    """
    sizes = []
    for tau0 in (0.02, 0.1, 0.4):
        calc = DynamicBatchingCalc()
        calc.set_sources(200.0).set_deterministic_service(alpha=_ALPHA, tau0=tau0)
        result = calc.run()
        sizes.append(result.max_batch)
        assert result.batch_mean > 200.0 * tau0  # at least the arrivals during tau0
    assert sizes[-1] > sizes[0]


# --------------------------------------------------------------------------
# the corner where the chain has a closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rate,tau0", [(5.0, 0.2), (20.0, 0.05), (2.0, 1.0)])
def test_batch_size_independent_service_matches_the_closed_form(rate, tau0):
    """
    With ``alpha = 0`` the processing time does not depend on the batch size,
    so the arrival count is Poisson(``lambda*tau0``) whatever the current
    batch: the embedded chain becomes i.i.d. and everything is elementary.

    With ``B = A + 1{A = 0}`` and ``m = lambda*tau0``,
    ``E[B] = m + e^-m`` and ``E[B^2] = m^2 + m + e^-m``.
    """
    calc = DynamicBatchingCalc()
    calc.set_sources(rate).set_deterministic_service(alpha=0.0, tau0=tau0)
    m = rate * tau0
    expected_mean = m + np.exp(-m)
    expected_second = m * m + m + np.exp(-m)
    mean, second = calc.get_batch_moments(2)
    assert mean == pytest.approx(expected_mean, rel=1e-10)
    assert second == pytest.approx(expected_second, rel=1e-10)

    expected_wait = (expected_second - expected_mean) / (2 * rate * expected_mean)
    assert calc.get_waiting_mean() == pytest.approx(expected_wait, rel=1e-10)
    assert calc.get_service_time_mean() == pytest.approx(tau0, rel=1e-12)
    assert calc.get_latency() == pytest.approx(expected_wait + tau0, rel=1e-10)


def test_a_vanishing_arrival_rate_leaves_batches_of_one():
    """Nobody ever arrives during a service, so every batch holds one request."""
    calc = DynamicBatchingCalc()
    calc.set_sources(1e-6).set_deterministic_service(alpha=0.0, tau0=0.01)
    assert calc.get_batch_moments(1)[0] == pytest.approx(1.0, abs=1e-6)
    assert calc.get_waiting_mean() == pytest.approx(0.0, abs=1e-6)
    assert calc.get_latency() == pytest.approx(0.01, abs=1e-6)


# --------------------------------------------------------------------------
# the paper's own identities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("load", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_lemma4_at_the_exact_idle_probability_reproduces_the_latency(load):
    """
    The paper's Lemma 4 is an identity given ``pi_0``, and Theorem 2 is that
    identity with ``pi_0`` bounded. Feeding Lemma 4 the exact idle probability
    must therefore return the exact latency -- which checks the chain and the
    paper's algebra against each other.
    """
    report = _calc(load).bound_report()
    assert report["lemma4_at_idle"] == pytest.approx(report["latency"], rel=1e-9)


@pytest.mark.parametrize("load", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_the_published_bounds_really_are_upper_bounds(load):
    report = _calc(load).bound_report()
    assert report["phi0"] >= report["latency"] - 1e-12
    assert report["phi1"] >= report["latency"] - 1e-12
    assert report["best_bound_ratio"] >= 1.0 - 1e-12


@pytest.mark.parametrize("load", [0.1, 0.3, 0.5, 0.9])
def test_the_idle_probability_respects_the_papers_lower_bound(load):
    """Lemma 5, which is the only thing standing between Lemma 4 and exactness."""
    report = _calc(load).bound_report()
    assert report["idle_probability"] >= report["idle_lower_bound"] - 1e-12
    assert 0.0 <= report["idle_probability"] <= 1.0


def test_which_bound_wins_follows_the_papers_own_criterion():
    """
    The paper states ``phi0 <= phi1`` exactly when ``lambda <= 1/(alpha+tau0)``.
    """
    for load in (0.05, 0.1, 0.2, 0.4, 0.8):
        calc = _calc(load)
        crossover = 1.0 / (_ALPHA + _TAU0)
        if calc.rate <= crossover:
            assert calc.bound_phi0() <= calc.bound_phi1() + 1e-12, load
        else:
            assert calc.bound_phi1() <= calc.bound_phi0() + 1e-12, load


def test_the_bounds_are_refused_for_service_laws_they_were_not_stated_for():
    """Theorem 2 is proved under Assumption 4: deterministic and affine."""
    calc = DynamicBatchingCalc()
    calc.set_sources(100.0).set_erlang_service(_ALPHA, _TAU0, shape=4)
    with pytest.raises(ValueError, match="Assumption 4"):
        calc.bound_phi0()
    with pytest.raises(ValueError, match="Assumption 4"):
        calc.bound_report()


def test_littles_law_links_the_latency_and_the_number_in_system():
    calc = _calc(0.6)
    assert calc.get_n_mean() == pytest.approx(calc.rate * calc.get_latency(), rel=1e-12)


def test_the_size_biased_service_mean_exceeds_the_plain_one():
    """
    A request is more likely to sit in a big batch simply because big batches
    hold more requests, so ``E[H-hat]`` must exceed the mean over batches.
    """
    calc = _calc(0.7)
    pi = calc.get_batch_distribution()
    sizes = np.arange(1, pi.size + 1)
    plain = float(pi @ (_ALPHA * sizes + _TAU0))
    assert calc.get_service_time_mean() > plain


# --------------------------------------------------------------------------
# the batch-size distribution
# --------------------------------------------------------------------------


def test_the_batch_distribution_is_a_distribution():
    pi = _calc(0.6).get_batch_distribution()
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi >= -1e-15)


def test_a_heavier_load_makes_bigger_batches_and_longer_waits():
    previous = None
    for load in (0.2, 0.4, 0.6, 0.8):
        calc = _calc(load)
        current = (calc.get_batch_moments(1)[0], calc.get_waiting_mean())
        if previous is not None:
            assert current[0] > previous[0], load  # batches grow
            assert current[1] > previous[1], load  # and so does the wait
        previous = current


def test_moment_order_is_validated():
    with pytest.raises(ValueError, match="num"):
        _calc().get_batch_moments(0)


# --------------------------------------------------------------------------
# truncation: the part that has to fail loudly rather than quietly
# --------------------------------------------------------------------------


def test_the_reported_truncation_error_is_small_when_it_converges():
    result = _calc(0.5).run()
    assert result.boundary_second_moment_fraction < 1e-9
    assert result.max_batch >= 1
    assert result.latency_mean == pytest.approx(result.waiting_mean + result.service_mean, rel=1e-12)


def test_a_pinned_truncation_is_honoured():
    calc = _calc(0.5, max_batch=200)
    assert calc.run().max_batch == 200


def test_a_high_variance_service_law_is_refused_rather_than_answered_badly():
    """
    An exponential batch time spreads the batch size out far enough that the
    chain would need a truncation it cannot afford. The honest outcome is a
    refusal that names the number, not a quietly wrong latency.
    """
    calc = DynamicBatchingCalc()
    calc.set_sources(240.0).set_exponential_service(_ALPHA, _TAU0)
    with pytest.raises((ValueError, MemoryError), match="converge|budget"):
        calc.get_latency()


def test_an_over_ambitious_truncation_is_priced_before_it_is_allocated():
    """The budget is checked up front, so the failure costs nothing."""
    calc = DynamicBatchingCalc(max_batch=20_000, max_nonzeros=100_000)
    calc.set_sources(240.0).set_exponential_service(_ALPHA, _TAU0)
    with pytest.raises(MemoryError, match="budget"):
        calc.get_latency()


def test_a_low_variance_erlang_still_converges():
    """Many phases means a tight batch-size distribution, which is cheap."""
    calc = DynamicBatchingCalc()
    calc.set_sources(240.0).set_erlang_service(_ALPHA, _TAU0, shape=50)
    assert calc.get_latency() > 0.0
    assert calc.run().boundary_second_moment_fraction < 1e-6


def test_more_service_phases_move_the_answer_toward_the_deterministic_one():
    """
    Erlang interpolates toward a deterministic time as the phases multiply, so
    the latency must approach the deterministic answer from above.
    """
    deterministic = _calc(0.6).get_latency()
    previous = None
    for shape in (10, 30, 90):
        calc = DynamicBatchingCalc()
        calc.set_sources(0.6 / _ALPHA).set_erlang_service(_ALPHA, _TAU0, shape)
        latency = calc.get_latency()
        assert latency > deterministic
        if previous is not None:
            assert latency < previous
        previous = latency


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def test_results_hang_together():
    result = _calc(0.7).run()
    assert result.utilization == pytest.approx(1.0 - result.idle_probability)
    assert result.jobs_mean == pytest.approx(result.latency_mean * (0.7 / _ALPHA), rel=1e-12)
    assert result.batch_distribution.size == result.max_batch
    assert result.duration >= 0.0


def test_repr_names_the_service_law():
    assert "deterministic" in repr(_calc())
