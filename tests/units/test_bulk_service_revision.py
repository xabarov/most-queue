"""Independent counterexamples from the single-server article audit.

Regression between implementations sharing the same tagged construction cannot
detect omitted abandonment behind the tag. These tests use a closed-form
per-state oracle and stationary rate conservation instead.
"""

# pylint: disable=missing-function-docstring,unnecessary-lambda-assignment

import math

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from examples.batch_service_optimal_bmax import minimal_b_max_for_sla
from examples.batch_service_sla_exact_tail_comparison import inoue_phi
from most_queue.theory.batch._idle_refill import abandonment_chain
from most_queue.theory.batch._tagged_batch import TaggedBatchWait
from most_queue.theory.batch._tagged_boundary import behind_cap, behind_error
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc, _h2_abandonment_chain


@pytest.mark.parametrize("order", range(5))
def test_behind_bound_at_zero_has_independent_exponential_formula(order):
    lam, gamma = 1.7, 0.6
    expected = math.factorial(order) * (gamma ** (-order) - gamma / (gamma + lam) ** (order + 1))
    assert behind_error(lam, gamma, 0, order)[order] == pytest.approx(expected, rel=2e-14)


def test_adaptive_behind_cap_and_exact_reductions():
    cap = behind_cap(3, 1.0, 0.25)
    assert behind_error(1.0, 0.25, cap)[0] <= 1e-13
    assert behind_error(1.0, 0.25, cap - 1)[0] > 1e-13
    assert behind_cap(1, 1.0, 0.25) == 0
    assert behind_cap(3, 1.0, 0.0) == 2


@pytest.mark.parametrize("family", ["erlang", "h2"])
def test_abandoning_behind_customers_closed_form_counterexample(family):
    # R=0, observed Exp(1), lambda=gamma=1, a=b=2. At time t, the
    # surviving behind count is Poisson(1-exp(-t)), not Poisson(t).
    # Integrating over service completion and tag survival gives
    # P(dispatch) = integral exp(-2t)*(1-.5*exp(-(1-exp(-t)))) dt.
    expected = 0.5 - 0.5 / math.e
    if family == "erlang":
        sub, size, start, serve = abandonment_chain(1.0, 1, 1.0, 1, 0, 2, 2, 1.0, 1.0)
    else:
        sub, size, start, serve = _h2_abandonment_chain(1.0, 1.0, 2.0, 1.0, 0, 2, 2, 1.0, 1.0)
    assert spsolve(-sub, serve)[start] == pytest.approx(expected, abs=2e-13)
    assert np.max(np.abs(sub @ np.ones(size) + serve + 1)) < 1e-12


@pytest.mark.parametrize("cap", [1, 2, 4, 8])
def test_dispatch_cap_error_is_enclosed_by_independent_hitting_bound(cap):
    sub, _, start, serve = abandonment_chain(1.0, 1, 1.0, 1, 0, 2, 2, 1.0, 1.0, behind_limit=cap)
    got = float(spsolve(-sub, serve)[start])
    expected = 0.5 - 0.5 / math.e
    assert got <= expected + 1e-14
    assert expected - got <= behind_error(1.0, 1.0, cap)[0] + 1e-14


@pytest.mark.parametrize("family", ["exp", "erlang", "h2"])
def test_tagged_abandonment_agrees_with_stationary_customer_flow(family):
    gamma, n = 0.5, 35
    if family == "exp":
        calc = BulkServiceMM1Calc(2, 4, n, gamma=gamma)
        calc.set_servers(0.8)
    elif family == "erlang":
        calc = BulkServiceErlangCalc(2, 4, 2, n, gamma=gamma)
        calc.set_servers(1.6)
    else:
        calc = BulkServiceH2Calc(2, 4, n, gamma=gamma)
        calc.set_servers(0.3, 0.6, 1.5)
    calc.set_sources(1.0)
    qmass = calc._solve_pi().reshape(-1, n + 1).sum(axis=0)
    assert qmass[-1] < 1e-15
    assert calc.get_abandonment_prob() == pytest.approx(gamma * (qmass @ np.arange(n + 1)), abs=3e-12)


def test_h2_mean_wait_uses_customer_not_busy_time_service_weights():
    calc = BulkServiceH2Calc(1, 8, 600)
    calc.set_sources(0.8)
    calc.set_servers(0.25, lambda s: 0.25 / (0.1438 * s + 1.8874), lambda s: 3 / (0.1438 * s + 1.8874))
    pi = calc._solve_pi()
    qmass = pi.reshape(17, 601).sum(axis=0)
    assert qmass[-1] < 1e-10
    customer_service = 0.0
    departures = 0.0
    for batch in range(1, 9):
        batch_flow = sum(
            rate * sum(pi[calc._busy_index(batch, phase, j)] for j in range(601))
            for phase, rate in enumerate((calc.mu1_fn(batch), calc.mu2_fn(batch)))
        )
        service_mean = 0.25 / calc.mu1_fn(batch) + 0.75 / calc.mu2_fn(batch)
        customer_service += batch * batch_flow * service_mean / calc.l
        departures += batch * batch_flow
    result = calc.run()
    assert departures == pytest.approx(calc.l, abs=1e-9)
    assert result.w[0] == pytest.approx(result.v[0] - customer_service, abs=1e-8)


def test_inoue_low_load_uses_published_denominator():
    assert inoue_phi(0.05, 0.1438, 1.8874) == pytest.approx(2.1358329606676785)
    assert inoue_phi(0.8, 0.1438, 1.8874) == pytest.approx(3.3532385550504302)


def test_minimum_candidate_does_not_require_all_larger_candidates_to_pass():
    tails = {2: {5: 0.2}, 3: {5: 0.01}, 4: {5: 0.3}, 5: {5: 0.005}}
    assert minimal_b_max_for_sla(tails, 5, 0.02) == 3


@pytest.mark.parametrize("family", ["exp", "erlang", "h2"])
@pytest.mark.parametrize("gamma", [0.0, 0.2])
def test_aggregated_chain_matches_per_state_oracle_and_invalidates(family, gamma):
    if family == "exp":
        calc = BulkServiceMM1Calc(2, 3, 22, gamma)
        configure = lambda c, scale: c.set_servers(scale)
    elif family == "erlang":
        calc = BulkServiceErlangCalc(2, 3, 2, 22, gamma)
        configure = lambda c, scale: c.set_servers(2 * scale)
    else:
        calc = BulkServiceH2Calc(2, 3, 22, gamma)
        configure = lambda c, scale: c.set_servers(0.3, scale, 2 * scale)
    calc.set_sources(0.6)
    configure(calc, 1.0)
    chain = calc.get_tagged_wait()
    assert chain is calc.get_tagged_wait()
    assert chain.served + chain.abandoned + chain.omitted_mass == pytest.approx(1, abs=1e-12)
    assert chain.tail(0) == pytest.approx(1 - chain.atom / chain.served)
    assert chain.tail(-1) == 1
    if gamma == 0:
        assert chain.tail(1.3) == pytest.approx(calc.get_tail(1.3), abs=3e-12)
        if family != "h2":
            assert chain.moments(2) == pytest.approx(calc.get_w(2), abs=1e-10)
    before = chain.moments(1)[0]
    configure(calc, 1.7)
    faster = calc.get_tagged_wait()
    assert faster is not chain
    assert faster.moments(1)[0] != pytest.approx(before)
    calc.set_sources(0.9)
    assert calc.get_tagged_wait() is not faster


def test_aggregated_chain_per_state_h2_positive_patience():
    calc = BulkServiceH2Calc(2, 3, 16, gamma=0.4)
    calc.set_sources(0.7)
    calc.set_servers(0.4, lambda s: 0.6 + s * 0.1, 1.6)
    chain = calc.get_tagged_wait()
    pi = calc._solve_pi()
    oracle = pi[1]
    for j in range(17):
        if j == 0:
            sub, _, start, rates = calc._abandon_chain_for_state(0, j, True)
            oracle += pi[j] * spsolve(-sub, rates)[start]
        for batch in range(2, 4):
            for phase, rate in enumerate((calc.mu1_fn(batch), calc.mu2_fn(batch))):
                sub, _, start, rates = calc._abandon_chain_for_state(rate, j)
                oracle += pi[calc._busy_index(batch, phase, j)] * spsolve(-sub, rates)[start]
    assert chain.served == pytest.approx(oracle, abs=2e-12)


def test_tagged_budget_and_bounds_are_explicit():
    calc = BulkServiceMM1Calc(2, 3, 40, gamma=0.2)
    calc.set_sources(0.5)
    calc.set_servers(1.0)
    with pytest.raises(ValueError, match="budget"):
        TaggedBatchWait(calc, max_states=2)
    chain = TaggedBatchWait(calc)
    assert chain.moment_error_bounds(2)[0] <= 2e-13
    assert sum(chain.outcomes(1)[k] for k in ("on_time", "late", "abandoned", "omitted_mass")) == pytest.approx(1)


@pytest.mark.parametrize("args", [(0, 1, 1), (2, -1, 1), (2, 1, -1), (2, np.nan, 1), (2, 1, np.inf)])
def test_behind_cap_rejects_invalid_parameters(args):
    with pytest.raises(ValueError):
        behind_cap(*args)


def test_integer_patience_does_not_truncate_noninteger_transition_rates():
    first = BulkServiceMM1Calc(2, 3, 20, gamma=1)
    second = BulkServiceMM1Calc(2, 3, 20, gamma=1.0)
    for calc in (first, second):
        calc.set_sources(0.7)
        calc.set_servers(0.8)
    left, right = first.get_tagged_wait(), second.get_tagged_wait()
    assert left.abandoned == pytest.approx(right.abandoned, abs=1e-14)
    assert left.moments(2) == pytest.approx(right.moments(2), abs=1e-14)
    assert np.max(np.abs(left.sub @ np.ones(len(left.h)) + left.serve_rate + 1)) < 1e-12
