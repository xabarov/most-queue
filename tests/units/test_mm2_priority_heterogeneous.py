"""
Unit tests for M/M/2 with two preemptive priority classes and heterogeneous
servers (most_queue.theory.priority.preemptive.mm2_heterogeneous), EPIC-025.
"""

import numpy as np
import pytest

from most_queue.theory.priority.preemptive.mm2_heterogeneous import (
    MM2PriorityHeterogeneousCalc,
    _canonicalize,
    _class0_arrival,
    _class1_arrival,
    _departure,
    _servers_from_state,
)
from most_queue.theory.priority.preemptive.mmk_prty_exact import MMkPriorityExact


# ------------------------------------------------------------------ primitives
def test_canonicalize_fills_idle_from_backlog():
    # n0=2 but only A is currently marked class-0 and B idle -> B must be filled with class 0
    assert _canonicalize(2, 0, 0, None) == (2, 0, None)
    # n0=0, n1=2, both idle -> both filled with class 1
    assert _canonicalize(0, 2, None, None) == (0, 2, None)


def test_class0_arrival_prefers_idle_then_preempts_fastest():
    # empty system: class-0 arrival takes A
    assert _class0_arrival(0, 0, None, None) == (1, 0, "A")
    # (n0=0,n1=1,config=B): class1 on B, A idle -> class0 takes idle A
    a, b = _servers_from_state(0, 1, "B")
    assert _class0_arrival(0, 1, a, b) == (1, 1, "A")
    # (n0=0,n1=1,config=A): class1 on A, B idle -> class0 takes idle B
    a, b = _servers_from_state(0, 1, "A")
    assert _class0_arrival(0, 1, a, b) == (1, 1, "B")
    # (n0=0,n1=2): both busy with class1 -> class0 must preempt A (fastest)
    a, b = _servers_from_state(0, 2, None)
    assert _class0_arrival(0, 2, a, b) == (1, 2, "A")


def test_class1_arrival_never_preempts():
    # (n0=1,n1=0,config=A): class0 on A, B idle -> class1 takes idle B
    a, b = _servers_from_state(1, 0, "A")
    assert _class1_arrival(1, 0, a, b) == (1, 1, "A")
    # (n0=2,n1=0): both busy with class0 -> class1 just queues, config unaffected
    a, b = _servers_from_state(2, 0, None)
    assert _class1_arrival(2, 0, a, b) == (2, 1, None)


def test_departure_from_two_class0_leaves_the_other_server_labelled():
    # n0=2 (both class0) -- A departs -> remaining class0 job is on B
    a, b = _servers_from_state(2, 0, None)
    assert _departure(2, 0, a, b, "a") == (1, 0, "B")
    # B departs -> remaining class0 job is on A
    assert _departure(2, 0, a, b, "b") == (1, 0, "A")


def test_departure_sticky_no_migration():
    # (n0=1,n1=2,config=A): class0@A, class1@B (+1 class1 queued). B (class1) departs:
    # the queued class1 immediately re-fills B (sticky -- A's class0 job is untouched).
    a, b = _servers_from_state(1, 2, "A")
    assert _departure(1, 2, a, b, "b") == (1, 1, "A")


# ------------------------------------------------------------------ exact reduction
@pytest.mark.parametrize("l0,l1,mu", [(0.3, 0.3, 1.0), (0.2, 0.5, 1.0), (0.6, 0.1, 1.2)])
def test_reduces_to_homogeneous_when_rates_equal(l0, l1, mu):
    """mu_a = mu_b must exactly reproduce MMkPriorityExact(n=2, ...)."""
    het = MM2PriorityHeterogeneousCalc(truncation=45)
    het.set_sources([l0, l1])
    het.set_servers(mu, mu)
    res_het = het.run()

    homo = MMkPriorityExact(n=2, truncation=45)
    homo.set_sources([l0, l1])
    homo.set_servers([mu, mu])
    res_homo = homo.run()

    assert np.isclose(res_het.v[0][0], res_homo.v[0][0], atol=1e-6)
    assert np.isclose(res_het.v[1][0], res_homo.v[1][0], atol=1e-6)
    assert np.isclose(res_het.w[0][0], res_homo.w[0][0], atol=1e-6)
    assert np.isclose(res_het.w[1][0], res_homo.w[1][0], atol=1e-6)


# ------------------------------------------------------------------ invariants
def test_flow_balance_invariant():
    """Steady-state throughput (mu_a*P(A busy) + mu_b*P(B busy)) must equal l0 + l1."""
    l0, l1, mu_a, mu_b = 0.4, 0.3, 1.5, 0.5
    cap = 50
    calc = MM2PriorityHeterogeneousCalc(truncation=cap)
    calc.set_sources([l0, l1])
    calc.set_servers(mu_a, mu_b)
    states = calc._states(cap)  # pylint: disable=protected-access
    index = {s: i for i, s in enumerate(states)}
    transitions = []
    for n0, n1, config in states:
        a, b = _servers_from_state(n0, n1, config)
        src = index[(n0, n1, config)]
        if n0 < cap:
            transitions.append((src, index[_class0_arrival(n0, n1, a, b)], l0))
        if n1 < cap:
            transitions.append((src, index[_class1_arrival(n0, n1, a, b)], l1))
        if a is not None:
            transitions.append((src, index[_departure(n0, n1, a, b, "a")], mu_a))
        if b is not None:
            transitions.append((src, index[_departure(n0, n1, a, b, "b")], mu_b))

    from most_queue.theory.reliability.utils import ctmc_stationary  # pylint: disable=import-outside-toplevel

    pi = ctmc_stationary(transitions, len(states))
    busy_a = sum(pi[index[s]] for s in states if _servers_from_state(*s)[0] is not None)
    busy_b = sum(pi[index[s]] for s in states if _servers_from_state(*s)[1] is not None)
    throughput = mu_a * busy_a + mu_b * busy_b
    assert np.isclose(throughput, l0 + l1, atol=1e-9)


def test_high_priority_has_lower_sojourn():
    calc = MM2PriorityHeterogeneousCalc(truncation=50)
    calc.set_sources([0.4, 0.3])
    calc.set_servers(1.5, 0.5)
    res = calc.run()
    assert res.v[0][0] < res.v[1][0]


def test_set_servers_normalises_rate_order():
    c1 = MM2PriorityHeterogeneousCalc(truncation=40)
    c1.set_sources([0.3, 0.2])
    c1.set_servers(1.5, 0.5)
    r1 = c1.run()

    c2 = MM2PriorityHeterogeneousCalc(truncation=40)
    c2.set_sources([0.3, 0.2])
    c2.set_servers(0.5, 1.5)
    r2 = c2.run()

    assert np.allclose(r1.v, r2.v)


if __name__ == "__main__":
    test_canonicalize_fills_idle_from_backlog()
    test_class0_arrival_prefers_idle_then_preempts_fastest()
    test_class1_arrival_never_preempts()
    test_departure_from_two_class0_leaves_the_other_server_labelled()
    test_departure_sticky_no_migration()
    for args in [(0.3, 0.3, 1.0), (0.2, 0.5, 1.0), (0.6, 0.1, 1.2)]:
        test_reduces_to_homogeneous_when_rates_equal(*args)
    test_flow_balance_invariant()
    test_high_priority_has_lower_sojourn()
    test_set_servers_normalises_rate_order()
    print("all mm2 priority heterogeneous tests passed")
