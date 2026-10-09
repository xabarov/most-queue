"""
Unit tests for EPIC-075 (Р1 of the literature-catchup roadmap): exact
waiting-time DISTRIBUTION (moments and tail) for the M/M/1 queueing-inventory
system, which previously reported only the mean.

The distribution itself is known theory (Jeganathan et al. 2021 derive its LST
via the matrix-geometric technique; see the module docstring of
`most_queue.theory.inventory.mm1_inventory` for the full attribution). What is
tested here is our implementation of it by a different computational route --
a tagged-customer absorbing chain.
"""

from collections import deque

import numpy as np
import pytest
from scipy import integrate

from most_queue.theory.inventory.mm1_inventory import MM1QueueingInventoryCalc


def _independent_des(s_max, s, lam, mu, theta, total=1_500_000, warmfrac=0.1, seed=1, lost=False):
    """From-scratch DES (no most_queue code reused): M/M/1 queueing-inventory,
    (s,S) policy, Exp(theta) lead time. Returns the sample of waiting times
    measured at the instant each customer ENTERS service."""
    rng = np.random.default_rng(seed)
    inf = float("inf")
    t = 0.0
    stock = s_max
    order_out = False
    queue: deque = deque()
    busy = False
    next_arr = rng.exponential(1 / lam)
    next_dep = inf
    next_repl = inf
    n_acc = 0
    waits: list[float] = []

    def maybe_order():
        nonlocal order_out, next_repl
        if not order_out and stock <= s:
            order_out = True
            next_repl = t + rng.exponential(1 / theta)

    def maybe_start():
        nonlocal busy, next_dep
        if (not busy) and queue and stock > 0:
            arr = queue.popleft()
            busy = True
            next_dep = t + rng.exponential(1 / mu)
            waits.append(t - arr)

    maybe_order()
    while n_acc < total:
        tnext = min(next_arr, next_dep, next_repl)
        t = tnext
        if tnext == next_arr:
            if not (lost and stock == 0):
                n_acc += 1
                queue.append(t)
            next_arr = t + rng.exponential(1 / lam)
            maybe_start()
        elif tnext == next_dep:
            stock -= 1
            busy = False
            next_dep = inf
            maybe_order()
            maybe_start()
        else:
            stock = s_max
            order_out = False
            next_repl = inf
            maybe_order()
            maybe_start()
    return np.array(waits[int(len(waits) * warmfrac) :])


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_phase_type_mean_matches_littles_law(policy):
    """Strongest internal cross-check: the first moment from the absorbing
    chain must equal `get_w()`, which goes through an entirely independent
    path (Little's law on the matrix-geometric stationary distribution)."""
    calc = MM1QueueingInventoryCalc(s_max=4, s=1, policy=policy)
    calc.set_sources(0.7)
    calc.set_servers(mu=1.2, theta=0.9)
    assert calc.get_w_moments(1)[0] == pytest.approx(calc.get_w()[0], rel=1e-9)


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_tail_integrates_to_mean(policy):
    """Internal consistency of the tail against the moments: int_0^inf P(W>t) dt = E[W]."""
    calc = MM1QueueingInventoryCalc(s_max=4, s=1, policy=policy)
    calc.set_sources(0.7)
    calc.set_servers(mu=1.2, theta=0.9)
    e_w = calc.get_w_moments(1)[0]
    e_w_int, _ = integrate.quad(calc.get_tail, 0, 400, limit=300)
    assert e_w_int == pytest.approx(e_w, rel=1e-6)


def test_tail_at_zero_is_probability_of_waiting():
    """P(W>0) must equal 1 minus the probability of immediate service, i.e.
    the mass the construction assigns to zero wait."""
    calc = MM1QueueingInventoryCalc(s_max=5, s=1)
    calc.set_sources(0.6)
    calc.set_servers(mu=1.0, theta=0.8)
    alpha, _, p_zero = calc._wait_phase_type()  # pylint: disable=protected-access
    assert calc.get_tail(0.0) == pytest.approx(float(alpha.sum()), rel=1e-12)
    assert float(alpha.sum()) + p_zero == pytest.approx(1.0, abs=1e-10)


def test_stockout_makes_empty_queue_customer_wait():
    """Model-specific check that distinguishes this from an ordinary M/M/1:
    with a slow replenishment, a customer can wait even with an empty queue,
    so P(W>0) must stay well above the ordinary utilization rho.

    Note on the parameters: the stock-limited throughput is about
    ``s_max * theta``, and it -- not rho = lam/mu -- is the binding stability
    condition here (rho < 1 is necessary but NOT sufficient in a
    queueing-inventory system). An earlier draft of this test used
    ``s_max=2, theta=0.05`` with ``lam=0.3``: rho was only 0.15, yet
    ``s_max * theta = 0.1 < lam``, so the chain was genuinely unstable and the
    solver correctly refused it. Keep ``s_max * theta`` comfortably above
    ``lam`` when making the lead time slow.
    """
    lam, mu = 0.3, 2.0
    calc = MM1QueueingInventoryCalc(s_max=10, s=0)
    calc.set_sources(lam)
    calc.set_servers(mu=mu, theta=0.1)  # lead time 10 vs service 0.5 -> 20x
    assert calc.get_tail(0.0) > lam / mu + 0.2


def test_moments_increase_and_are_positive():
    calc = MM1QueueingInventoryCalc(s_max=3, s=1)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=1.0)
    moments = calc.get_w_moments(4)
    assert all(m > 0 for m in moments)
    # raw moments of a non-degenerate positive rv grow after normalisation
    assert moments[1] > moments[0] ** 2


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_mean_and_tail_match_independent_des(policy):
    """Validation against an independent, from-scratch discrete-event model."""
    s_max, s, lam, mu, theta = 5, 1, 0.6, 1.0, 0.8
    calc = MM1QueueingInventoryCalc(s_max=s_max, s=s, policy=policy)
    calc.set_sources(lam)
    calc.set_servers(mu=mu, theta=theta)

    w = _independent_des(s_max, s, lam, mu, theta, total=1_500_000, seed=7, lost=(policy == "lost_sales"))
    assert w.mean() == pytest.approx(calc.get_w_moments(1)[0], rel=0.03)
    for t in (0.0, 1.0, 3.0, 6.0):
        assert float((w > t).mean()) == pytest.approx(calc.get_tail(t), rel=0.05)


def test_cdf_complements_tail():
    calc = MM1QueueingInventoryCalc(s_max=3, s=1)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=1.0)
    for t in (0.0, 0.5, 2.0):
        assert calc.get_cdf(t) == pytest.approx(1.0 - calc.get_tail(t), rel=1e-12)


def test_rejects_negative_time():
    calc = MM1QueueingInventoryCalc(s_max=3, s=1)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=1.0)
    with pytest.raises(ValueError):
        calc.get_tail(-1.0)
