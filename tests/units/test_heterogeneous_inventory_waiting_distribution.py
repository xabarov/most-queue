"""
Unit tests for the exact waiting-time distribution of the HETEROGENEOUS-server
queueing-inventory calculators (EPIC-075, roadmap item R1).

These close the Д2 defect: all four heterogeneous classes used to report
``E[W] = E[V] - E[S_active]``, which overstates the wait because a service in
progress can be suspended by a stockout and that suspension belongs to the
sojourn, not to the wait. The construction lives in
``most_queue.theory.inventory._wait_phase`` (block-based builder plus
``WaitPhaseTypeMixin``) and reads the dynamics off the QBD blocks, so the same
code covers exponential, Erlang and H2 service times.

The strongest checks here are the DEGENERATE reductions: each richer class must
reproduce the simpler one bit-for-bit (Erlang with r=1 and H2 with p1=1 are just
exponential; equal rates reduce to identical servers). Those catch
implementation drift. They cannot catch an error shared by both sides -- that is
exactly how Д2 survived for eight epics -- so the magnitude of the fix was
established against an independent simulation instead (see the epic document:
E[W] matched at -0.07 sigma and the old formula was off by +23 sigma).
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.integrate import simpson

from most_queue.random.utils.params import ErlangParams, H2Params
from most_queue.theory.inventory.mm2_heterogeneous_inventory import MM2QueueingInventoryHeterogeneousCalc
from most_queue.theory.inventory.mmc_heterogeneous_erlang_inventory import (
    MMcQueueingInventoryHeterogeneousErlangCalc,
)
from most_queue.theory.inventory.mmc_heterogeneous_h2_inventory import MMcQueueingInventoryHeterogeneousH2Calc
from most_queue.theory.inventory.mmc_heterogeneous_inventory import MMcQueueingInventoryHeterogeneousCalc
from most_queue.theory.inventory.mmc_inventory import MMcQueueingInventoryCalc

MUS = [0.8, 1.5]
LAM, THETA = 1.2, 2.0
S_MAX, S = 6, 2


def _het(c=2, mus=None, policy="backorder", s_max=S_MAX, s=S):
    calc = MMcQueueingInventoryHeterogeneousCalc(c=c, s_max=s_max, s=s, policy=policy)
    calc.set_sources(LAM)
    calc.set_servers(mus or MUS, THETA)
    return calc


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
@pytest.mark.parametrize("c", [2, 3])
def test_equal_rates_match_identical_server_class_exactly(policy, c):
    """Equal rates must reproduce MMcQueueingInventoryCalc's distribution, not just its mean."""
    het = _het(c=c, mus=[1.0] * c, policy=policy)
    ref = MMcQueueingInventoryCalc(c=c, s_max=S_MAX, s=S, policy=policy)
    ref.set_sources(LAM)
    ref.set_servers(mu=1.0, theta=THETA)

    assert np.allclose(het.get_w_moments(4), ref.get_w_moments(4), rtol=1e-12)
    for t in (0.0, 0.5, 2.0, 5.0):
        assert np.isclose(het.get_tail(t), ref.get_tail(t), rtol=1e-12)


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_mm2_matches_general_c_at_c2(policy):
    """The c=2-only class and the general-c class must agree to machine precision."""
    mm2 = MM2QueueingInventoryHeterogeneousCalc(s_max=S_MAX, s=S, policy=policy)
    mm2.set_sources(LAM)
    mm2.set_servers(mu1=MUS[0], mu2=MUS[1], theta=THETA)
    mmc = _het(policy=policy)

    assert np.allclose(mm2.get_w_moments(4), mmc.get_w_moments(4), rtol=1e-12)
    assert np.isclose(mm2.get_tail(1.5), mmc.get_tail(1.5), rtol=1e-12)


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_erlang_r1_degenerates_to_exponential(policy):
    """Erlang(1, mu) IS Exp(mu), so the Erlang class must reproduce the exponential one."""
    erl = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=S_MAX, s=S, policy=policy)
    erl.set_sources(LAM)
    erl.set_servers([ErlangParams(r=1, mu=mu) for mu in MUS], THETA)

    ref = _het(policy=policy)
    assert np.allclose(erl.get_w_moments(4), ref.get_w_moments(4), rtol=1e-10)
    assert np.isclose(erl.get_tail(1.5), ref.get_tail(1.5), rtol=1e-10)


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_h2_p1_one_degenerates_to_exponential(policy):
    """H2 with p1=1 never enters branch 2, so it must reproduce the exponential class."""
    h2 = MMcQueueingInventoryHeterogeneousH2Calc(c=2, s_max=S_MAX, s=S, policy=policy)
    h2.set_sources(LAM)
    h2.set_servers([H2Params(p1=1.0, mu1=mu, mu2=9.0) for mu in MUS], THETA)

    ref = _het(policy=policy)
    assert np.allclose(h2.get_w_moments(4), ref.get_w_moments(4), rtol=1e-10)
    assert np.isclose(h2.get_tail(1.5), ref.get_tail(1.5), rtol=1e-10)


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_tail_integrates_to_the_mean(policy):
    """
    int_0^inf P(W>t) dt == E[W] -- independent consistency of moments and tail.

    The tail is stepped forward with one propagator application per grid point
    rather than ``get_tail`` per point: the latter re-exponentiates from t=0
    every time, which makes a fine grid needlessly expensive. ``level_truncation``
    is set well past the convergence point (checked at 50/100/200) to keep the
    sub-generator small.
    """
    calc = _het(policy=policy)
    alpha, t_mat, _ = calc._wait_phase_type(level_truncation=60)  # pylint: disable=protected-access

    grid = np.linspace(0.0, 80.0, 801)  # odd count: Simpson needs an even number of intervals
    dt_mat = t_mat.tocsc() * (grid[1] - grid[0])
    # u(t) = exp(T t) 1, so P(W>t) = alpha @ u(t) and u(t+dt) = exp(T dt) u(t).
    u = np.ones(t_mat.shape[0])
    tail = [float(alpha @ u)]
    for _ in grid[1:]:
        u = spla.expm_multiply(dt_mat, u)
        tail.append(float(alpha @ u))
    # Simpson, not trapezoid: the tail is steeply convex near zero, where the
    # trapezoid rule's O(dt^2) error alone is ~1e-3 on this grid.
    assert np.isclose(simpson(tail, x=grid), calc.get_w_moments(1, level_truncation=60)[0], rtol=1e-5)


def test_sojourn_splits_into_wait_plus_occupancy():
    """E[V] = E[W] + E[S], with E[S] the server-occupancy time, not the active one."""
    calc = _het()
    v, w = calc.get_v()[0], calc.get_w()[0]
    assert np.isclose(v, w + calc.get_service_time_mean(), rtol=1e-12)


def test_occupancy_exceeds_active_service_when_stock_binds():
    """
    The mechanism behind the Д2 fix: with c > 1 one server can take the last
    stock unit while another customer is mid-service, suspending it. So the
    occupancy time strictly exceeds the actively-served time, and the old
    ``E[V] - E[S_active]`` strictly overstates the wait.
    """
    calc = _het(s_max=4, s=1)
    calc.set_servers(MUS, 1.0)  # slow replenishment -> stockouts actually bind
    active = calc._mean_service_time()  # pylint: disable=protected-access
    assert calc.get_service_time_mean() > active * (1 + 1e-6)
    assert calc.get_v()[0] - active > calc.get_w()[0] * (1 + 1e-6)


def test_single_server_has_no_occupancy_inflation():
    """
    With c=1 the stock cannot drop while the only service runs, so the
    occupancy time equals 1/mu exactly and the old formula was already right.
    """
    calc = _het(c=1, mus=[1.5])
    assert np.isclose(calc.get_service_time_mean(), 1.0 / 1.5, rtol=1e-9)


@pytest.mark.parametrize(
    "policy",
    ["backorder", "lost_sales"],
)
def test_tail_is_a_proper_survival_function(policy):
    """P(W>t) must be in [0,1] and non-increasing, down into the deep tail."""
    calc = _het(policy=policy)
    grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    tail = [calc.get_tail(t) for t in grid]
    assert all(0.0 <= x <= 1.0 for x in tail)
    assert all(a >= b - 1e-14 for a, b in zip(tail, tail[1:]))
    assert np.isclose(calc.get_cdf(0.0) + calc.get_tail(0.0), 1.0)


def test_faster_servers_shorten_the_tail():
    """Speeding both servers up must not raise the deadline-violation probability."""
    prev = None
    for scale in (1.0, 1.3, 1.8):
        calc = _het(mus=[mu * scale for mu in MUS])
        cur = calc.get_tail(2.0)
        if prev is not None:
            assert cur <= prev + 1e-12
        prev = cur


def test_erlang_and_h2_give_exact_moments_in_the_non_degenerate_case():
    """Smoke check that the richer service laws produce a sane, ordered moment sequence."""
    erl = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=S_MAX, s=S)
    erl.set_sources(LAM)
    erl.set_servers([ErlangParams(r=3, mu=2.4), ErlangParams(r=2, mu=3.0)], THETA)

    h2 = MMcQueueingInventoryHeterogeneousH2Calc(c=2, s_max=S_MAX, s=S)
    h2.set_sources(LAM)
    h2.set_servers([H2Params(p1=0.4, mu1=0.5, mu2=3.0), H2Params(p1=0.6, mu1=1.0, mu2=4.0)], THETA)

    for calc in (erl, h2):
        moments = calc.get_w_moments(3)
        assert all(x > 0 for x in moments)
        assert moments[1] > moments[0] ** 2  # positive variance
        assert np.isclose(calc.get_v()[0], calc.get_w()[0] + calc.get_service_time_mean(), rtol=1e-12)
