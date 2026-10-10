"""
Unit tests for the M/M/c queueing-inventory system
(most_queue.theory.inventory.mmc_inventory, EPIC-028).
"""

import numpy as np
import pytest

from most_queue.theory.inventory.mm1_inventory import MM1QueueingInventoryCalc
from most_queue.theory.inventory.mmc_inventory import MMcQueueingInventoryCalc


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
@pytest.mark.parametrize("s", [0, 2])
def test_c_one_reduces_exactly_to_mm1(policy, s):
    """c=1 must exactly reproduce MM1QueueingInventoryCalc (same (s,S,policy))."""
    kwargs = dict(s_max=4, s=s, policy=policy, l=0.5, mu=1.0, theta=0.5)

    mmc = MMcQueueingInventoryCalc(c=1, s_max=kwargs["s_max"], s=kwargs["s"], policy=kwargs["policy"])
    mmc.set_sources(kwargs["l"])
    mmc.set_servers(mu=kwargs["mu"], theta=kwargs["theta"])
    res_c = mmc.run(num_levels=100)

    mm1 = MM1QueueingInventoryCalc(s_max=kwargs["s_max"], s=kwargs["s"], policy=kwargs["policy"])
    mm1.set_sources(kwargs["l"])
    mm1.set_servers(mu=kwargs["mu"], theta=kwargs["theta"])
    res_1 = mm1.run(num_levels=100)

    assert np.isclose(res_c.v[0], res_1.v[0], atol=1e-9)
    assert np.isclose(res_c.stockout_prob, res_1.stockout_prob, atol=1e-9)
    assert np.allclose(res_c.p, res_1.p, atol=1e-9)
    assert np.allclose(res_c.stock_distribution, res_1.stock_distribution, atol=1e-9)


def test_qbd_residual_is_negligible():
    calc = MMcQueueingInventoryCalc(c=2, s_max=5, s=2, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.8)
    calc.run(num_levels=100)
    assert calc._solver.residual() < 1e-8  # pylint: disable=protected-access


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_more_servers_does_not_increase_wait_or_stockout(policy):
    """More servers (more throughput) must not increase mean wait or stockout probability."""
    prev_w = prev_stockout = None
    for c in (2, 3, 4):
        calc = MMcQueueingInventoryCalc(c=c, s_max=4, s=1, policy=policy)
        calc.set_sources(1.0)
        calc.set_servers(mu=1.0, theta=0.5)
        res = calc.run(num_levels=150)
        if prev_w is not None:
            assert res.w[0] <= prev_w + 1e-9
            assert res.stockout_prob <= prev_stockout + 1e-9
        prev_w, prev_stockout = res.w[0], res.stockout_prob


def test_distributions_are_valid_probability_vectors():
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    assert np.isclose(sum(res.p), 1.0, atol=1e-6)
    assert np.isclose(sum(res.stock_distribution), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert all(x >= -1e-12 for x in res.stock_distribution)
    assert np.isclose(res.stockout_prob, res.stock_distribution[0])
    assert np.isclose(res.fill_rate, 1.0 - res.stockout_prob)


def test_sojourn_equals_wait_plus_actual_service():
    """V = W + S must hold with the ACTUAL mean service time, which exceeds
    1/mu once c > 1 and stockouts occur.

    This test previously asserted ``v == w + 1/mu`` and was tautological:
    ``get_w()`` was itself defined as ``v - 1/mu``, so it could never fail --
    which is exactly why the modelling error it was meant to guard went
    unnoticed until EPIC-075. With several servers one of them can take the
    last stock unit while another customer is still mid-service, suspending
    that service until a replenishment arrives.
    """
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert np.isclose(res.v[0], res.w[0] + calc.get_service_time_mean(), atol=1e-9)
    # and the blocking really is there: service takes longer than 1/mu
    assert calc.get_service_time_mean() > 1.0 / calc.mu


def test_single_server_service_time_is_exactly_one_over_mu():
    """With one server the stock cannot drop during that service, so no
    blocking is possible and E[S] = 1/mu exactly -- the boundary case that
    makes the c>1 discrepancy above meaningful rather than a numerical fluke."""
    calc = MMcQueueingInventoryCalc(c=1, s_max=4, s=1, policy="backorder")
    calc.set_sources(0.6)
    calc.set_servers(mu=1.0, theta=0.5)
    calc.run()
    assert np.isclose(calc.get_service_time_mean(), 1.0 / calc.mu, atol=1e-9)


def test_lost_sales_loss_prob_equals_stockout_prob():
    """By PASTA, every arrival sees time-stationary state, so loss_prob == stockout_prob."""
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert np.isclose(res.loss_prob, res.stockout_prob)


def test_backorder_never_loses():
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert res.loss_prob == 0.0


def test_unstable_system_rejected():
    """rho >= 1 (necessary condition) must be caught before even building the QBD."""
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, policy="backorder")
    calc.set_sources(2.5)
    calc.set_servers(mu=1.0, theta=0.5)
    with pytest.raises(ValueError):
        calc.run()


def test_invalid_c_rejected():
    with pytest.raises(ValueError):
        MMcQueueingInventoryCalc(c=0, s_max=4)


def test_invalid_reorder_point_rejected():
    for bad_s in (-1, 4, 5):
        with pytest.raises(ValueError):
            MMcQueueingInventoryCalc(c=2, s_max=4, s=bad_s)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        MMcQueueingInventoryCalc(c=2, s_max=4, policy="nonsense")


if __name__ == "__main__":
    for p in ("backorder", "lost_sales"):
        for s_val in (0, 2):
            test_c_one_reduces_exactly_to_mm1(p, s_val)
        test_more_servers_does_not_increase_wait_or_stockout(p)
    test_qbd_residual_is_negligible()
    test_distributions_are_valid_probability_vectors()
    test_sojourn_equals_wait_plus_service()
    test_lost_sales_loss_prob_equals_stockout_prob()
    test_backorder_never_loses()
    test_unstable_system_rejected()
    test_invalid_c_rejected()
    test_invalid_reorder_point_rejected()
    test_invalid_policy_rejected()
    print("all mmc queueing-inventory tests passed")
