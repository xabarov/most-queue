"""
Unit tests for the M/M/2 queueing-inventory system with two heterogeneous
servers (most_queue.theory.inventory.mm2_heterogeneous_inventory, EPIC-033).
"""

import numpy as np
import pytest

from most_queue.theory.inventory.mm2_heterogeneous_inventory import MM2QueueingInventoryHeterogeneousCalc
from most_queue.theory.inventory.mmc_inventory import MMcQueueingInventoryCalc


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
@pytest.mark.parametrize("s", [0, 1])
def test_equal_rates_reduces_exactly_to_mmc_c2(policy, s):
    """mu1=mu2 must exactly reproduce MMcQueueingInventoryCalc(c=2, ...)."""
    kwargs = dict(s_max=4, s=s, policy=policy, l=1.0, mu=1.0, theta=0.5)

    het = MM2QueueingInventoryHeterogeneousCalc(s_max=kwargs["s_max"], s=kwargs["s"], policy=kwargs["policy"])
    het.set_sources(kwargs["l"])
    het.set_servers(mu1=kwargs["mu"], mu2=kwargs["mu"], theta=kwargs["theta"])
    res_het = het.run(num_levels=100)

    c2 = MMcQueueingInventoryCalc(c=2, s_max=kwargs["s_max"], s=kwargs["s"], policy=kwargs["policy"])
    c2.set_sources(kwargs["l"])
    c2.set_servers(mu=kwargs["mu"], theta=kwargs["theta"])
    res_c2 = c2.run(num_levels=100)

    assert np.isclose(res_het.v[0], res_c2.v[0], atol=1e-6)
    assert np.isclose(res_het.w[0], res_c2.w[0], atol=1e-6)
    assert np.isclose(res_het.stockout_prob, res_c2.stockout_prob, atol=1e-9)
    assert np.allclose(res_het.p, res_c2.p, atol=1e-9)


def test_qbd_residual_is_negligible():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=5, s=2, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu1=1.2, mu2=0.8, theta=0.8)
    calc.run(num_levels=100)
    assert calc._solver.residual() < 1e-8  # pylint: disable=protected-access


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_faster_server1_does_not_increase_wait_or_stockout(policy):
    """Speeding up server 1 (fixed server 2, theta) must not increase E[W] or stockout_prob."""
    prev_w = prev_stockout = None
    for mu1 in (0.8, 1.0, 1.5, 2.0):
        calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1, policy=policy)
        calc.set_sources(1.0)
        calc.set_servers(mu1=mu1, mu2=0.7, theta=1.0)
        res = calc.run(num_levels=100)
        if prev_w is not None:
            assert res.w[0] <= prev_w + 1e-9
            assert res.stockout_prob <= prev_stockout + 1e-9
        prev_w, prev_stockout = res.w[0], res.stockout_prob


def test_distributions_are_valid_probability_vectors():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu1=1.2, mu2=0.8, theta=1.0)
    res = calc.run(num_levels=200)

    assert np.isclose(sum(res.p), 1.0, atol=1e-6)
    assert np.isclose(sum(res.stock_distribution), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert all(x >= -1e-12 for x in res.stock_distribution)
    assert np.isclose(res.stockout_prob, res.stock_distribution[0])
    assert np.isclose(res.fill_rate, 1.0 - res.stockout_prob)


def test_lost_sales_loss_prob_equals_stockout_prob():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mu1=1.2, mu2=0.8, theta=1.0)
    res = calc.run()
    assert np.isclose(res.loss_prob, res.stockout_prob)


def test_backorder_never_loses():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu1=1.2, mu2=0.8, theta=1.0)
    res = calc.run()
    assert res.loss_prob == 0.0


def test_unstable_system_rejected():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, policy="backorder")
    calc.set_sources(3.0)
    calc.set_servers(mu1=1.0, mu2=0.8, theta=0.5)
    with pytest.raises(ValueError):
        calc.run()


def test_invalid_reorder_point_rejected():
    for bad_s in (-1, 4, 5):
        with pytest.raises(ValueError):
            MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=bad_s)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        MM2QueueingInventoryHeterogeneousCalc(s_max=4, policy="nonsense")


def test_invalid_rates_rejected():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers(mu1=0.0, mu2=1.0, theta=1.0)


if __name__ == "__main__":
    for p in ("backorder", "lost_sales"):
        for s_val in (0, 1):
            test_equal_rates_reduces_exactly_to_mmc_c2(p, s_val)
        test_faster_server1_does_not_increase_wait_or_stockout(p)
    test_qbd_residual_is_negligible()
    test_distributions_are_valid_probability_vectors()
    test_lost_sales_loss_prob_equals_stockout_prob()
    test_backorder_never_loses()
    test_unstable_system_rejected()
    test_invalid_reorder_point_rejected()
    test_invalid_policy_rejected()
    test_invalid_rates_rejected()
    print("all mm2 heterogeneous queueing-inventory tests passed")
