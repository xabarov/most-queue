"""
Unit tests for the M/M/c queueing-inventory system with c heterogeneous
servers (most_queue.theory.inventory.mmc_heterogeneous_inventory, EPIC-038).
"""

import numpy as np
import pytest

from most_queue.theory.inventory.mm2_heterogeneous_inventory import MM2QueueingInventoryHeterogeneousCalc
from most_queue.theory.inventory.mmc_heterogeneous_inventory import MMcQueueingInventoryHeterogeneousCalc
from most_queue.theory.inventory.mmc_inventory import MMcQueueingInventoryCalc


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
@pytest.mark.parametrize("s", [0, 1])
def test_c2_reduces_exactly_to_mm2_heterogeneous(policy, s):
    """c=2 must exactly reproduce MM2QueueingInventoryHeterogeneousCalc (structural regression)."""
    kwargs = dict(s_max=4, s=s, policy=policy, l=1.0, mu1=1.2, mu2=0.8, theta=0.8)

    het = MMcQueueingInventoryHeterogeneousCalc(c=2, s_max=kwargs["s_max"], s=kwargs["s"], policy=kwargs["policy"])
    het.set_sources(kwargs["l"])
    het.set_servers(mus=[kwargs["mu1"], kwargs["mu2"]], theta=kwargs["theta"])
    res_het = het.run(num_levels=100)

    ref = MM2QueueingInventoryHeterogeneousCalc(s_max=kwargs["s_max"], s=kwargs["s"], policy=kwargs["policy"])
    ref.set_sources(kwargs["l"])
    ref.set_servers(mu1=kwargs["mu1"], mu2=kwargs["mu2"], theta=kwargs["theta"])
    res_ref = ref.run(num_levels=100)

    assert np.isclose(res_het.v[0], res_ref.v[0], atol=1e-9)
    assert np.isclose(res_het.w[0], res_ref.w[0], atol=1e-9)
    assert np.isclose(res_het.stockout_prob, res_ref.stockout_prob, atol=1e-9)
    assert np.allclose(res_het.p, res_ref.p, atol=1e-9)


@pytest.mark.parametrize("c", [1, 3, 4, 5])
def test_equal_rates_reduces_exactly_to_mmc(c):
    """mu_1=...=mu_c must exactly reproduce MMcQueueingInventoryCalc(c, ...)."""
    mu, lam, theta = 1.3, 1.0, 0.7

    het = MMcQueueingInventoryHeterogeneousCalc(c=c, s_max=5, s=1, policy="backorder")
    het.set_sources(lam)
    het.set_servers(mus=[mu] * c, theta=theta)
    res_het = het.run(num_levels=100)

    ref = MMcQueueingInventoryCalc(c=c, s_max=5, s=1, policy="backorder")
    ref.set_sources(lam)
    ref.set_servers(mu=mu, theta=theta)
    res_ref = ref.run(num_levels=100)

    assert np.isclose(res_het.v[0], res_ref.v[0], atol=1e-6)
    assert np.isclose(res_het.w[0], res_ref.w[0], atol=1e-6)
    assert np.isclose(res_het.stockout_prob, res_ref.stockout_prob, atol=1e-9)


def test_qbd_residual_is_negligible():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=5, s=2, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mus=[1.5, 1.0, 0.7], theta=0.8)
    calc.run(num_levels=100)
    assert calc._solver.residual() < 1e-8  # pylint: disable=protected-access


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
def test_faster_first_server_does_not_increase_wait_or_stockout(policy):
    """Speeding up server 0 (fixed others, theta) must not increase E[W] or stockout_prob."""
    prev_w = prev_stockout = None
    for mu0 in (0.8, 1.0, 1.5, 2.0):
        calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4, s=1, policy=policy)
        calc.set_sources(1.0)
        calc.set_servers(mus=[mu0, 0.7, 0.6], theta=1.0)
        res = calc.run(num_levels=100)
        if prev_w is not None:
            assert res.w[0] <= prev_w + 1e-9
            assert res.stockout_prob <= prev_stockout + 1e-9
        prev_w, prev_stockout = res.w[0], res.stockout_prob


def test_distributions_are_valid_probability_vectors():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    res = calc.run(num_levels=200)

    assert np.isclose(sum(res.p), 1.0, atol=1e-6)
    assert np.isclose(sum(res.stock_distribution), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert all(x >= -1e-12 for x in res.stock_distribution)
    assert np.isclose(res.stockout_prob, res.stock_distribution[0])
    assert np.isclose(res.fill_rate, 1.0 - res.stockout_prob)


def test_lost_sales_loss_prob_equals_stockout_prob():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    res = calc.run()
    assert np.isclose(res.loss_prob, res.stockout_prob)


def test_backorder_never_loses():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    res = calc.run()
    assert res.loss_prob == 0.0


def test_unstable_system_rejected():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=2, s_max=4, policy="backorder")
    calc.set_sources(3.0)
    calc.set_servers(mus=[1.0, 0.8], theta=0.5)
    with pytest.raises(ValueError):
        calc.run()


def test_invalid_c_rejected():
    with pytest.raises(ValueError):
        MMcQueueingInventoryHeterogeneousCalc(c=0, s_max=4)


def test_invalid_reorder_point_rejected():
    for bad_s in (-1, 4, 5):
        with pytest.raises(ValueError):
            MMcQueueingInventoryHeterogeneousCalc(c=2, s_max=4, s=bad_s)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        MMcQueueingInventoryHeterogeneousCalc(c=2, s_max=4, policy="nonsense")


def test_wrong_number_of_rates_rejected():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers(mus=[1.0, 0.8], theta=1.0)


def test_invalid_rates_rejected():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=2, s_max=4)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers(mus=[0.0, 1.0], theta=1.0)


if __name__ == "__main__":
    for p in ("backorder", "lost_sales"):
        for s_val in (0, 1):
            test_c2_reduces_exactly_to_mm2_heterogeneous(p, s_val)
        test_faster_first_server_does_not_increase_wait_or_stockout(p)
    for c_val in (1, 3, 4, 5):
        test_equal_rates_reduces_exactly_to_mmc(c_val)
    test_qbd_residual_is_negligible()
    test_distributions_are_valid_probability_vectors()
    test_lost_sales_loss_prob_equals_stockout_prob()
    test_backorder_never_loses()
    test_unstable_system_rejected()
    test_invalid_c_rejected()
    test_invalid_reorder_point_rejected()
    test_invalid_policy_rejected()
    test_wrong_number_of_rates_rejected()
    test_invalid_rates_rejected()
    print("all mmc heterogeneous queueing-inventory tests passed")
