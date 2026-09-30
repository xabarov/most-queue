"""
Unit tests for the M/Erlang/c queueing-inventory system with c heterogeneous
servers, each with its own Erlang service-time distribution
(most_queue.theory.inventory.mmc_heterogeneous_erlang_inventory, EPIC-041).
"""

import numpy as np
import pytest

from most_queue.random.utils.params import ErlangParams
from most_queue.theory.inventory.mmc_heterogeneous_erlang_inventory import (
    MMcQueueingInventoryHeterogeneousErlangCalc,
)
from most_queue.theory.inventory.mmc_heterogeneous_inventory import MMcQueueingInventoryHeterogeneousCalc


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
@pytest.mark.parametrize("c", [2, 3])
def test_r_one_reduces_exactly_to_plain_heterogeneous(policy, c):
    """r_k=1 for all k (Erlang -> Exp(mu_k)) must exactly reproduce MMcQueueingInventoryHeterogeneousCalc."""
    mus = [1.2, 0.8, 1.5][:c]

    erl = MMcQueueingInventoryHeterogeneousErlangCalc(c=c, s_max=4, s=1, policy=policy)
    erl.set_sources(1.0)
    erl.set_servers([ErlangParams(r=1, mu=mu) for mu in mus], theta=0.8)
    res_erl = erl.run(num_levels=100)

    ref = MMcQueueingInventoryHeterogeneousCalc(c=c, s_max=4, s=1, policy=policy)
    ref.set_sources(1.0)
    ref.set_servers(mus=mus, theta=0.8)
    res_ref = ref.run(num_levels=100)

    assert np.isclose(res_erl.v[0], res_ref.v[0], atol=1e-9)
    assert np.isclose(res_erl.w[0], res_ref.w[0], atol=1e-9)
    assert np.isclose(res_erl.stockout_prob, res_ref.stockout_prob, atol=1e-9)
    assert np.allclose(res_erl.p, res_ref.p, atol=1e-9)


def test_qbd_residual_is_negligible_for_nondegenerate_r():
    """Non-degenerate r_k>1, heterogeneous across servers -- the case that caught the
    departure/advance-rate outflow-splitting bug during development."""
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers([ErlangParams(r=2, mu=2.4), ErlangParams(r=3, mu=2.4)], theta=0.8)
    calc.run(num_levels=100)
    assert calc._solver.residual() < 1e-8  # pylint: disable=protected-access


def test_distributions_are_valid_probability_vectors():
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers([ErlangParams(r=2, mu=2.4), ErlangParams(r=3, mu=3.2)], theta=1.0)
    res = calc.run(num_levels=200)

    assert np.isclose(sum(res.p), 1.0, atol=1e-6)
    assert np.isclose(sum(res.stock_distribution), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert all(x >= -1e-12 for x in res.stock_distribution)
    assert np.isclose(res.stockout_prob, res.stock_distribution[0])
    assert np.isclose(res.fill_rate, 1.0 - res.stockout_prob)


def test_set_servers_from_moments_fits_each_server_independently():
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers_from_moments([[1.0, 1.2], [1.5, 2.4]], theta=1.0)
    assert len(calc.servers) == 2
    for params in calc.servers:
        assert params.r >= 1
        assert params.mu > 0
    res = calc.run()
    assert res.v[0] > 0


def test_lost_sales_loss_prob_equals_stockout_prob():
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers([ErlangParams(r=2, mu=2.4), ErlangParams(r=3, mu=3.2)], theta=1.0)
    res = calc.run()
    assert np.isclose(res.loss_prob, res.stockout_prob)


def test_backorder_never_loses():
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers([ErlangParams(r=2, mu=2.4), ErlangParams(r=3, mu=3.2)], theta=1.0)
    res = calc.run()
    assert res.loss_prob == 0.0


def test_invalid_c_rejected():
    with pytest.raises(ValueError):
        MMcQueueingInventoryHeterogeneousErlangCalc(c=0, s_max=4)


def test_invalid_reorder_point_rejected():
    for bad_s in (-1, 4, 5):
        with pytest.raises(ValueError):
            MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=bad_s)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, policy="nonsense")


def test_wrong_number_of_servers_rejected():
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=3, s_max=4)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers([ErlangParams(r=1, mu=1.0)] * 2, theta=1.0)


def test_invalid_erlang_params_rejected():
    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=1, s_max=4)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers([ErlangParams(r=0, mu=1.0)], theta=1.0)
    with pytest.raises(ValueError):
        calc.set_servers([ErlangParams(r=1, mu=0.0)], theta=1.0)


if __name__ == "__main__":
    for p in ("backorder", "lost_sales"):
        for c_val in (2, 3):
            test_r_one_reduces_exactly_to_plain_heterogeneous(p, c_val)
    test_qbd_residual_is_negligible_for_nondegenerate_r()
    test_distributions_are_valid_probability_vectors()
    test_set_servers_from_moments_fits_each_server_independently()
    test_lost_sales_loss_prob_equals_stockout_prob()
    test_backorder_never_loses()
    test_invalid_c_rejected()
    test_invalid_reorder_point_rejected()
    test_invalid_policy_rejected()
    test_wrong_number_of_servers_rejected()
    test_invalid_erlang_params_rejected()
    print("all mmc heterogeneous Erlang queueing-inventory tests passed")
