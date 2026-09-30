"""
Unit tests for the M/M/1 queueing-inventory system with Erlang-fitted
replenishment lead time
(most_queue.theory.inventory.mm1_inventory_erlang_replenishment, EPIC-040).
"""

import numpy as np
import pytest

from most_queue.theory.inventory.mm1_inventory import MM1QueueingInventoryCalc
from most_queue.theory.inventory.mm1_inventory_erlang_replenishment import (
    MM1QueueingInventoryErlangReplenishmentCalc,
)


@pytest.mark.parametrize("policy", ["backorder", "lost_sales"])
@pytest.mark.parametrize("s", [0, 1])
def test_r1_reduces_exactly_to_exponential_replenishment(policy, s):
    """r=1 (Erlang -> Exp) must exactly reproduce MM1QueueingInventoryCalc with theta=rate."""
    s_max, lam, mu, rate = 4, 1.0, 2.0, 0.8

    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=s_max, s=s, policy=policy)
    calc.set_sources(lam)
    calc.set_servers(mu=mu, r=1, rate=rate)
    res = calc.run(num_levels=100)

    ref = MM1QueueingInventoryCalc(s_max=s_max, s=s, policy=policy)
    ref.set_sources(lam)
    ref.set_servers(mu=mu, theta=rate)
    res_ref = ref.run(num_levels=100)

    assert np.isclose(res.v[0], res_ref.v[0], atol=1e-9)
    assert np.isclose(res.w[0], res_ref.w[0], atol=1e-9)
    assert np.isclose(res.stockout_prob, res_ref.stockout_prob, atol=1e-9)
    assert np.allclose(res.stock_distribution, res_ref.stock_distribution, atol=1e-9)


def test_qbd_residual_is_negligible():
    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=5, s=2, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=2.0, r=3, rate=2.4)
    calc.run(num_levels=100)
    assert calc._solver.residual() < 1e-8  # pylint: disable=protected-access


def test_larger_r_same_mean_lead_does_not_increase_wait():
    """Fixed mean lead time (r/rate constant): larger r (lower CV) must not increase E[W]."""
    s_max, s, lam, mu = 4, 1, 1.0, 2.0
    mean_lead = 1.25
    prev_v = None
    for r in (1, 2, 4, 8):
        rate = r / mean_lead
        calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=s_max, s=s, policy="backorder")
        calc.set_sources(lam)
        calc.set_servers(mu=mu, r=r, rate=rate)
        res = calc.run(num_levels=100)
        if prev_v is not None:
            assert res.v[0] <= prev_v + 1e-9
        prev_v = res.v[0]


def test_distributions_are_valid_probability_vectors():
    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=2.0, r=3, rate=2.4)
    res = calc.run(num_levels=200)

    assert np.isclose(sum(res.p), 1.0, atol=1e-6)
    assert np.isclose(sum(res.stock_distribution), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert all(x >= -1e-12 for x in res.stock_distribution)
    assert np.isclose(res.stockout_prob, res.stock_distribution[0])
    assert np.isclose(res.fill_rate, 1.0 - res.stockout_prob)


def test_set_replenishment_from_moments_fits_erlang():
    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    mean, m2 = 1.25, 1.8  # var < mean^2 -> cv < 1, fittable to Erlang
    calc.set_replenishment_from_moments(mu=2.0, moments=[mean, m2])
    assert calc.r >= 1
    assert calc.rate > 0
    res = calc.run()
    assert res.v[0] > 0


def test_lost_sales_loss_prob_equals_stockout_prob():
    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mu=2.0, r=3, rate=2.4)
    res = calc.run()
    assert np.isclose(res.loss_prob, res.stockout_prob)


def test_backorder_never_loses():
    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=4, s=1, policy="backorder")
    calc.set_sources(1.0)
    calc.set_servers(mu=2.0, r=3, rate=2.4)
    res = calc.run()
    assert res.loss_prob == 0.0


def test_invalid_reorder_point_rejected():
    for bad_s in (-1, 4, 5):
        with pytest.raises(ValueError):
            MM1QueueingInventoryErlangReplenishmentCalc(s_max=4, s=bad_s)


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        MM1QueueingInventoryErlangReplenishmentCalc(s_max=4, policy="nonsense")


def test_invalid_server_params_rejected():
    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=4)
    calc.set_sources(1.0)
    with pytest.raises(ValueError):
        calc.set_servers(mu=0.0, r=2, rate=1.0)
    with pytest.raises(ValueError):
        calc.set_servers(mu=1.0, r=0, rate=1.0)
    with pytest.raises(ValueError):
        calc.set_servers(mu=1.0, r=2, rate=0.0)


if __name__ == "__main__":
    for p in ("backorder", "lost_sales"):
        for s_val in (0, 1):
            test_r1_reduces_exactly_to_exponential_replenishment(p, s_val)
    test_qbd_residual_is_negligible()
    test_larger_r_same_mean_lead_does_not_increase_wait()
    test_distributions_are_valid_probability_vectors()
    test_set_replenishment_from_moments_fits_erlang()
    test_lost_sales_loss_prob_equals_stockout_prob()
    test_backorder_never_loses()
    test_invalid_reorder_point_rejected()
    test_invalid_policy_rejected()
    test_invalid_server_params_rejected()
    print("all mm1 Erlang-replenishment queueing-inventory tests passed")
