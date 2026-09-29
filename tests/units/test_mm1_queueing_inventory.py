"""
Unit tests for the M/M/1 queueing-inventory system, (0,S) policy, backorder
(most_queue.theory.inventory.mm1_inventory), EPIC-024, and lost-sales
(EPIC-026).
"""

import numpy as np

from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.inventory import MM1QueueingInventoryCalc


def test_reduces_to_plain_mm1_when_stock_never_runs_out():
    """Large S and fast replenishment: stockout is negligible, must match exact M/M/1."""
    lam, mu = 0.6, 1.0
    calc = MM1QueueingInventoryCalc(s_max=200)
    calc.set_sources(lam)
    calc.set_servers(mu=mu, theta=1000.0)
    res = calc.run(num_levels=300)

    mg1 = MG1Calc()
    mg1.set_sources(lam)
    mg1.set_servers([1 / mu, 2 / mu**2, 6 / mu**3, 24 / mu**4])
    ref = mg1.run()

    assert np.isclose(res.v[0], ref.v[0], rtol=1e-3)
    assert np.isclose(res.w[0], ref.w[0], rtol=1e-3)
    assert res.stockout_prob < 1e-4


def test_qbd_residual_is_negligible():
    """QBDSolver's own balance-equation residual must be ~0."""
    calc = MM1QueueingInventoryCalc(s_max=5)
    calc.set_sources(0.4)
    calc.set_servers(mu=1.0, theta=0.8)
    calc.run(num_levels=100)
    assert calc._solver.residual() < 1e-8  # pylint: disable=protected-access


def test_distributions_are_valid_probability_vectors():
    calc = MM1QueueingInventoryCalc(s_max=4)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    assert np.isclose(sum(res.p), 1.0, atol=1e-6)
    assert np.isclose(sum(res.stock_distribution), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert all(x >= -1e-12 for x in res.stock_distribution)
    assert np.isclose(res.stockout_prob, res.stock_distribution[0])
    assert np.isclose(res.fill_rate, 1.0 - res.stockout_prob)


def test_sojourn_equals_wait_plus_service():
    """V = W + S must hold exactly for FCFS (Little's law decomposition)."""
    calc = MM1QueueingInventoryCalc(s_max=3)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert np.isclose(res.v[0], res.w[0] + 1.0 / calc.mu, atol=1e-9)


def test_smaller_stock_or_slower_replenishment_increases_wait():
    """More stockout risk (smaller S, or slower theta) must not decrease mean wait."""

    def _wait(s_max, theta):
        c = MM1QueueingInventoryCalc(s_max=s_max)
        c.set_sources(0.5)
        c.set_servers(mu=1.0, theta=theta)
        return c.run().w[0]

    assert _wait(2, 0.5) >= _wait(5, 0.5) - 1e-9
    assert _wait(5, 0.2) >= _wait(5, 1.0) - 1e-9


def test_unstable_system_rejected():
    calc = MM1QueueingInventoryCalc(s_max=3)
    calc.set_sources(1.5)
    calc.set_servers(mu=1.0, theta=0.5)
    try:
        calc.run()
        assert False, "expected ValueError for rho >= 1"
    except ValueError:
        pass


def test_invalid_policy_rejected():
    try:
        MM1QueueingInventoryCalc(s_max=3, policy="nonsense")
        assert False, "expected ValueError for an unknown policy"
    except ValueError:
        pass


# ------------------------------------------------------------------ EPIC-026: lost sales
def test_lost_sales_reduces_to_plain_mm1_when_stock_never_runs_out():
    """Same reduction check as backorder, but for policy='lost_sales'."""
    lam, mu = 0.6, 1.0
    calc = MM1QueueingInventoryCalc(s_max=200, policy="lost_sales")
    calc.set_sources(lam)
    calc.set_servers(mu=mu, theta=1000.0)
    res = calc.run(num_levels=300)

    mg1 = MG1Calc()
    mg1.set_sources(lam)
    mg1.set_servers([1 / mu, 2 / mu**2, 6 / mu**3, 24 / mu**4])
    ref = mg1.run()

    assert np.isclose(res.v[0], ref.v[0], rtol=1e-3)
    assert np.isclose(res.w[0], ref.w[0], rtol=1e-3)
    assert res.loss_prob < 1e-4


def test_lost_sales_has_smaller_mean_in_system_than_backorder():
    """Shedding demand during a stockout must not increase mean sojourn/queue length."""
    kwargs = {"s_max": 3, "l": 0.5, "mu": 1.0, "theta": 0.5}

    backorder = MM1QueueingInventoryCalc(s_max=kwargs["s_max"], policy="backorder")
    backorder.set_sources(kwargs["l"])
    backorder.set_servers(mu=kwargs["mu"], theta=kwargs["theta"])
    res_backorder = backorder.run()

    lost = MM1QueueingInventoryCalc(s_max=kwargs["s_max"], policy="lost_sales")
    lost.set_sources(kwargs["l"])
    lost.set_servers(mu=kwargs["mu"], theta=kwargs["theta"])
    res_lost = lost.run()

    assert res_lost.v[0] < res_backorder.v[0]
    assert res_lost.w[0] < res_backorder.w[0]


def test_lost_sales_loss_prob_equals_stockout_prob():
    """By PASTA, every arrival sees time-stationary state, so loss_prob == stockout_prob."""
    calc = MM1QueueingInventoryCalc(s_max=4, policy="lost_sales")
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert np.isclose(res.loss_prob, res.stockout_prob)


def test_backorder_never_loses():
    calc = MM1QueueingInventoryCalc(s_max=4, policy="backorder")
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert res.loss_prob == 0.0


def test_lost_sales_sojourn_equals_wait_plus_service():
    """V = W + S must still hold exactly for accepted customers under lost sales."""
    calc = MM1QueueingInventoryCalc(s_max=3, policy="lost_sales")
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run()
    assert np.isclose(res.v[0], res.w[0] + 1.0 / calc.mu, atol=1e-9)


if __name__ == "__main__":
    test_reduces_to_plain_mm1_when_stock_never_runs_out()
    test_qbd_residual_is_negligible()
    test_distributions_are_valid_probability_vectors()
    test_sojourn_equals_wait_plus_service()
    test_smaller_stock_or_slower_replenishment_increases_wait()
    test_unstable_system_rejected()
    test_invalid_policy_rejected()
    test_lost_sales_reduces_to_plain_mm1_when_stock_never_runs_out()
    test_lost_sales_has_smaller_mean_in_system_than_backorder()
    test_lost_sales_loss_prob_equals_stockout_prob()
    test_backorder_never_loses()
    test_lost_sales_sojourn_equals_wait_plus_service()
    print("all mm1 queueing-inventory tests passed")
