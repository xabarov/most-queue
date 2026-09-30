"""
Cross-validate the M/M/1 queueing-inventory QBD calculator (EPIC-024,
backorder; EPIC-026, lost sales; EPIC-027, general (s,S)), the M/M/c
generalization (EPIC-028), the two-heterogeneous-server generalization
(EPIC-033), the general-c heterogeneous-server generalization (EPIC-038),
the per-server H2 service-time generalization (EPIC-039), the
Erlang-fitted replenishment lead time generalization (EPIC-040), and the
per-server Erlang service-time generalization (EPIC-041) against
discrete-event simulation.
"""

import numpy as np

from most_queue.random.utils.params import ErlangParams, H2Params
from most_queue.sim.inventory import (
    MM1QueueingInventoryErlangReplenishmentSim,
    MM1QueueingInventorySim,
    MM2QueueingInventoryHeterogeneousSim,
    MMcQueueingInventoryHeterogeneousErlangSim,
    MMcQueueingInventoryHeterogeneousH2Sim,
    MMcQueueingInventoryHeterogeneousSim,
    MMcQueueingInventorySim,
)
from most_queue.theory.inventory import (
    MM1QueueingInventoryCalc,
    MM1QueueingInventoryErlangReplenishmentCalc,
    MM2QueueingInventoryHeterogeneousCalc,
    MMcQueueingInventoryCalc,
    MMcQueueingInventoryHeterogeneousCalc,
    MMcQueueingInventoryHeterogeneousErlangCalc,
    MMcQueueingInventoryHeterogeneousH2Calc,
)


def test_mm1_queueing_inventory_vs_sim():
    calc = MM1QueueingInventoryCalc(s_max=4)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    sim = MM1QueueingInventorySim(s_max=4, seed=42)
    sim.set_sources(0.5)
    sim.set_servers(mu=1.0, theta=0.5)
    sim.run(400_000)

    mean_in_system_sim = sim.mean_in_system
    mean_v_sim = mean_in_system_sim / 0.5

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.03)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mm1_queueing_inventory_lost_sales_vs_sim():
    calc = MM1QueueingInventoryCalc(s_max=4, policy="lost_sales")
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    sim = MM1QueueingInventorySim(s_max=4, policy="lost_sales", seed=42)
    sim.set_sources(0.5)
    sim.set_servers(mu=1.0, theta=0.5)
    sim.run(400_000)

    lambda_eff_sim = 0.5 * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.03)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mm1_queueing_inventory_general_sS_vs_sim():
    calc = MM1QueueingInventoryCalc(s_max=4, s=2)
    calc.set_sources(0.5)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    sim = MM1QueueingInventorySim(s_max=4, s=2, seed=42)
    sim.set_sources(0.5)
    sim.set_servers(mu=1.0, theta=0.5)
    sim.run(400_000)

    mean_v_sim = sim.mean_in_system / 0.5

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.03)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_vs_sim():
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1)
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventorySim(c=2, s_max=4, s=1, seed=42)
    sim.set_sources(1.0)
    sim.set_servers(mu=1.0, theta=0.5)
    sim.run(600_000)

    mean_v_sim = sim.mean_in_system / 1.0

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_lost_sales_vs_sim():
    calc = MMcQueueingInventoryCalc(c=2, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mu=1.0, theta=0.5)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventorySim(c=2, s_max=4, s=1, policy="lost_sales", seed=42)
    sim.set_sources(1.0)
    sim.set_servers(mu=1.0, theta=0.5)
    sim.run(600_000)

    lambda_eff_sim = 1.0 * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mm2_queueing_inventory_heterogeneous_vs_sim():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1)
    calc.set_sources(1.0)
    calc.set_servers(mu1=1.5, mu2=0.7, theta=1.0)
    res = calc.run(num_levels=200)

    sim = MM2QueueingInventoryHeterogeneousSim(s_max=4, s=1, seed=42)
    sim.set_sources(1.0)
    sim.set_servers(mu1=1.5, mu2=0.7, theta=1.0)
    sim.run(600_000)

    mean_v_sim = sim.mean_in_system / 1.0

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mm2_queueing_inventory_heterogeneous_lost_sales_vs_sim():
    calc = MM2QueueingInventoryHeterogeneousCalc(s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mu1=1.5, mu2=0.7, theta=1.0)
    res = calc.run(num_levels=200)

    sim = MM2QueueingInventoryHeterogeneousSim(s_max=4, s=1, policy="lost_sales", seed=42)
    sim.set_sources(1.0)
    sim.set_servers(mu1=1.5, mu2=0.7, theta=1.0)
    sim.run(600_000)

    lambda_eff_sim = 1.0 * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_heterogeneous_vs_sim():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4, s=1)
    calc.set_sources(1.0)
    calc.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventoryHeterogeneousSim(c=3, s_max=4, s=1, seed=42)
    sim.set_sources(1.0)
    sim.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    sim.run(600_000)

    mean_v_sim = sim.mean_in_system / 1.0

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_heterogeneous_lost_sales_vs_sim():
    calc = MMcQueueingInventoryHeterogeneousCalc(c=3, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventoryHeterogeneousSim(c=3, s_max=4, s=1, policy="lost_sales", seed=42)
    sim.set_sources(1.0)
    sim.set_servers(mus=[1.5, 1.0, 0.7], theta=1.0)
    sim.run(600_000)

    lambda_eff_sim = 1.0 * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_heterogeneous_h2_vs_sim():
    servers = [
        H2Params(p1=0.5, mu1=1.5, mu2=3.0),
        H2Params(p1=0.3, mu1=0.8, mu2=2.5),
        H2Params(p1=0.6, mu1=1.0, mu2=2.0),
    ]

    calc = MMcQueueingInventoryHeterogeneousH2Calc(c=3, s_max=4, s=1)
    calc.set_sources(1.0)
    calc.set_servers(servers, theta=1.0)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventoryHeterogeneousH2Sim(c=3, s_max=4, s=1, seed=42)
    sim.set_sources(1.0)
    sim.set_servers(servers, theta=1.0)
    sim.run(600_000)

    mean_v_sim = sim.mean_in_system / 1.0

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_heterogeneous_h2_lost_sales_vs_sim():
    servers = [
        H2Params(p1=0.5, mu1=1.5, mu2=3.0),
        H2Params(p1=0.3, mu1=0.8, mu2=2.5),
        H2Params(p1=0.6, mu1=1.0, mu2=2.0),
    ]

    calc = MMcQueueingInventoryHeterogeneousH2Calc(c=3, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(servers, theta=1.0)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventoryHeterogeneousH2Sim(c=3, s_max=4, s=1, policy="lost_sales", seed=42)
    sim.set_sources(1.0)
    sim.set_servers(servers, theta=1.0)
    sim.run(600_000)

    lambda_eff_sim = 1.0 * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mm1_queueing_inventory_erlang_replenishment_vs_sim():
    s_max, s, lam, mu = 4, 1, 1.0, 2.0
    mean_lead = 1.25
    r = 3
    rate = r / mean_lead

    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=s_max, s=s)
    calc.set_sources(lam)
    calc.set_servers(mu=mu, r=r, rate=rate)
    res = calc.run(num_levels=200)

    sim = MM1QueueingInventoryErlangReplenishmentSim(s_max=s_max, s=s, seed=42)
    sim.set_sources(lam)
    sim.set_servers(mu=mu, r=r, rate=rate)
    sim.run(600_000)

    mean_v_sim = sim.mean_in_system / lam

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mm1_queueing_inventory_erlang_replenishment_lost_sales_vs_sim():
    s_max, s, lam, mu = 4, 1, 1.0, 2.0
    mean_lead = 1.25
    r = 3
    rate = r / mean_lead

    calc = MM1QueueingInventoryErlangReplenishmentCalc(s_max=s_max, s=s, policy="lost_sales")
    calc.set_sources(lam)
    calc.set_servers(mu=mu, r=r, rate=rate)
    res = calc.run(num_levels=200)

    sim = MM1QueueingInventoryErlangReplenishmentSim(s_max=s_max, s=s, policy="lost_sales", seed=42)
    sim.set_sources(lam)
    sim.set_servers(mu=mu, r=r, rate=rate)
    sim.run(600_000)

    lambda_eff_sim = lam * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_heterogeneous_erlang_vs_sim():
    mean1, mean2 = 1 / 1.2, 1 / 0.8
    servers = [ErlangParams(r=2, mu=2 / mean1), ErlangParams(r=3, mu=3 / mean2)]

    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1)
    calc.set_sources(1.0)
    calc.set_servers(servers, theta=0.8)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventoryHeterogeneousErlangSim(c=2, s_max=4, s=1, seed=42)
    sim.set_sources(1.0)
    sim.set_servers(servers, theta=0.8)
    sim.run(600_000)

    mean_v_sim = sim.mean_in_system / 1.0

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


def test_mmc_queueing_inventory_heterogeneous_erlang_lost_sales_vs_sim():
    mean1, mean2 = 1 / 1.2, 1 / 0.8
    servers = [ErlangParams(r=2, mu=2 / mean1), ErlangParams(r=3, mu=3 / mean2)]

    calc = MMcQueueingInventoryHeterogeneousErlangCalc(c=2, s_max=4, s=1, policy="lost_sales")
    calc.set_sources(1.0)
    calc.set_servers(servers, theta=0.8)
    res = calc.run(num_levels=200)

    sim = MMcQueueingInventoryHeterogeneousErlangSim(c=2, s_max=4, s=1, policy="lost_sales", seed=42)
    sim.set_sources(1.0)
    sim.set_servers(servers, theta=0.8)
    sim.run(600_000)

    lambda_eff_sim = 1.0 * (1.0 - sim.loss_prob)
    mean_v_sim = sim.mean_in_system / lambda_eff_sim

    assert np.isclose(res.v[0], mean_v_sim, rtol=0.05)
    assert np.isclose(res.loss_prob, sim.loss_prob, rtol=0.05)
    assert np.isclose(res.stockout_prob, sim.stockout_prob, rtol=0.05)
    assert np.allclose(res.stock_distribution, sim.stock_distribution, atol=0.02)


if __name__ == "__main__":
    test_mm1_queueing_inventory_vs_sim()
    test_mm1_queueing_inventory_lost_sales_vs_sim()
    test_mm1_queueing_inventory_general_sS_vs_sim()
    test_mmc_queueing_inventory_vs_sim()
    test_mmc_queueing_inventory_lost_sales_vs_sim()
    test_mm2_queueing_inventory_heterogeneous_vs_sim()
    test_mm2_queueing_inventory_heterogeneous_lost_sales_vs_sim()
    test_mmc_queueing_inventory_heterogeneous_vs_sim()
    test_mmc_queueing_inventory_heterogeneous_lost_sales_vs_sim()
    test_mmc_queueing_inventory_heterogeneous_h2_vs_sim()
    test_mmc_queueing_inventory_heterogeneous_h2_lost_sales_vs_sim()
    test_mm1_queueing_inventory_erlang_replenishment_vs_sim()
    test_mm1_queueing_inventory_erlang_replenishment_lost_sales_vs_sim()
    test_mmc_queueing_inventory_heterogeneous_erlang_vs_sim()
    test_mmc_queueing_inventory_heterogeneous_erlang_lost_sales_vs_sim()
