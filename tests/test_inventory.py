"""
Cross-validate the M/M/1 queueing-inventory QBD calculator (EPIC-024) against
discrete-event simulation.
"""

import numpy as np

from most_queue.sim.inventory import MM1QueueingInventorySim
from most_queue.theory.inventory import MM1QueueingInventoryCalc


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


if __name__ == "__main__":
    test_mm1_queueing_inventory_vs_sim()
