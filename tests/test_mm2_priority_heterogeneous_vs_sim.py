"""
Cross-validate the exact M/M/2 priority-heterogeneous-servers CTMC (EPIC-025)
against discrete-event simulation.
"""

import numpy as np

from most_queue.sim.priority_heterogeneous import MM2PriorityHeterogeneousSim
from most_queue.theory.priority.preemptive.mm2_heterogeneous import MM2PriorityHeterogeneousCalc


def test_mm2_priority_heterogeneous_vs_sim():
    calc = MM2PriorityHeterogeneousCalc(truncation=50)
    calc.set_sources([0.4, 0.3])
    calc.set_servers(1.5, 0.5)
    res = calc.run()

    sim = MM2PriorityHeterogeneousSim(seed=42)
    sim.set_sources([0.4, 0.3])
    sim.set_servers(1.5, 0.5)
    sim_res = sim.run(400_000)

    assert np.isclose(res.v[0][0], sim_res.v[0][0], rtol=0.03)
    assert np.isclose(res.v[1][0], sim_res.v[1][0], rtol=0.03)
    assert np.isclose(res.w[0][0], sim_res.w[0][0], rtol=0.05)
    assert np.isclose(res.w[1][0], sim_res.w[1][0], rtol=0.05)


if __name__ == "__main__":
    test_mm2_priority_heterogeneous_vs_sim()
