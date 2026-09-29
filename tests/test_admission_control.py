"""
Cross-validate the exact M/M/1 deadline-admission-control QBD-free series
calculator (EPIC-034) against discrete-event simulation.
"""

import numpy as np

from most_queue.sim.admission_control import MM1DeadlineAdmissionControlSim
from most_queue.theory.admission_control import MM1DeadlineAdmissionControlCalc


def test_mm1_deadline_admission_control_vs_sim():
    calc = MM1DeadlineAdmissionControlCalc()
    calc.set_sources(1.2)
    calc.set_servers(1.0)
    calc.set_deadline(0.5)
    res = calc.run(num=2)

    sim = MM1DeadlineAdmissionControlSim(seed=42)
    sim.set_sources(1.2)
    sim.set_servers(1.0)
    sim.set_deadline(0.5)
    sres = sim.run(800_000)

    assert np.isclose(res.loss_prob, sres.loss_prob, rtol=0.03)
    assert np.isclose(res.v[0], sres.v[0], rtol=0.03)
    assert np.isclose(res.v[1], sres.v[1], rtol=0.05)


def test_mm1_deadline_admission_control_unconditional_workload_vs_sim():
    lam, mu, theta = 0.8, 1.0, 1.0
    calc = MM1DeadlineAdmissionControlCalc()
    calc.set_sources(lam)
    calc.set_servers(mu)
    calc.set_deadline(theta)
    u_mean = calc.get_u_moments(num=1)[0]

    sim = MM1DeadlineAdmissionControlSim(seed=7)
    sim.set_sources(lam)
    sim.set_servers(mu)
    sim.set_deadline(theta)
    sim.run(800_000)

    assert np.isclose(u_mean, sim.mean_workload, rtol=0.03)


if __name__ == "__main__":
    test_mm1_deadline_admission_control_vs_sim()
    test_mm1_deadline_admission_control_unconditional_workload_vs_sim()
