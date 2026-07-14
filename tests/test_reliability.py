"""
Tests for the reliability models (unreliable servers): each calculator is
checked against a known special case with tight tolerance and against its
paired seeded simulator.
"""

import math

import numpy as np

from most_queue.sim.reliability import (
    MachineRepairSim,
    MM1DisasterRepairSim,
    MM1RetrialUnreliableSim,
    MM1WorkingBreakdownsSim,
    MMcBreakdownsSim,
)
from most_queue.theory.reliability import (
    MachineRepairCalc,
    MM1DisasterRepairCalc,
    MM1RetrialUnreliableCalc,
    MM1WorkingBreakdownsCalc,
    MMcBreakdownsCalc,
)


def _erlang_c_sojourn(lam, mu, c):
    a = lam / mu
    rho = a / c
    p0_inv = sum(a**k / math.factorial(k) for k in range(c)) + a**c / (math.factorial(c) * (1 - rho))
    wq = a**c / (math.factorial(c) * (1 - rho)) / p0_inv / (c * mu - lam)
    return wq + 1 / mu


# ---------------------------------------------------------------- П.1 M/M/c breakdowns
def test_mmc_breakdowns_reduces_to_erlang_c():
    """xi -> 0 must reproduce the exact M/M/c sojourn time."""
    calc = MMcBreakdownsCalc(n=3)
    calc.set_sources(l=1.5)
    calc.set_servers(mu=0.8, xi=1e-9, eta=1.0)
    res = calc.run()

    assert np.isclose(res.v[0], _erlang_c_sojourn(1.5, 0.8, 3), rtol=1e-6)


def test_mmc_breakdowns_availability_binomial():
    """With unlimited repairmen the up-servers marginal is Binomial(c, eta/(xi+eta))."""
    calc = MMcBreakdownsCalc(n=3)
    calc.set_sources(l=0.8)
    calc.set_servers(mu=0.9, xi=0.15, eta=0.5)
    calc.run()

    p_up = 0.5 / 0.65
    binom = [math.comb(3, u) * p_up**u * (1 - p_up) ** (3 - u) for u in range(4)]
    assert np.allclose(calc.up_distribution, binom, atol=1e-8)
    assert np.isclose(calc.availability, p_up, atol=1e-8)


def test_mmc_breakdowns_vs_sim():
    """Truncated CTMC against the seeded event simulator."""
    calc = MMcBreakdownsCalc(n=2)
    calc.set_sources(l=0.9)
    calc.set_servers(mu=0.85, xi=0.1, eta=0.6)
    res = calc.run()

    sim = MMcBreakdownsSim(n=2, seed=42)
    sim.set_sources(l=0.9)
    sim.set_servers(mu=0.85, xi=0.1, eta=0.6)
    sim_res = sim.run(300_000)

    assert np.isclose(res.v[0], sim_res.v[0], rtol=0.05)
    assert np.isclose(calc.availability, sim.availability, rtol=0.02)


# ---------------------------------------------------------------- П.2 machine repair
def test_machine_repair_binomial_special_case():
    """S=0, R=M: machines are independent, failed ~ Binomial(M, xi/(xi+eta))."""
    calc = MachineRepairCalc(n_machines=4, n_repairmen=4, n_spares=0)
    calc.set_sources(xi=0.3, eta=0.7)
    res = calc.run()

    p_fail = 0.3 / 1.0
    binom = [math.comb(4, j) * p_fail**j * (1 - p_fail) ** (4 - j) for j in range(5)]
    assert np.allclose(res.p, binom, atol=1e-12)
    assert np.isclose(res.availability, binom[0], atol=1e-12)


def test_machine_repair_vs_sim():
    """Birth-death solution against the seeded simulator (with warm spares)."""
    calc = MachineRepairCalc(n_machines=5, n_repairmen=2, n_spares=2)
    calc.set_sources(xi=0.25, eta=1.0, xi_s=0.05)
    res = calc.run()

    sim = MachineRepairSim(n_machines=5, n_repairmen=2, n_spares=2, seed=42)
    sim.set_sources(xi=0.25, eta=1.0, xi_s=0.05)
    sim.run(400_000)

    assert np.isclose(res.mean_failed, sim.mean_failed, rtol=0.02)
    assert np.isclose(res.availability, sim.availability, rtol=0.02)


# ---------------------------------------------------------------- П.3 working breakdowns
def test_working_breakdowns_reduces_to_mm1():
    """mu_d = mu must reproduce the M/M/1 mean sojourn 1/(mu - lambda)."""
    calc = MM1WorkingBreakdownsCalc()
    calc.set_sources(l=0.7)
    calc.set_servers(mu=1.0, mu_d=1.0, xi=0.3, eta=0.5)
    res = calc.run()

    assert np.isclose(res.v[0], 1.0 / (1.0 - 0.7), rtol=1e-8)


def test_working_breakdowns_vs_sim():
    """Two-phase CTMC against the seeded simulator; degraded share is xi/(xi+eta)."""
    params = {"mu": 1.2, "mu_d": 0.4, "xi": 0.2, "eta": 0.8}
    calc = MM1WorkingBreakdownsCalc()
    calc.set_sources(l=0.7)
    calc.set_servers(**params)
    res = calc.run()

    sim = MM1WorkingBreakdownsSim(seed=42)
    sim.set_sources(l=0.7)
    sim.set_servers(**params)
    sim_res = sim.run(300_000)

    assert np.isclose(res.v[0], sim_res.v[0], rtol=0.05)
    assert np.isclose(calc.degraded_prob, 0.2 / 1.0, atol=1e-8)
    assert np.isclose(sim.degraded_prob, 0.2 / 1.0, rtol=0.03)


# ---------------------------------------------------------------- П.4 disasters + repair
def test_disaster_repair_down_probability():
    """P(down) = delta / (delta + eta) exactly (renewal argument)."""
    calc = MM1DisasterRepairCalc()
    calc.set_sources(l=0.8, delta=0.15)
    calc.set_servers(mu=1.0, eta=0.5)
    calc.run()

    assert np.isclose(calc.down_prob, 0.15 / 0.65, atol=1e-8)


def test_disaster_repair_vs_sim():
    """Truncated CTMC against the seeded simulator."""
    calc = MM1DisasterRepairCalc()
    calc.set_sources(l=0.9, delta=0.1)
    calc.set_servers(mu=1.0, eta=0.4)
    res = calc.run()

    sim = MM1DisasterRepairSim(seed=42)
    sim.set_sources(l=0.9, delta=0.1)
    sim.set_servers(mu=1.0, eta=0.4)
    sim_res = sim.run(600_000)

    assert np.isclose(res.v[0], sim_res.v[0], rtol=0.05)
    assert np.isclose(calc.down_prob, sim.down_prob, rtol=0.05)


def test_disaster_repair_fast_repair_limit():
    """eta -> infinity approaches the instant-recovery disaster model: the
    queue-length distribution must match the M/M/1-with-disasters geometric
    law pi_k ~ z^k, where z solves mu z^2 - (lambda+mu+delta) z + lambda = 0."""
    lam, mu, delta = 0.9, 1.0, 0.2
    calc = MM1DisasterRepairCalc()
    calc.set_sources(l=lam, delta=delta)
    calc.set_servers(mu=mu, eta=1e7)
    res = calc.run()

    z = ((lam + mu + delta) - math.sqrt((lam + mu + delta) ** 2 - 4 * lam * mu)) / (2 * mu)
    geo = [(1 - z) * z**k for k in range(15)]
    assert np.allclose(res.p[:15], geo, atol=1e-5)


# ---------------------------------------------------------------- П.5 retrial + unreliable
def test_retrial_unreliable_reduces_to_reliable_retrial():
    """xi = 0 must reproduce the exact M/M/1 retrial mean number in system
    L = rho + rho * (lambda + gamma * rho) / (gamma * (1 - rho)) ... checked
    against the closed-form mean orbit of the M/M/1 retrial queue."""
    lam, mu, gamma = 0.6, 1.0, 0.5
    calc = MM1RetrialUnreliableCalc(gamma=gamma)
    calc.set_sources(l=lam)
    calc.set_servers(mu=mu, xi=0.0, eta=1.0)
    calc.run()

    rho = lam / mu
    # Falin-Templeton: E[orbit] = rho^2/(1-rho) + lam*rho/(gamma*(1-rho))
    orbit_exact = rho**2 / (1 - rho) + lam * rho / (gamma * (1 - rho))
    assert np.isclose(calc.mean_orbit, orbit_exact, rtol=1e-8)
    assert np.isclose(calc.availability, 1.0, atol=1e-12)


def test_retrial_unreliable_vs_sim():
    """Orbit-truncated CTMC against the seeded simulator."""
    calc = MM1RetrialUnreliableCalc(gamma=0.7)
    calc.set_sources(l=0.5)
    calc.set_servers(mu=1.0, xi=0.15, eta=0.6)
    res = calc.run()

    sim = MM1RetrialUnreliableSim(gamma=0.7, seed=42)
    sim.set_sources(l=0.5)
    sim.set_servers(mu=1.0, xi=0.15, eta=0.6)
    sim_res = sim.run(300_000)

    assert np.isclose(res.v[0], sim_res.v[0], rtol=0.05)
    assert np.isclose(calc.availability, sim.availability, rtol=0.02)
    assert np.isclose(calc.mean_orbit, sim.mean_orbit, rtol=0.06)


if __name__ == "__main__":
    test_mmc_breakdowns_reduces_to_erlang_c()
    test_mmc_breakdowns_availability_binomial()
    test_mmc_breakdowns_vs_sim()
    test_machine_repair_binomial_special_case()
    test_machine_repair_vs_sim()
    test_working_breakdowns_reduces_to_mm1()
    test_working_breakdowns_vs_sim()
    test_disaster_repair_down_probability()
    test_disaster_repair_vs_sim()
    test_disaster_repair_fast_repair_limit()
    test_retrial_unreliable_reduces_to_reliable_retrial()
    test_retrial_unreliable_vs_sim()
