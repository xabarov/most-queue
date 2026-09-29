"""
Unit tests for the machine repair problem with two heterogeneous repairmen
(most_queue.theory.reliability.machine_repair_heterogeneous), EPIC-023.
"""

import numpy as np
import pytest

from most_queue.theory.reliability import MachineRepairCalc, MachineRepairHeterogeneousCalc


@pytest.mark.parametrize(
    "n_machines,n_spares,xi,eta,xi_s",
    [
        (5, 2, 0.25, 1.0, 0.05),
        (4, 0, 0.3, 0.8, None),
        (3, 1, 0.5, 0.6, 0.5),
    ],
)
def test_reduces_to_homogeneous_when_rates_equal(n_machines, n_spares, xi, eta, xi_s):
    """eta_a = eta_b must exactly reproduce MachineRepairCalc(n_repairmen=2, ...)."""
    het = MachineRepairHeterogeneousCalc(n_machines=n_machines, n_spares=n_spares)
    het.set_sources(xi=xi, eta_a=eta, eta_b=eta, xi_s=xi_s)
    res_het = het.run()

    homo = MachineRepairCalc(n_machines=n_machines, n_repairmen=2, n_spares=n_spares)
    homo.set_sources(xi=xi, eta=eta, xi_s=xi_s)
    res_homo = homo.run()

    assert np.allclose(res_het.p, res_homo.p, atol=1e-10)
    assert np.isclose(res_het.availability, res_homo.availability, atol=1e-10)
    assert np.isclose(res_het.mean_failed, res_homo.mean_failed, atol=1e-10)
    assert np.isclose(res_het.repairmen_utilization, res_homo.repairmen_utilization, atol=1e-10)
    # Fastest-first tie-breaking makes A individually busier than B even at
    # equal rates, but their average must match the homogeneous utilization.
    assert np.isclose((res_het.utilization_a + res_het.utilization_b) / 2.0, res_homo.repairmen_utilization, atol=1e-10)


@pytest.mark.parametrize(
    "n_machines,n_spares,xi,eta_a,eta_b,xi_s",
    [
        (6, 2, 0.3, 1.5, 0.5, 0.1),
        (8, 0, 0.15, 2.0, 0.3, None),
        (4, 1, 0.2, 1.0, 0.9, 0.05),
    ],
)
def test_flow_balance_invariant(n_machines, n_spares, xi, eta_a, eta_b, xi_s):
    """
    Steady-state flow balance: the long-run failure rate must equal the
    long-run repair-completion rate (eta_a * P(A busy) + eta_b * P(B busy)).
    A strong, model-independent sanity check on the CTMC construction.
    """
    calc = MachineRepairHeterogeneousCalc(n_machines=n_machines, n_spares=n_spares)
    calc.set_sources(xi=xi, eta_a=eta_a, eta_b=eta_b, xi_s=xi_s)
    res = calc.run()

    repair_throughput = eta_a * res.utilization_a + eta_b * res.utilization_b
    assert np.isclose(res.failure_throughput, repair_throughput, rtol=1e-9)


def test_set_sources_normalises_rate_order():
    """set_sources(eta_a, eta_b) and set_sources(eta_b, eta_a) give the same result."""
    calc1 = MachineRepairHeterogeneousCalc(n_machines=5, n_spares=1)
    calc1.set_sources(xi=0.2, eta_a=1.5, eta_b=0.4)
    res1 = calc1.run()

    calc2 = MachineRepairHeterogeneousCalc(n_machines=5, n_spares=1)
    calc2.set_sources(xi=0.2, eta_a=0.4, eta_b=1.5)
    res2 = calc2.run()

    assert np.allclose(res1.p, res2.p)
    assert np.isclose(res1.utilization_a, res2.utilization_a)
    assert np.isclose(res1.utilization_b, res2.utilization_b)


def test_distribution_is_a_valid_probability_vector():
    calc = MachineRepairHeterogeneousCalc(n_machines=6, n_spares=2)
    calc.set_sources(xi=0.3, eta_a=1.5, eta_b=0.5, xi_s=0.1)
    res = calc.run()

    assert np.isclose(sum(res.p), 1.0, atol=1e-10)
    assert all(x >= -1e-12 for x in res.p)
    assert np.isclose(res.availability, sum(res.p[:3]), atol=1e-10)  # S=2 -> j in {0,1,2}


if __name__ == "__main__":
    for args in [(5, 2, 0.25, 1.0, 0.05), (4, 0, 0.3, 0.8, None), (3, 1, 0.5, 0.6, 0.5)]:
        test_reduces_to_homogeneous_when_rates_equal(*args)
    for args in [(6, 2, 0.3, 1.5, 0.5, 0.1), (8, 0, 0.15, 2.0, 0.3, None), (4, 1, 0.2, 1.0, 0.9, 0.05)]:
        test_flow_balance_invariant(*args)
    test_set_sources_normalises_rate_order()
    test_distribution_is_a_valid_probability_vector()
    print("all machine repair heterogeneous tests passed")
