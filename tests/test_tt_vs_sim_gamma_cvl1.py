"""
Validation for CV<1 using Gamma simulation as proxy.

For CV<1 a real H2 distribution does not exist, but Takahashi–Takami calculations
can use complex-fit H2 approximation (complex parameters).

Since QsSim cannot generate H2 with complex parameters, we validate "closeness"
by simulating the corresponding Gamma distributions with the same mean and CV.

This is an approximation-on-approximation check (H2 complex-fit ≈ Gamma),
so tolerances are intentionally looser than for the CV>=1 "same H2 params" tests.
"""

import os

import numpy as np
import pytest
import yaml

from most_queue.io.tables import probs_print
from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc

cur_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(cur_dir, "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

N = int(params.get("num_of_channels", 3))
ARRIVAL_RATE = float(params["arrival"]["rate"])
RHO = float(params.get("utilization_factor", 0.7))

# Pick a CV<1 scenario (can be overridden by editing this test).
ARRIVAL_CV = 0.8
SERVICE_CV = 0.8

# Slow (but stable) simulation.
NUM_OF_JOBS = 1_000_000

# Tolerances: H2 complex-fit is only an approximation of Gamma when CV<1.
PROBS_ATOL = 0.06
PROBS_RTOL = 0.20


def _h2_params_complex_fit(mean: float, cv: float) -> H2Params:
    h2 = H2Distribution.get_params_by_mean_and_cv(mean, cv, is_clx=True)
    return H2Params(p1=h2.p1, mu1=h2.mu1, mu2=h2.mu2)


@pytest.mark.slow
def test_gamma_vs_h2m_tt_cvl1_probabilities():
    """
    Gamma/M/n simulation vs H2/M/n TT (complex-fit H2 approximation for arrivals).
    """
    a1 = 1.0 / ARRIVAL_RATE
    b_mean = RHO * N * a1

    # Simulation: Gamma arrivals with CV<1, exponential service.
    gamma_arr = GammaDistribution.get_params_by_mean_and_cv(a1, ARRIVAL_CV)
    qs = QsSim(N)
    qs.set_sources(gamma_arr, "Gamma")
    qs.set_servers(1.0 / b_mean, "M")
    sim = qs.run(NUM_OF_JOBS)

    # Calculation: H2/M/n with complex-fit H2 params for the same mean & CV.
    h2_arr = _h2_params_complex_fit(a1, ARRIVAL_CV)
    calc = H2MnCalc(n=N)
    calc.set_sources(h2_arr)
    calc.set_servers(b_mean)
    res = calc.run()

    print("\nGamma/M/n simulation vs H2/M/n (TT, H2 complex-fit) at CV<1")
    print(f"n={N}, rho={RHO:.2f}, arrival: mean={a1:.4f}, cv={ARRIVAL_CV:.2f}, service: M mean={b_mean:.4f}")
    print(f"TT iterations: {calc.num_of_iter_}")
    probs_print(sim.p, res.p, 10)

    assert np.isclose(sum(res.p), 1.0, atol=1e-9, rtol=1e-8)
    assert np.all(np.isfinite(res.p))
    assert np.allclose(sim.p[:10], res.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL)


@pytest.mark.slow
def test_gamma_vs_h2h2_tt_cvl1_probabilities():
    """
    Gamma/Gamma/n simulation vs H2/H2/n TT (complex-fit H2 approximation for both).
    """
    a1 = 1.0 / ARRIVAL_RATE
    b_mean = RHO * N * a1

    gamma_arr = GammaDistribution.get_params_by_mean_and_cv(a1, ARRIVAL_CV)
    gamma_srv = GammaDistribution.get_params_by_mean_and_cv(b_mean, SERVICE_CV)

    qs = QsSim(N)
    qs.set_sources(gamma_arr, "Gamma")
    qs.set_servers(gamma_srv, "Gamma")
    sim = qs.run(NUM_OF_JOBS)

    h2_arr = _h2_params_complex_fit(a1, ARRIVAL_CV)
    h2_srv = _h2_params_complex_fit(b_mean, SERVICE_CV)

    calc = HkHkNCalc(n=N, k=2)
    calc.set_sources(u=[h2_arr.p1, 1.0 - h2_arr.p1], lam=[h2_arr.mu1, h2_arr.mu2])
    calc.set_servers(y=[h2_srv.p1, 1.0 - h2_srv.p1], mu=[h2_srv.mu1, h2_srv.mu2])
    res = calc.run()

    print("\nGamma/Gamma/n simulation vs H2/H2/n (TT, H2 complex-fit) at CV<1")
    print(
        f"n={N}, rho={RHO:.2f}, arrival: mean={a1:.4f}, cv={ARRIVAL_CV:.2f}, "
        f"service: mean={b_mean:.4f}, cv={SERVICE_CV:.2f}"
    )
    print(f"TT iterations: {calc.num_of_iter_}")
    probs_print(sim.p, res.p, 10)

    assert np.isclose(sum(res.p), 1.0, atol=1e-9, rtol=1e-8)
    assert np.all(np.isfinite(res.p))
    assert np.allclose(sim.p[:10], res.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL)


if __name__ == "__main__":
    # Run directly to see printed tables without pytest (-s).
    test_gamma_vs_h2m_tt_cvl1_probabilities()
    test_gamma_vs_h2h2_tt_cvl1_probabilities()
