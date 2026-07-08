"""
Tests for M/G/1 Processor Sharing and LCFS-PR calculators
against dedicated single-server work-tracking simulators.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.sim.single_server_disciplines import LcfsPRSim, ProcessorSharingSim
from most_queue.theory.fifo.mg1_lcfs_pr import MG1LcfsPrCalc
from most_queue.theory.fifo.mg1_ps import MG1PSCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

SERVICE_MEAN = UTILIZATION_FACTOR / ARRIVAL_RATE


def _gamma_moments(mean: float, cv: float, num: int = 4) -> list[float]:
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean, cv)
    return GammaDistribution.calc_theory_moments(gamma_params, num)


def test_mg1_ps_vs_sim():
    """
    M/G/1 PS vs simulation: geometric state probabilities (insensitivity,
    checked with Gamma service, CV != 1), mean sojourn and sharing delay.
    """
    b = _gamma_moments(SERVICE_MEAN, SERVICE_CV)

    calc = MG1PSCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc_results = calc.run()

    sim = ProcessorSharingSim()
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(GammaDistribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV), "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG

    print(f"PS: v1 sim={sim_results.v[0]:.4f}, calc={calc_results.v[0]:.4f}")
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.isclose(sim_results.w[0], calc_results.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    # uniform slowdown: E[V] = b1 / (1 - rho) exactly
    assert np.isclose(calc_results.v[0], b[0] / (1 - UTILIZATION_FACTOR), rtol=1e-12), ERROR_MSG
    assert np.isclose(calc.get_mean_slowdown(), 1.0 / (1 - UTILIZATION_FACTOR), rtol=1e-12), ERROR_MSG


def test_mg1_lcfs_pr_vs_sim():
    """
    M/G/1 LCFS-PR vs simulation: sojourn time moments equal the busy period
    moments (Takacs), state probabilities are geometric.
    """
    b = _gamma_moments(SERVICE_MEAN, SERVICE_CV)

    calc = MG1LcfsPrCalc()
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(b)
    calc_results = calc.run()

    sim = LcfsPRSim()
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(GammaDistribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV), "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    probs_print(sim_results.p, calc_results.p, size=10)
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG

    print(f"LCFS-PR: v sim={[f'{m:.3f}' for m in sim_results.v[:3]]}, calc={[f'{m:.3f}' for m in calc_results.v]}")
    assert np.allclose(sim_results.v[:2], calc_results.v[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    # mean sojourn equals b1 / (1 - rho) exactly (same as PS)
    assert np.isclose(calc_results.v[0], b[0] / (1 - UTILIZATION_FACTOR), rtol=1e-12), ERROR_MSG


if __name__ == "__main__":
    test_mg1_ps_vs_sim()
    test_mg1_lcfs_pr_vs_sim()
