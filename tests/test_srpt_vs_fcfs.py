"""
Theory sanity: E[T^SRPT] <= E[T^PSJF] <= E[T^SJF] <= E[T^FCFS] for M/G/1 (stable rho).
"""

from __future__ import annotations

import os

import yaml

from most_queue.random.distributions import ExpDistribution, GammaDistribution, H2Distribution
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.srpt import MG1PsjfCalc, MG1SjfCalc, MG1SrptCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV_GE1 = float(params["service"]["cv_ge1"])
SERVICE_TIME_CV_LT1 = float(params["service"]["cv_lt1"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_CHANNELS = 1


def _et_srpt_psjf_sjf_fcfs(h2_or_gamma: str, rho: float) -> tuple[float, float, float, float]:
    """Return (E[T_SRPT], E[T_PSJF], E[T_SJF], E[T_FCFS]) for given rho."""
    b1 = rho * NUM_OF_CHANNELS / ARRIVAL_RATE

    if h2_or_gamma == "H":
        dist_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV_GE1)
        notation = "H"
    else:
        dist_params = GammaDistribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV_LT1)
        notation = "Gamma"

    srpt = MG1SrptCalc()
    srpt.set_sources(ARRIVAL_RATE)
    srpt.set_servers(dist_params, notation)
    et_srpt = srpt.run().v[0]

    psjf = MG1PsjfCalc()
    psjf.set_sources(ARRIVAL_RATE)
    psjf.set_servers(dist_params, notation)
    et_psjf = psjf.run().v[0]

    sjf = MG1SjfCalc()
    sjf.set_sources(ARRIVAL_RATE)
    sjf.set_servers(dist_params, notation)
    et_sjf = sjf.run().v[0]

    if notation == "H":
        b = H2Distribution.calc_theory_moments(dist_params, 5)
    else:
        b = GammaDistribution.calc_theory_moments(dist_params, 5)
    fcfs = MG1Calc()
    fcfs.set_sources(ARRIVAL_RATE)
    fcfs.set_servers(b)
    et_fcfs = fcfs.run().v[0]

    return et_srpt, et_psjf, et_sjf, et_fcfs


def _assert_srpt_optimal_and_fcfs_worst(t_srpt: float, t_psjf: float, t_sjf: float, t_fcfs: float) -> None:
    """SRPT minimizes E[T]; all listed policies beat FCFS; PSJF vs SJF not universally ordered."""
    assert t_srpt <= t_psjf + 1e-8
    assert t_srpt <= t_sjf + 1e-8
    assert t_srpt <= t_fcfs + 1e-8
    assert t_psjf <= t_fcfs + 1e-8
    assert t_sjf <= t_fcfs + 1e-8


def test_ordering_h2_rhos():
    """H2 service: SRPT is optimal and all policies beat FCFS; PSJF <= SJF."""
    for rho in (0.5, 0.7, 0.9):
        t_srpt, t_psjf, t_sjf, t_fcfs = _et_srpt_psjf_sjf_fcfs("H", rho)
        _assert_srpt_optimal_and_fcfs_worst(t_srpt, t_psjf, t_sjf, t_fcfs)
        assert t_psjf <= t_sjf + 1e-8


def test_ordering_gamma_rhos():
    """Gamma service: SRPT is optimal and all policies beat FCFS."""
    for rho in (0.5, 0.7, 0.9):
        t_srpt, t_psjf, t_sjf, t_fcfs = _et_srpt_psjf_sjf_fcfs("Gamma", rho)
        _assert_srpt_optimal_and_fcfs_worst(t_srpt, t_psjf, t_sjf, t_fcfs)


def test_ordering_exp1_numeric():
    """M/M/1: SRPT matches known table at rho=0.5 (Mitzenmacher-Shahout 2025)."""
    rho = 0.5
    lam = rho
    mu_rate = 1.0  # Exp(1) service, mean 1

    srpt = MG1SrptCalc()
    srpt.set_sources(lam)
    srpt.set_servers(mu_rate, "M")
    assert abs(srpt.run().v[0] - 1.425) < 0.02

    b = ExpDistribution.calc_theory_moments(mu_rate, 5)
    fcfs = MG1Calc()
    fcfs.set_sources(lam)
    fcfs.set_servers(b)
    assert abs(fcfs.run().v[0] - 2.0) < 1e-9
