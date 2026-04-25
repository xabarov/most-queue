"""
Mitzenmacher & Shahout (2025), Table 3.3: S ~ Exp(1), Y|X=x ~ Exp(1/x), mean sojourn E[T].

Values from ``works/queueing_systems_review/SRPT/SPJF.md`` (repro vs ``SizeBasedQsSim``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from most_queue.random.distributions import ExpDistribution
from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.sim.utils.predictor import ExpNoiseSimPredictor
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.srpt import MG1PsjfCalc, MG1SjfCalc, MG1SpjfCalc, MG1SrptCalc
from most_queue.theory.srpt.utils.predictor import ExpNoisePredictor

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])
NUM_OF_JOBS_SRPT = int(params.get("num_of_jobs_srpt", int(params["num_of_jobs"])))

# Table 3.3 -- mean sojourn E[T] (Mitzenmacher-Shahout 2025, via SPJF.md)
PAPER_ET: dict[float, dict[str, float]] = {
    0.5: {"FCFS": 2.000, "SJF": 1.713, "SPJF": 1.795, "PSJF": 1.531, "PSPJF": 1.664, "SRPT": 1.425, "SPRPT": 1.653},
    0.8: {"FCFS": 5.000, "SJF": 2.882, "SPJF": 3.376, "PSJF": 2.659, "PSPJF": 3.194, "SRPT": 2.353, "SPRPT": 3.117},
    0.9: {"FCFS": 10.000, "SJF": 4.462, "SPJF": 5.527, "PSJF": 4.130, "PSPJF": 5.285, "SRPT": 3.642, "SPRPT": 5.131},
    0.95: {"FCFS": 20.000, "SJF": 6.264, "SPJF": 8.654, "PSJF": 6.265, "PSPJF": 8.617, "SRPT": 5.541, "SPRPT": 8.322},
    0.99: {"FCFS": 100.000, "SJF": 18.45, "SPJF": 29.05, "PSJF": 18.96, "PSPJF": 29.38, "SRPT": 17.63, "SPRPT": 28.73},
}

EXP_SERVICE_RATE = 1.0  # Exp(rate): mean service = 1


def _sim_seed(rho: float, discipline: str) -> int:
    salt = {"FCFS": 1, "SJF": 2, "SPJF": 3, "PSJF": 4, "PSPJF": 5, "SRPT": 6, "SPRPT": 7}[discipline]
    return 50_000_000 + int(rho * 1_000_000) + salt


def _run_sim_et(discipline: str, rho: float, n_jobs: int) -> float:
    lam = rho  # E[S]=1 => rho = lambda
    sim = SizeBasedQsSim(1, discipline=discipline, verbose=False)  # type: ignore[arg-type]
    sim.generator = np.random.default_rng(_sim_seed(rho, discipline))
    sim.set_servers(EXP_SERVICE_RATE, "M")
    sim.set_sources(lam, "M")
    if discipline in ("SPJF", "PSPJF", "SPRPT"):
        sim.set_predictor(ExpNoiseSimPredictor())
    return float(sim.run(n_jobs).v[0])


def _assert_ordering(et: dict[str, float]) -> None:
    """SRPT is optimal (lowest E[T]); all work-conserving policies beat FCFS here."""
    t_srpt = et["SRPT"]
    assert t_srpt <= et["FCFS"] + MOMENTS_ATOL
    for disc in ("SJF", "SPJF", "PSJF", "PSPJF", "SPRPT"):
        assert t_srpt <= et[disc] + MOMENTS_ATOL


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_predictions_table_sim_vs_paper_standard_rho(rho: float):
    """Simulated E[T] for all disciplines matches Table 3.3 within project tolerances."""
    paper = PAPER_ET[rho]
    n = NUM_OF_JOBS_SRPT
    sim_ets: dict[str, float] = {}
    for disc in ("FCFS", "SJF", "SPJF", "PSJF", "PSPJF", "SRPT", "SPRPT"):
        sim_ets[disc] = _run_sim_et(disc, rho, n)

    _assert_ordering(sim_ets)

    for disc, expected in paper.items():
        assert np.isclose(
            sim_ets[disc], expected, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL
        ), f"rho={rho} {disc}: sim E[T]={sim_ets[disc]:.4f} paper={expected}"


@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.95, 0.99])
def test_predictions_table_sim_vs_paper_high_rho(rho: float):
    """High-utilisation (rho=0.95, 0.99): E[T] ordering and paper table match with 1M jobs."""
    paper = PAPER_ET[rho]
    n_jobs = 1_000_000
    sim_ets: dict[str, float] = {}
    for disc in ("FCFS", "SJF", "SPJF", "PSJF", "PSPJF", "SRPT", "SPRPT"):
        sim_ets[disc] = _run_sim_et(disc, rho, n_jobs)

    _assert_ordering(sim_ets)

    for disc, expected in paper.items():
        assert np.isclose(
            sim_ets[disc], expected, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL
        ), f"rho={rho} {disc}: sim E[T]={sim_ets[disc]:.4f} paper={expected}"


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_predictions_table_theory_fcfs_exact(rho: float):
    """M/M/1 FCFS: E[T] = 1/(1-rho) with mean service 1."""
    lam = rho
    fcfs = MG1Calc()
    fcfs.set_sources(lam)
    fcfs.set_servers(ExpDistribution.calc_theory_moments(EXP_SERVICE_RATE, 5))
    et_fcfs = fcfs.run().v[0]
    assert np.isclose(et_fcfs, 1.0 / (1.0 - rho), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_predictions_table_theory_ordering_exp1(rho: float):
    """Exp(1) service: SRPT optimal; SJF/PSJF/SPJF each <= FCFS (PSJF vs SJF not ordered)."""
    lam = rho
    moments = ExpDistribution.calc_theory_moments(EXP_SERVICE_RATE, 5)

    fcfs_calc = MG1Calc()
    fcfs_calc.set_sources(lam)
    fcfs_calc.set_servers(moments)
    et_fcfs = fcfs_calc.run().v[0]

    sjf_calc = MG1SjfCalc()
    sjf_calc.set_sources(lam)
    sjf_calc.set_servers(EXP_SERVICE_RATE, "M")
    et_sjf = sjf_calc.run().v[0]

    psjf_calc = MG1PsjfCalc()
    psjf_calc.set_sources(lam)
    psjf_calc.set_servers(EXP_SERVICE_RATE, "M")
    et_psjf = psjf_calc.run().v[0]

    srpt_calc = MG1SrptCalc()
    srpt_calc.set_sources(lam)
    srpt_calc.set_servers(EXP_SERVICE_RATE, "M")
    et_srpt = srpt_calc.run().v[0]

    assert et_srpt <= et_psjf + 1e-8
    assert et_srpt <= et_sjf + 1e-8
    assert et_sjf <= et_fcfs + 1e-8
    assert et_psjf <= et_fcfs + 1e-8

    spjf_calc = MG1SpjfCalc()
    spjf_calc.set_sources(lam)
    spjf_calc.set_servers(EXP_SERVICE_RATE, "M")
    spjf_calc.set_predictor(ExpNoisePredictor())
    et_spjf = spjf_calc.run().v[0]
    assert et_srpt <= et_spjf + 1e-8
    assert et_spjf <= et_fcfs + 1e-8
