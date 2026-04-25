"""
Reproduce Mitzenmacher & Shahout (2025), Table 3.3: S ~ Exp(1), Y|X=x ~ Exp(1/x).

Theory vs simulation vs paper E[T]. Saves ``examples/srpt_table.png``.
Set env ``SRPT_TABLE_QUICK=1`` for smaller job counts (faster, less accurate).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init

from most_queue.random.distributions import ExpDistribution
from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.sim.utils.predictor import ExpNoiseSimPredictor
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.srpt import MG1PsjfCalc, MG1SjfCalc, MG1SpjfCalc, MG1SrptCalc
from most_queue.theory.srpt.utils.predictor import ExpNoisePredictor

init(autoreset=True)

EXP_SERVICE_RATE = 1.0

PAPER_ET: dict[float, dict[str, float]] = {
    0.5: {"FCFS": 2.000, "SJF": 1.713, "SPJF": 1.795, "PSJF": 1.531, "PSPJF": 1.664, "SRPT": 1.425, "SPRPT": 1.653},
    0.8: {"FCFS": 5.000, "SJF": 2.882, "SPJF": 3.376, "PSJF": 2.659, "PSPJF": 3.194, "SRPT": 2.353, "SPRPT": 3.117},
    0.9: {"FCFS": 10.000, "SJF": 4.462, "SPJF": 5.527, "PSJF": 4.130, "PSPJF": 5.285, "SRPT": 3.642, "SPRPT": 5.131},
    0.95: {"FCFS": 20.000, "SJF": 6.264, "SPJF": 8.654, "PSJF": 6.265, "PSPJF": 8.617, "SRPT": 5.541, "SPRPT": 8.322},
    0.99: {"FCFS": 100.000, "SJF": 18.45, "SPJF": 29.05, "PSJF": 18.96, "PSPJF": 29.38, "SRPT": 17.63, "SPRPT": 28.73},
}

DISCIPLINES: tuple[str, ...] = ("FCFS", "SJF", "SPJF", "PSJF", "PSPJF", "SRPT", "SPRPT")


def _sim_seed(rho: float, discipline: str) -> int:
    salt = {"FCFS": 1, "SJF": 2, "SPJF": 3, "PSJF": 4, "PSPJF": 5, "SRPT": 6, "SPRPT": 7}[discipline]
    return 50_000_000 + int(rho * 1_000_000) + salt


def _n_jobs(rho: float, quick: bool) -> int:
    if quick:
        return 20_000 if rho <= 0.9 else 50_000
    if rho <= 0.9:
        return 300_000
    if rho <= 0.95:
        return 500_000
    return 1_000_000


def _theory_et(discipline: str, rho: float) -> float | None:
    lam = rho
    moments = ExpDistribution.calc_theory_moments(EXP_SERVICE_RATE, 5)
    if discipline == "FCFS":
        c = MG1Calc()
        c.set_sources(lam)
        c.set_servers(moments)
        return float(c.run().v[0])
    if discipline == "SJF":
        c = MG1SjfCalc()
        c.set_sources(lam)
        c.set_servers(EXP_SERVICE_RATE, "M")
        return float(c.run().v[0])
    if discipline == "PSJF":
        c = MG1PsjfCalc()
        c.set_sources(lam)
        c.set_servers(EXP_SERVICE_RATE, "M")
        return float(c.run().v[0])
    if discipline == "SRPT":
        c = MG1SrptCalc()
        c.set_sources(lam)
        c.set_servers(EXP_SERVICE_RATE, "M")
        return float(c.run().v[0])
    if discipline == "SPJF":
        c = MG1SpjfCalc()
        c.set_sources(lam)
        c.set_servers(EXP_SERVICE_RATE, "M")
        c.set_predictor(ExpNoisePredictor())
        return float(c.run().v[0])
    return None


def _sim_et(discipline: str, rho: float, n_jobs: int, verbose: bool) -> float:
    lam = rho
    sim = SizeBasedQsSim(1, discipline=discipline, verbose=verbose)  # type: ignore[arg-type]
    sim.generator = np.random.default_rng(_sim_seed(rho, discipline))
    sim.set_servers(EXP_SERVICE_RATE, "M")
    sim.set_sources(lam, "M")
    if discipline in ("SPJF", "PSPJF", "SPRPT"):
        sim.set_predictor(ExpNoiseSimPredictor())
    return float(sim.run(n_jobs).v[0])


def _color_for_rel_err(rel: float) -> str:
    if rel < 0.05:
        return Fore.GREEN
    if rel < 0.15:
        return Fore.YELLOW
    return Fore.RED


def main() -> int:
    quick = os.environ.get("SRPT_TABLE_QUICK", "").lower() in ("1", "true", "yes")
    verbose = os.environ.get("SRPT_TABLE_VERBOSE", "").lower() in ("1", "true", "yes")
    here = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(here, "srpt_table.png")

    print("Mitzenmacher & Shahout (2025) Table 3.3 -- E[T] (mean sojourn)")
    if quick:
        print(f"{Fore.CYAN}SRPT_TABLE_QUICK=1: using smaller N for simulation{Style.RESET_ALL}")

    sim_curves: dict[str, list[float]] = {d: [] for d in DISCIPLINES}

    for rho in sorted(PAPER_ET):
        paper = PAPER_ET[rho]
        n = _n_jobs(rho, quick)
        print(f"\n--- rho = {rho} (sim N = {n}) ---")
        for disc in DISCIPLINES:
            the = _theory_et(disc, rho)
            sim = _sim_et(disc, rho, n, verbose=verbose)
            sim_curves[disc].append(sim)
            exp = paper[disc]
            rel = abs(sim - exp) / max(exp, 1e-12)
            col = _color_for_rel_err(rel)
            the_s = f"{the:.4f}" if the is not None else "--"
            print(
                f"  {disc:6s}  paper={exp:8.3f}  theory={the_s:>8}  sim={sim:8.4f}  "
                f"{col}|sim-paper|/paper={rel*100:.2f}%{Style.RESET_ALL}"
            )

    rhos = sorted(PAPER_ET)
    plt.figure(figsize=(9, 6))
    for disc in DISCIPLINES:
        plt.semilogy(rhos, sim_curves[disc], marker="o", label=disc)
    plt.xlabel("rho")
    plt.ylabel("E[T] (sim)")
    plt.title("Table 3.3 scenario: sim E[T] vs rho (log scale)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
