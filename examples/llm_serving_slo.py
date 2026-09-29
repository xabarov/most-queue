"""
LLM-inference serving TTFT SLO: composite example for the SLA layer
(most_queue.theory.utils.sla).

Models a GPU batch-inference server as MAP/PH/1: bursty (MMPP-2) request
arrivals -- LLM-serving traffic is autocorrelated, not Poisson -- and a
PH-fitted batch-inference service time (mean + variability of a batched
GPU forward pass). For a range of loads, computes the deadline-violation
probability P(TTFT > D) from the exact MapPh1Calc moments via the SLA layer.
The curve qualitatively reproduces the shape reported by SLO-aware
LLM-serving papers (QUARTZ, Hermes, TailGuard/TPDS 2025): violation
probability stays low, then rises sharply as rho approaches the stability
boundary -- see docs/research/sla-deadline-queueing-2026.md.

A second section extends this to two SLO tiers (premium / free) sharing the
same server. MapPh1PriorityCalc (the MAP+PH priority calculator) only
returns the mean waiting time per class -- not enough raw moments for the
SLA tail layer, see docs/research/sla-deadline-queueing-2026.md -- so this
section uses Poisson-split arrivals with MG1NonPreemptiveCalc instead (which
does return full moments per class), still on the same PH-fitted
batch-inference service distribution. This demonstrates the *priority*
dimension of SLO-tiering; the bursty-arrival case is covered by the MAP/PH/1
curve above.

Saves examples/llm_serving_slo.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init

from most_queue.random.distributions import H2Distribution
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.theory.matrix.map_ph1 import MapPh1Calc
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptiveCalc
from most_queue.theory.utils.sla import deadline_violation_prob, slo_quantile

init(autoreset=True)

# Batch-inference service: mean 1.0 time unit, cv=1.5 (batching variability).
SERVICE_MEAN = 1.0
SERVICE_CV = 1.5

# MMPP-2 burstiness shape (arbitrary units, scaled below to hit a target rho).
BASE_RATES = [2.0, 0.4]
PHASE_Q = np.array([[-0.2, 0.2], [0.3, -0.3]])

# TTFT deadline: 3x mean service time -- a representative "firm-ish" SLO.
DEADLINE = 3.0 * SERVICE_MEAN

RHOS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95]


def _service_ph():
    h2 = H2Distribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV)
    return PHDistribution.from_h2(h2)


def _mmpp_for_rho(rho: float):
    """Scale MMPP-2 rates so the fundamental arrival rate hits rho * (1/SERVICE_MEAN)."""
    base = MAP.mmpp(BASE_RATES, PHASE_Q)
    base_lambda = MAP.arrival_rate(base)
    target_lambda = rho / SERVICE_MEAN
    scale = target_lambda / base_lambda
    return MAP.mmpp([r * scale for r in BASE_RATES], PHASE_Q)


def violation_curve() -> list[tuple[float, float, float]]:
    """(rho, P(TTFT > DEADLINE), SLO quantile D_0.05) for each rho in RHOS."""
    ph_srv = _service_ph()
    rows = []
    for rho in RHOS:
        mmpp = _mmpp_for_rho(rho)
        calc = MapPh1Calc()
        calc.set_sources(mmpp)
        calc.set_servers(ph_srv)
        w = calc.run().w
        p_violation = deadline_violation_prob(w, DEADLINE)
        d95 = slo_quantile(w, 0.05)
        rows.append((rho, p_violation, d95))
    return rows


def priority_tiers_at(rho: float, premium_share: float = 0.3) -> dict[str, float]:
    """Poisson-split premium/free SLO tiers sharing the same batch-inference server."""
    total_lambda = rho / SERVICE_MEAN
    lambdas = [premium_share * total_lambda, (1.0 - premium_share) * total_lambda]

    ph_srv = _service_ph()
    b = PHDistribution.calc_theory_moments(ph_srv, 4)

    calc = MG1NonPreemptiveCalc()
    calc.set_sources(lambdas)
    calc.set_servers([b, b])
    results = calc.run()

    return {
        "premium": deadline_violation_prob(results.w[0], DEADLINE),
        "free": deadline_violation_prob(results.w[1], DEADLINE),
    }


def main() -> None:
    print(Fore.GREEN + "LLM-inference serving TTFT SLO -- MAP(MMPP-2)/PH/1")
    print(f"Batch-inference service: mean={SERVICE_MEAN}, cv={SERVICE_CV}, deadline D={DEADLINE}\n")

    rows = violation_curve()
    print(f"{'rho':>6} {'P(TTFT > D)':>14} {'D_0.05 (95th pct)':>20}")
    for rho, p_violation, d95 in rows:
        print(f"{rho:>6.2f} {p_violation:>14.4f} {d95:>20.3f}")

    rhos, probs, _ = zip(*rows)
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(rhos, probs, marker="o")
    plt.xlabel("utilization rho")
    plt.ylabel(f"P(TTFT > {DEADLINE:.1f})")
    plt.title("LLM-serving TTFT SLO-violation probability vs load")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("examples/llm_serving_slo.png", dpi=120)
    print(Fore.GREEN + "\nSaved examples/llm_serving_slo.png")

    print(Style.RESET_ALL + "\nTwo SLO tiers (premium 30% / free 70%) at rho=0.9:")
    tiers = priority_tiers_at(0.9)
    print(f"  premium P(TTFT > D) = {tiers['premium']:.4f}")
    print(f"  free    P(TTFT > D) = {tiers['free']:.4f}")


if __name__ == "__main__":
    main()
