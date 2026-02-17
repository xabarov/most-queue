"""
Generate results for the paper's Fig. 2: v1(n) for four disciplines on one axes.

Unlike `work_mgn_negatives.py` (which caps delta for the n-sweep to keep all scenarios stable),
this script allows choosing delta explicitly to match the paper's "base" example if desired.

Run from repo root, e.g.:
  ./.venv/bin/python works/negative_queues/run_channels_for_paper_fig2.py --delta 0.30 --cv 1.2 --jobs 200000

It writes per-scenario `results.json` under:
  works/negative_queues/negative_queues_figures/paper_fig2_delta_<...>/<scenario>/

Then you can build a combined "old style" overlay plot with:
  ./.venv/bin/python works/negative_queues/plot_channels_4disciplines_oldstyle.py --base <that dir>
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from works.negative_queues.work_mgn_negatives import (
    DisasterScenario,
    NegativeServiceType,
    RcsScenario,
    collect_calc_results,
    collect_sim_results,
    gamma_moments_by_mean_and_cv,
    read_parameters_from_yaml,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate n-sweep data for paper Fig. 2.")
    parser.add_argument("--delta", type=float, default=0.02, help="Negative arrival rate delta.")
    parser.add_argument("--cv", type=float, default=1.2, help="Service-time CV.")
    parser.add_argument("--jobs", type=int, default=200_000, help="Number of jobs for simulation per n.")
    parser.add_argument("--nmax", type=int, default=10, help="Max number of channels.")
    args = parser.parse_args()

    base_qp = read_parameters_from_yaml("works/negative_queues/base_parameters.yaml")
    qp = dict(base_qp)
    qp["arrival_rate"] = dict(base_qp["arrival_rate"])
    qp["arrival_rate"]["negative"] = float(args.delta)
    qp["num_of_jobs"] = int(args.jobs)

    # Fix rho and lambda_pos; mean service increases with n.
    rho = float(qp["utilization"]["base"])
    l_pos = float(qp["arrival_rate"]["positive"])
    cv = float(args.cv)

    scenarios = [
        ("rcs_remove", NegativeServiceType.RCS, False, RcsScenario.REMOVE, DisasterScenario.CLEAR_SYSTEM),
        ("rcs_requeue", NegativeServiceType.RCS, True, RcsScenario.REQUEUE, DisasterScenario.CLEAR_SYSTEM),
        ("dis_clear", NegativeServiceType.DISASTER, False, RcsScenario.REMOVE, DisasterScenario.CLEAR_SYSTEM),
        ("dis_requeue", NegativeServiceType.DISASTER, True, RcsScenario.REMOVE, DisasterScenario.REQUEUE_ALL),
    ]

    # Output folder (inside repo) so LaTeX can include plots.
    out_root = os.path.join(
        os.path.dirname(__file__),
        "negative_queues_figures",
        f"paper_fig2_delta_{args.delta:.3f}".replace(".", "_"),
    )
    os.makedirs(out_root, exist_ok=True)

    channels = list(range(1, int(args.nmax) + 1))

    for tag, discipline, requeue, rcs_sc, dis_sc in scenarios:
        calc = []
        sim = []
        for n in channels:
            service_mean = n * rho / l_pos
            b = gamma_moments_by_mean_and_cv(service_mean, cv)
            calc.append(
                collect_calc_results(
                    qp=qp,
                    n=n,
                    b=b,
                    discipline=discipline,
                    requeue_on_negative=requeue,
                    max_p=100,
                )
            )
            sim.append(
                collect_sim_results(
                    qp=qp,
                    n=n,
                    b=b,
                    discipline=discipline,
                    rcs_scenario=rcs_sc,
                    disaster_scenario=dis_sc,
                    max_p=100,
                )
            )

        # Store a minimal schema compatible with plotting scripts, but avoid dumping
        # full dataclasses (they may contain complex / numpy types for H2 parameters).
        def _pack(results_list):
            packed = []
            for r in results_list:
                v = [float(x) for x in (getattr(r, "v", []) or [])]
                w = [float(x) for x in (getattr(r, "w", []) or [])]
                dct = {"v": v, "w": w}
                if hasattr(r, "q"):
                    try:
                        dct["q"] = float(getattr(r, "q") or 0.0)
                    except Exception:
                        pass
                packed.append(dct)
            return packed

        payload = {
            "channels": [int(x) for x in channels],
            "calc": _pack(calc),
            "sim": _pack(sim),
            "utilization_factor": float(rho),
            "service_time_variation_coef": float(cv),
            "arrival_rate": {"positive": float(l_pos), "negative": float(args.delta)},
        }

        out_dir = os.path.join(out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"Saved: {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
