"""
Build a single "n-sweep" plot for four negative-arrival disciplines (CV fixed).

This is used to reduce the number of subfigures in the paper:
instead of 4 separate v1(n) panels (one per discipline), we draw them on one axes,
keeping the "old" visual language:
  - dashed line: simulation (ИМ)
  - solid line: numerical (Числ)
and using color to distinguish disciplines.

Run from repo root:
  ./.venv/bin/python works/negative_queues/plot_channels_4disciplines_oldstyle.py

Or for a custom base directory (e.g. paper_fig2_delta_0_300/...):
  ./.venv/bin/python works/negative_queues/plot_channels_4disciplines_oldstyle.py --base works/negative_queues/negative_queues_figures/paper_fig2_delta_0_300
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Series:
    xs: list[float]
    sim: list[float]
    calc: list[float]


def _load_v1_series(results_json_path: str) -> Series:
    with open(results_json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    xs = [float(x) for x in d["channels"]]
    sim = [float(r["v"][0]) for r in d["sim"]]
    calc = [float(r["v"][0]) for r in d["calc"]]
    return Series(xs=xs, sim=sim, calc=calc)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build combined v1(n) plot for four disciplines.")
    parser.add_argument(
        "--base",
        type=str,
        default="",
        help="Base directory that contains per-scenario results.json folders.",
    )
    args = parser.parse_args()

    base_dir = args.base or os.path.join(os.path.dirname(__file__), "negative_queues_figures")

    # Where to read results.json from:
    # - default (produced by work_mgn_negatives.py): per-scenario/channels_cv1_2/results.json
    # - paper_fig2_delta_* (produced by run_channels_for_paper_fig2.py): per-scenario/results.json
    is_paper_fig2_layout = os.path.basename(base_dir).startswith("paper_fig2_delta_")
    cv_dir = "" if is_paper_fig2_layout else "channels_cv1_2"

    # Use the same color palette approach as other paper figures:
    # Okabe–Ito (colorblind-friendly), plus light grid.
    disciplines: list[tuple[str, str, str]] = [
        ("RCS (удаление)", "rcs/remove", "#0072B2"),  # blue
        ("RCS (повторное обслуж.)", "rcs/requeue", "#E69F00"),  # orange
        ("Катастрофы (очищение)", "disaster/clear", "#009E73"),  # green
        ("Катастрофы (повторное обслуж.)", "disaster/requeue", "#D55E00"),  # vermillion
    ]

    label_to_series: dict[str, Series] = {}
    missing: list[str] = []
    for label, subdir, _color in disciplines:
        if is_paper_fig2_layout:
            # Different folder tags in the "paper" layout.
            subdir = {
                "rcs/remove": "rcs_remove",
                "rcs/requeue": "rcs_requeue",
                "disaster/clear": "dis_clear",
                "disaster/requeue": "dis_requeue",
            }[subdir]
            p = os.path.join(base_dir, subdir, "results.json")
        else:
            p = os.path.join(base_dir, subdir, cv_dir, "results.json")
        if not os.path.exists(p):
            missing.append(p)
            continue
        label_to_series[label] = _load_v1_series(p)
    if missing:
        raise FileNotFoundError("Missing results.json:\n" + "\n".join(missing))

    _fig, ax = plt.subplots()

    for label, _subdir, color in disciplines:
        s = label_to_series[label]
        # Add markers for IM, like in overlay scripts, to preserve readability in print.
        ax.plot(
            s.xs,
            s.sim,
            linestyle="--",
            linewidth=2.2,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            color=color,
        )
        ax.plot(s.xs, s.calc, linestyle="-", linewidth=2.6, color=color)

    ax.set_xlabel("n")
    ax.set_ylabel(r"$\upsilon_{1}$")
    ax.grid(True, alpha=0.15)

    # Two compact legends: colors for disciplines, styles for methods.
    disc_handles = [Line2D([0], [0], color=c, lw=2.6, label=lab) for (lab, _sd, c) in disciplines]
    leg1 = ax.legend(handles=disc_handles, loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=2.4, linestyle="--", label="ИМ"),
        Line2D([0], [0], color="k", lw=2.6, linestyle="-", label="Числ"),
    ]
    ax.legend(handles=style_handles, loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

    out_dir = os.path.join(base_dir, "combined")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v1_n_cv1_2_oldstyle.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(_fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
