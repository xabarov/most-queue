"""
Build a single n-sweep plot for "repeat without resampling" (fixed-service restarts),
showing two disciplines on one axes (CV fixed).

We use the same visual style as other compact paper plots:
  - Okabe–Ito palette (color by discipline)
  - dashed with markers: simulation (ИМ)
  - solid: numerical (Числ)
  - light grid

By default reads `results.json` produced by `work_mgn_negatives.py` in:
  negative_queues_figures/rcs/requeue_fixed/channels_cv1_2/results.json
  negative_queues_figures/disaster/requeue_fixed/channels_cv1_2/results.json

Run from repo root:
  ./.venv/bin/python works/negative_queues/plot_channels_fixed_2disciplines_cv1_2.py
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
    base_dir = os.path.join(os.path.dirname(__file__), "negative_queues_figures")

    disciplines: list[tuple[str, str, str]] = [
        ("RCS (повтор без пересемпл.)", "rcs/requeue_fixed", "#0072B2"),
        ("Катастрофы (повтор без пересемпл.)", "disaster/requeue_fixed", "#D55E00"),
    ]

    cv_dir = "channels_cv1_2"

    label_to_series: dict[str, Series] = {}
    missing: list[str] = []
    for label, subdir, _color in disciplines:
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

    disc_handles = [Line2D([0], [0], color=c, lw=2.6, label=lab) for (lab, _sd, c) in disciplines]
    leg1 = ax.legend(handles=disc_handles, loc="upper left", fontsize=9, frameon=True, framealpha=0.9)
    ax.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="k", lw=2.2, linestyle="--", marker="o", markersize=3.0, label="ИМ"),
        Line2D([0], [0], color="k", lw=2.6, linestyle="-", label="Числ"),
    ]
    ax.legend(handles=style_handles, loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

    out_dir = os.path.join(base_dir, "combined")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v1_n_fixed_cv1_2.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(_fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
