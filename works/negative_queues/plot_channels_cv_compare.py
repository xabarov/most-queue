"""
Build compact n-sweep figures comparing two CV values on same axes.

Reads `results.json` produced by `work_mgn_negatives.py` in
`channels_cv0_8/` and `channels_cv1_2/` and creates overlay plots in
`channels_cv_compare/v_ave.png` for each scenario.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt


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


def _plot_v1_compare(out_path: str, cv_to_series: dict[float, Series]) -> None:
    # Okabe–Ito (colorblind-friendly) palette.
    colors = {
        0.8: "#E69F00",  # orange
        1.2: "#0072B2",  # blue
    }

    _fig, ax = plt.subplots()

    for cv in sorted(cv_to_series.keys()):
        s = cv_to_series[cv]
        c = colors.get(cv, None)
        ax.plot(
            s.xs,
            s.sim,
            linestyle="--",
            linewidth=2.2,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            color=c,
            label=f"ИМ, CV={cv:g}",
        )
        ax.plot(s.xs, s.calc, linestyle="-", linewidth=2.4, color=c, label=f"Числ, CV={cv:g}")

    ax.set_xlabel("n")
    ax.set_ylabel(r"$\upsilon_{1}$")
    ax.legend(ncol=2, fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(_fig)


def main() -> None:
    # Keep consistent with `work_mgn_negatives.py` which supports OUTPUT_DIR override.
    base_dir = os.environ.get("OUTPUT_DIR") or os.path.join(os.path.dirname(__file__), "negative_queues_figures")

    scenarios = [
        ("rcs/remove", "RCS (удаление)"),
        ("rcs/requeue", "RCS (повторное обслуж.)"),
        ("rcs/requeue_fixed", "RCS (повтор без пересемпл.)"),
        ("disaster/clear", "Катастрофы (очищение)"),
        ("disaster/requeue", "Катастрофы (повторное обслуж.)"),
        ("disaster/requeue_fixed", "Катастрофы (повтор без пересемпл.)"),
    ]

    cv_dirs = {
        0.8: "channels_cv0_8",
        1.2: "channels_cv1_2",
    }

    for rel_path, _caption in scenarios:
        scenario_dir = os.path.join(base_dir, rel_path)
        cv_to_series: dict[float, Series] = {}
        missing = []
        for cv, cv_dir in cv_dirs.items():
            p = os.path.join(scenario_dir, cv_dir, "results.json")
            if not os.path.exists(p):
                missing.append(p)
                continue
            cv_to_series[cv] = _load_v1_series(p)

        if missing:
            raise FileNotFoundError("Missing results.json:\n" + "\n".join(missing))

        out_path = os.path.join(scenario_dir, "channels_cv_compare", "v_ave.png")
        _plot_v1_compare(out_path=out_path, cv_to_series=cv_to_series)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
