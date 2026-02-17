"""
Build compact delta-sweep figures comparing two CV values on same axes.

This script is intentionally lightweight: it only reads existing
`results.json` produced by `work_mgn_negatives.py` and generates
overlay plots (IM vs numeric, for CV=0.8 and CV=1.2) into
`negative_rate_cv_compare/v_ave.png` for each scenario directory.
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


def _load_series(results_json_path: str, field: str) -> Series:
    with open(results_json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    xs = [float(x) for x in d["negative_rate"]]
    if field == "v1":
        sim = [float(r["v"][0]) for r in d["sim"]]
        calc = [float(r["v"][0]) for r in d["calc"]]
    elif field == "q":
        sim = [float(r.get("q", 0.0) or 0.0) for r in d["sim"]]
        calc = [float(r.get("q", 0.0) or 0.0) for r in d["calc"]]
    else:
        raise ValueError(f"Unknown field: {field}")
    return Series(xs=xs, sim=sim, calc=calc)


def _plot_compare(out_path: str, cv_to_series: dict[float, Series], *, ylabel: str, ylim: tuple[float, float] | None):
    # CV color coding, method line-style coding:
    # - color: CV value
    # - linestyle: IM (--) vs numeric (-)
    # Okabe–Ito (colorblind-friendly) palette.
    # Use clearly separated hues so CV-curves are distinguishable even when close.
    colors = {
        0.8: "#E69F00",  # orange
        1.2: "#0072B2",  # blue
    }

    _fig, ax = plt.subplots()

    # Deterministic ordering in legend.
    for cv in sorted(cv_to_series.keys()):
        s = cv_to_series[cv]
        c = colors.get(cv, None)
        # Add markers for IM to keep curves distinguishable in grayscale/print.
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

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    if ylim is not None:
        ax.set_ylim(*ylim)

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
        0.8: "negative_rate_cv0_8",
        1.2: "negative_rate_cv1_2",
    }

    for rel_path, _caption in scenarios:
        scenario_dir = os.path.join(base_dir, rel_path)

        def _cv_to_series(field: str) -> dict[float, Series]:
            m: dict[float, Series] = {}
            missing = []
            for cv, cv_dir in cv_dirs.items():
                p = os.path.join(scenario_dir, cv_dir, "results.json")
                if not os.path.exists(p):
                    missing.append(p)
                    continue
                m[cv] = _load_series(p, field=field)
            if missing:
                raise FileNotFoundError("Missing results.json:\n" + "\n".join(missing))
            return m

        out_dir = os.path.join(scenario_dir, "negative_rate_cv_compare")

        out_path_v = os.path.join(out_dir, "v_ave.png")
        _plot_compare(out_path=out_path_v, cv_to_series=_cv_to_series("v1"), ylabel=r"$\upsilon_{1}$", ylim=None)
        print(f"Saved: {out_path_v}")

        out_path_q = os.path.join(out_dir, "q.png")
        _plot_compare(out_path=out_path_q, cv_to_series=_cv_to_series("q"), ylabel=r"$q$", ylim=(-0.02, 1.02))
        print(f"Saved: {out_path_q}")


if __name__ == "__main__":
    main()
