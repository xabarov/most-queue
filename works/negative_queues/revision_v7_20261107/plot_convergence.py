"""Standalone scientific convergence plot from the saved numerical grids."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from works.negative_queues.revision_v7_20261107.summarize_step2 import point

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "step2-results.json").read_text())["results"]
    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    # Match the manuscript's 16.2 cm image width, preserving readable type.
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 5.8), constrained_layout=True)
    for row, ax in zip(data, axes.ravel()):
        cv, mode = row["cv"], row["mode"]
        ks = np.array([4, 8, 12, 16])
        for j, color, label, style in [(3, "#9BA6AD", "J = 3", "--"), (4, "#17466B", "J = 4", "-")]:
            values = [point(mode, cv, j, int(k))["w"][0] for k in ks]
            ax.plot(1 / ks, values, marker="o", ms=4, color=color, ls=style, label=label)
        a = row["calculation"]["w"][0]
        b = row["alternative"]["w"][0]
        ax.scatter([0], [a], s=35, marker="D", color="#17466B", zorder=4, label="Экстраполяция по 1/K")
        ax.scatter([0], [b], s=35, marker="s", facecolors="none", edgecolors="#AD5C20", zorder=4, label="По 1/√K")
        ref = row["comparison"]["w"]
        ax.axhspan(
            ref["mean"] - ref["half_ci95"],
            ref["mean"] + ref["half_ci95"],
            color="#CCCCCC",
            alpha=0.45,
            label="95%-й ДИ имитации",
        )
        ax.set(
            title=f'{"RCS" if mode=="rcs" else "Общие прерывания"}, CV = {cv}', xlabel="1/K", ylabel="Среднее ожидание"
        )
        ax.set_xlim(-0.012, 0.26)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "convergence.pdf")
    figure.savefig(output / "convergence.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
