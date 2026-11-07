"""v9, figure 2 from cv-effect.json (grayscale-safe: marker + linestyle per series)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 12.5,
    "axes.labelsize": 12.5, "legend.fontsize": 11, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.linewidth": 0.8, "lines.markeredgewidth": 0.8, "mathtext.fontset": "dejavuserif",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
STYLE = {
    "rcs_remove": dict(marker="o", linestyle="-", color="#000000", label="RCS, удаление"),
    "clearing": dict(marker="s", linestyle="--", color="#404040", label="Очищение"),
    "rcs_restart": dict(marker="^", linestyle="-.", color="#000000", label="RCS, новая длит."),
    "common_restart": dict(marker="v", linestyle=(0, (5, 1.5, 1, 1.5, 1, 1.5)), color="#595959",
                           label="Все каналы, новая длит."),
    "common_retained": dict(marker="D", linestyle=(0, (1, 1.2)), color="#000000",
                            label="Все каналы, сохр. длит."),
    "no_events": dict(marker="x", linestyle=":", color="#7f7f7f", label="Без воздействий"),
}
comma = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))


def panel(ax, data, keys, ylabel, log=False):
    grid = data["grid"]
    for k in keys:
        ax.plot(grid, [data[str(cv)][k] for cv in grid], markersize=4.2, linewidth=1.1, markerfacecolor="white",
                **STYLE[k])
    ax.set_xlabel("Коэффициент вариации")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(comma)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.grid(axis="y", linewidth=0.3, color="#bfbfbf")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    d = json.loads((ROOT / "cv-effect.json").read_text())
    occ = {"grid": d["grid"], **d["occupation"]}
    wait = {"grid": d["grid"], **d["waiting"]}
    figs = [
        ("figure2a_occupation", occ, ["rcs_remove", "clearing", "rcs_restart", "common_restart", "common_retained"],
         "Время занятия канала / $b_1$", False),
        ("figure2b_waiting", wait, ["no_events", "rcs_remove", "clearing", "rcs_restart", "common_restart"],
         "Среднее ожидание $w_1$", True),
    ]
    for name, data, keys, ylabel, log in figs:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
        panel(ax, data, keys, ylabel, log)
        if not log:
            ax.axhline(1.0, color="#7f7f7f", linewidth=0.7, linestyle=":")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handlelength=3.0,
                  columnspacing=1.0, fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(ROOT / "figures" / f"{name}.{ext}")
        plt.close(fig)


if __name__ == "__main__":
    main()
