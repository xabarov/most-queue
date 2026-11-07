"""v9, figure 3 from delta-effect.json: w1 and q versus delta (TT + tagged equations)."""

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
    ("rcs", "0.8"): dict(marker="o", linestyle="-", color="#000000", label="RCS, CV = 0,8"),
    ("rcs", "1.2"): dict(marker="s", linestyle="--", color="#000000", label="RCS, CV = 1,2"),
    ("disaster", "0.8"): dict(marker="^", linestyle="-.", color="#595959", label="Очищение, CV = 0,8"),
    ("disaster", "1.2"): dict(marker="v", linestyle=":", color="#595959", label="Очищение, CV = 1,2"),
}
comma = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))


def panel(ax, data, key, ylabel):
    for (mode, cv), st in STYLE.items():
        rows = data[cv][mode]
        ax.plot([r["delta"] for r in rows], [r[key] for r in rows], markersize=4.2, linewidth=1.1,
                markerfacecolor="white", **st)
    ax.set_xlabel("Интенсивность воздействий $\\delta$")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(comma)
    ax.yaxis.set_major_formatter(comma)
    ax.grid(axis="y", linewidth=0.3, color="#bfbfbf")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    d = json.loads((ROOT / "delta-effect.json").read_text())["delta"]
    for name, key, ylabel in [("figure3a_waiting", "w1", "Среднее ожидание $w_1$"),
                              ("figure3b_completion", "q", "Вероятность $q$")]:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
        panel(ax, d, key, ylabel)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handlelength=3.0,
                  columnspacing=1.0, fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(ROOT / "figures" / f"{name}.{ext}")
        plt.close(fig)


if __name__ == "__main__":
    main()
