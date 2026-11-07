"""v9: figure 1b (clearing with auxiliary states), same graph as v4-v8.

Change from v8: the axis label "i" no longer overlaps the level label "0"
(levels shifted down), and the font matches figure 1a (STIX).
Run: .venv/bin/python works/negative_queues/revision_v9_20261108/figure1_disaster.py
"""

import importlib.util
import json
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("draw", ROOT.parent / "word/redraw_figure1.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
d.ROOT = ROOT / "figures"
d.plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})


def main():
    fig, ax = d.canvas(8.7)
    ys = {0: 1.0, 1: 2.5, 2: 4.25, 3: 6.0, 4: 7.75}
    dx = 13.8
    d.label(ax, 0.4, 0.3, "$i$", 12)
    d.label(ax, 0.4, ys[0], "0", 11.5)
    d.node(ax, dx, ys[0], "00")
    edges = []
    for i in range(1, 5):
        d.label(ax, 0.4, ys[i], str(i), 11.5)
        d.node(ax, dx, ys[i], rf"$\mathcal{{D}}_{i}$", dashed=True)
        d.arrow(ax, (dx, ys[i] - 0.30), (dx, ys[i - 1] + 0.30))
        d.label(ax, dx + 0.45, (ys[i] + ys[i - 1]) / 2, r"$\gamma$", 12)
        count = min(i, 3) + 1
        xs = [3.2 + 1.65 * j for j in range(count)]
        xleft, xright = xs[0] - 0.7, xs[-1] + 0.7
        ax.add_patch(FancyBboxPatch((xleft, ys[i] - 0.4), xright - xleft, 0.8,
                                    boxstyle="round,pad=0.02,rounding_size=0.1", edgecolor=d.INK,
                                    facecolor="white", linewidth=0.8))
        for j, x in enumerate(xs):
            d.label(ax, x, ys[i], f"{min(i, 3) - j}{j}", 12.5)
        d.arrow(ax, (xright + 0.05, ys[i] - 0.16), (dx - 0.60, ys[i - 1] + 0.1))
        d.label(ax, (xright + dx) / 2 - 0.1, (ys[i] + ys[i - 1]) / 2 - 0.4, r"$\delta$", 12)
        edges += [dict(source=[i, j], target=[i - 1, "D"] if i > 1 else [0, 0], rate="delta") for j in range(count)]
        edges.append(dict(source=[i, "D"], target=[i - 1, "D"] if i > 1 else [0, 0], rate="gamma"))
    d.save(fig, "figure1b_disaster")
    assert len(edges) == 17
    (ROOT / "figures/figure1b_disaster_edges.json").write_text(json.dumps(edges, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
