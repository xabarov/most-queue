"""Publication-size vector redraw of the two existing transition graphs.

Run with the project's matplotlib environment. The original figures are read only.
Only layout changes: the graph topology and all rates are retained.
"""

from pathlib import Path
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent / "figures_v2"
INK = "#202020"
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "savefig.facecolor": "white",
    }
)
EDGES = {"rcs": [], "disaster": []}


def canvas(height):
    fig = plt.figure(figsize=(16.2 / 2.54, height / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16.2), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, size=11, ha="center"):
    return ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=INK, zorder=5)


def node(ax, x, y, text, dashed=False, width=1.14):
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.28),
            width,
            0.56,
            boxstyle="round,pad=0.015,rounding_size=0.09",
            facecolor="white",
            edgecolor=INK,
            linewidth=0.85,
            linestyle=(0, (4, 2)) if dashed else "-",
            zorder=3,
        )
    )
    label(ax, x, y, text, 11.5)


def arrow(ax, start, end, control=None, lw=0.85):
    if control:
        path = MPath([start, *control, end], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=INK, zorder=1)
    else:
        patch = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=INK, shrinkA=0, shrinkB=0, zorder=1
        )
    ax.add_patch(patch)


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(ROOT / (name + "." + ext), dpi=400)
    plt.close(fig)


def rcs():
    fig, ax = canvas(9.9)
    ys = {0: 0.65, 1: 2.25, 2: 3.85, 3: 5.45, 4: 9.25}
    xs = {i: [8.25 + (j - min(i, 3) / 2) * 3.65 for j in range(min(i, 3) + 1)] for i in range(5)}
    label(ax, 0.42, 0.27, "$i$", 11)
    for i in range(5):
        label(ax, 0.42, ys[i], str(i), 10)
        for j, x in enumerate(xs[i]):
            node(ax, x, ys[i], f"{min(i,3)-j}{j}")
    for i in range(1, 4):
        for j, x in enumerate(xs[i]):
            for target, k in [(j, i - j), (j - 1, j)]:
                if not k:
                    continue
                tx = xs[i - 1][target]
                side = 1 if tx > x else -1
                arrow(ax, (x + 0.12 * side, ys[i] - 0.3), (tx - 0.12 * side, ys[i - 1] + 0.3))
                g = math.gcd(k, i)
                a, b = k // g, i // g
                rate = r"\delta" if a == b else (rf"\delta/{b}" if a == 1 else rf"{a}\delta/{b}")
                # Put labels on the outside of each diagonal, never on its stroke.
                label(ax, (x + tx) / 2 - side * 0.34, (ys[i] + ys[i - 1]) / 2 - 0.16, "$" + rate + "$", 11)
                EDGES["rcs"].append({"from": [i, j], "to": [i - 1, target], "rate": f"delta*{k}/{i}"})

    # The cross-diagonal labels sit in the upper triangular spaces; vertical
    # edge labels sit in the middle triangular spaces, away from every stroke.
    rates_same = [r"y_1", r"\frac{2y_1+y_2}{3}", r"\frac{y_1+2y_2}{3}", r"y_2"]
    rates_right = [r"y_2", r"\frac{2y_2}{3}", r"\frac{y_2}{3}"]
    rates_left = [r"\frac{y_1}{3}", r"\frac{2y_1}{3}", r"y_1"]
    y0, y1 = ys[4] - 0.3, ys[3] + 0.3
    for j, x in enumerate(xs[4]):
        arrow(ax, (x, y0), (x, y1))
        # Label beside a vertical edge, in the space below the crossing pair.
        sign = -1 if j in (0, 1) else 1
        label(ax, x + sign * 0.18, 7.35, r"$\delta\," + rates_same[j] + "$", 11.5, ha="right" if sign < 0 else "left")
        EDGES["rcs"].append({"from": [4, j], "to": [3, j], "rate": "delta*" + rates_same[j]})
    for j in range(3):
        x, z = xs[4][j], xs[3][j + 1]
        arrow(ax, (x + 0.13, y0), (z - 0.13, y1))
        arrow(ax, (z - 0.13, y0), (x + 0.13, y1))
        label(ax, x + 1.25, 6.15, r"$\delta\," + rates_left[j] + "$", 11.5)
        label(ax, x + 2.40, 6.15, r"$\delta\," + rates_right[j] + "$", 11.5)
        EDGES["rcs"].append({"from": [4, j], "to": [3, j + 1], "rate": "delta*" + rates_right[j]})
        EDGES["rcs"].append({"from": [4, j + 1], "to": [3, j], "rate": "delta*" + rates_left[j]})
    save(fig, "figure1a_rcs")


def disaster():
    fig, ax = canvas(9.3)
    ys = {0: 0.55, 1: 2.1, 2: 4.05, 3: 6.0, 4: 7.95}
    dx = 14.65
    label(ax, 0.42, 0.15, "$i$", 9)
    label(ax, 0.42, ys[0], "0", 10)
    node(ax, dx, ys[0], "00")
    for i in range(1, 5):
        label(ax, 0.42, ys[i], str(i), 10)
        node(ax, dx, ys[i], "$D$", dashed=True)
        arrow(ax, (dx, ys[i] - 0.3), (dx, ys[i - 1] + 0.3))
        label(ax, dx + 0.4, (ys[i] + ys[i - 1]) / 2, r"$\gamma$", 12)
        EDGES["disaster"].append({"from": [i, "D"], "to": [i - 1, "D"] if i > 1 else [0, 0], "rate": "gamma"})
        count = min(i, 3) + 1
        for j in range(count):
            x = 2.05 + j * 2.7
            y = ys[i]
            node(ax, x, y, f"{min(i,3)-j}{j}")
            # Nested arcs start at the top/bottom of the nodes, avoiding all
            # other state boxes. Labels sit in open space above/below the arc.
            sign = -1 if j % 2 == 0 else 1
            outer = j < 2
            h = 0.83 if outer else 0.47
            start = (x, y + sign * 0.3)
            end = (dx - 0.35 + (0 if outer else 0.2), y + sign * 0.3)
            c1 = (x + 1.1, y + sign * h)
            c2 = (dx - 2, y + sign * h)
            arrow(ax, start, end, [c1, c2])
            lx = (6.7 if sign < 0 else 9.1) if outer else (11.4 if sign < 0 else 12.35)
            ly = y + sign * (0.90 if outer else 0.12)
            label(ax, lx, ly, r"$\delta$", 12)
            EDGES["disaster"].append({"from": [i, j], "to": [i, "D"], "rate": "delta"})
    save(fig, "figure1b_disaster")


def main():
    ROOT.mkdir(exist_ok=True)
    rcs()
    disaster()
    assert len(EDGES["rcs"]) == 22
    assert len(EDGES["disaster"]) == 17
    (ROOT / "transitions.json").write_text(json.dumps(EDGES, ensure_ascii=False, indent=2) + "\n")
    print(ROOT)


if __name__ == "__main__":
    main()
