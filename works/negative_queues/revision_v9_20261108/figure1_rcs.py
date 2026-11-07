"""v9: transition graph of RCS removal for H2 service, n=3 (figure 1).

Topology and rates are those of the v4-v8 figure (checked against the
generator): from level i<=3 a busy server in phase l is removed with rate
delta*k_l/i; from level 4 the head of the queue starts service in phase l'
with probability y_l'. Changes from v8: rates are written in line
(delta*y1/3 instead of a fraction set like an exponent), and each label is
drawn along its own arrow near its source, so crossing arrows are unambiguous.

Run: .venv/bin/python works/negative_queues/revision_v9_20261108/figure1_rcs.py
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent / "figures"
INK = "#1a1a1a"
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
W, H = 16.2, 11.4
EDGES = []


def text(ax, x, y, s, size=11, rotation=0, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=INK, rotation=rotation, rotation_mode="anchor", zorder=6,
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none") if rotation or ha != "center" else None)


def node(ax, x, y, s):
    ax.add_patch(FancyBboxPatch((x - 0.55, y - 0.27), 1.1, 0.54, boxstyle="round,pad=0.015,rounding_size=0.09",
                                facecolor="white", edgecolor=INK, linewidth=0.9, zorder=3))
    ax.text(x, y, s, ha="center", va="center", fontsize=12.5, color=INK, zorder=4)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=0.85, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=1))


def along(ax, a, b, s, t, offset, size=11):
    """Label parallel to segment a->b at fraction t from a, shifted to its left by offset."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length  # left normal in data coordinates (y grows downwards)
    x, y = a[0] + t * dx + offset * nx, a[1] + t * dy + offset * ny
    angle = -math.degrees(math.atan2(dy, dx))  # screen angle, y axis inverted
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=INK, rotation=angle,
            rotation_mode="anchor", zorder=6, bbox=dict(boxstyle="square,pad=0.06", facecolor="white", edgecolor="none"))


def rate(k, i):
    f = Fraction(k, i)
    if f == 1:
        return r"$\delta$"
    if f.numerator == 1:
        return rf"$\delta/{f.denominator}$"
    return rf"${f.numerator}\delta/{f.denominator}$"


def main():
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
    ax.axis("off")
    ys = {0: 0.75, 1: 2.35, 2: 3.95, 3: 5.55, 4: 10.55}
    xs = {i: [8.6 + (j - min(i, 3) / 2) * 3.6 for j in range(min(i, 3) + 1)] for i in range(5)}
    ax.text(0.55, 0.2, "$i$", ha="center", va="center", fontsize=12, color=INK)
    for i in range(5):
        ax.text(0.55, ys[i], str(i), ha="center", va="center", fontsize=11.5, color=INK)
        for j, x in enumerate(xs[i]):
            node(ax, x, ys[i], f"{min(i, 3) - j}{j}")
    # Levels 1-3: removal of a server in phase 1 (left) or phase 2 (right).
    for i in range(1, 4):
        for j, x in enumerate(xs[i]):
            ones, twos = i - j, j
            for target, k in [(j, ones), (j - 1, twos)]:
                if not k:
                    continue
                tx = xs[i - 1][target]
                side = 1 if tx > x else -1
                a, b = (x + 0.14 * side, ys[i] - 0.28), (tx - 0.14 * side, ys[i - 1] + 0.28)
                arrow(ax, a, b)
                along(ax, a, b, rate(k, i), 0.5, 0.26 * side)
                EDGES.append({"from": [i, f"{ones}{twos}"], "to": [i - 1, f"{min(i-1,3)-target}{target}"],
                              "rate": f"delta*{k}/{i}"})
    # Level 4 -> 3: removal, then the queue head starts in phase 1 or 2.
    same = [r"$\delta y_1$", r"$\delta(2y_1+y_2)/3$", r"$\delta(y_1+2y_2)/3$", r"$\delta y_2$"]
    right = [r"$\delta y_2$", r"$2\delta y_2/3$", r"$\delta y_2/3$"]  # j -> j+1 (one more phase-2 server)
    left = [r"$\delta y_1/3$", r"$2\delta y_1/3$", r"$\delta y_1$"]  # j+1 -> j
    y0, y1 = ys[4] - 0.28, ys[3] + 0.28
    for j, x in enumerate(xs[4]):
        arrow(ax, (x, y0), (x, y1))
        along(ax, (x, y0), (x, y1), same[j], 0.5, 0.0, size=10.5)
        EDGES.append({"from": [4, f"{3-j}{j}"], "to": [3, f"{3-j}{j}"], "rate": same[j]})
    for j in range(3):
        x, z = xs[4][j], xs[3][j + 1]
        a, b = (x + 0.16, y0), (z - 0.16, y1)
        arrow(ax, a, b)
        along(ax, a, b, right[j], 0.2, 0.0, size=10.5)
        c, d = (z - 0.16, y0), (x + 0.16, y1)
        arrow(ax, c, d)
        along(ax, c, d, left[j], 0.2, 0.0, size=10.5)
        EDGES.append({"from": [4, f"{3-j}{j}"], "to": [3, f"{2-j}{j+1}"], "rate": right[j]})
        EDGES.append({"from": [4, f"{2-j}{j+1}"], "to": [3, f"{3-j}{j}"], "rate": left[j]})
    for ext in ("pdf", "png", "svg"):
        fig.savefig(ROOT / f"figure1_rcs_v9.{ext}", dpi=400)
    plt.close(fig)
    (ROOT / "figure1_rcs_v9_edges.json").write_text(json.dumps(EDGES, ensure_ascii=False, indent=1))


def check():
    """Rates out of each level-4 state must sum to delta (one removal, then y1+y2=1)."""
    import sympy as sp

    y1 = sp.Symbol("y1")
    y2 = 1 - y1
    for j in range(4):
        ones, twos = 3 - j, j
        total = 0
        # remove phase-1 server w.p. ones/3, phase-2 server w.p. twos/3; new phase from y.
        out = {}
        for removed, p in [("1", sp.Rational(ones, 3)), ("2", sp.Rational(twos, 3))]:
            for new, q in [("1", y1), ("2", y2)]:
                o, t = ones - (removed == "1") + (new == "1"), twos - (removed == "2") + (new == "2")
                out[(o, t)] = out.get((o, t), 0) + p * q
                total += p * q
        assert sp.simplify(total - 1) == 0
        print(f"{ones}{twos}:", {f"{o}{t}": sp.factor(v) for (o, t), v in out.items() if v != 0})


if __name__ == "__main__":
    check()
    main()
