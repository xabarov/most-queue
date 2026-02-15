"""
Generate transition diagrams for M/H2/n with negative arrivals.

We generate separate diagrams for each discipline:
- negative arrivals: only transitions caused by negative arrivals (delta);
- service: only service completions (mu), without negative effects.

The diagrams are intentionally limited to n=3 and levels 0..n+1 to keep the figure readable
and consistent with the expository style used in the accompanying LaTeX article.

Note: the article text explains that the B matrices combine service transitions (shown in
the service diagram) with negative effects (shown in the negative arrivals diagram).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


@dataclass(frozen=True)
class Node:
    level: int
    j: int


def _key(level: int, j: int, n: int) -> str:
    """
    Microstate key used in the Takahashi–Takami implementation:
    a two-digit code ab, where
      a = number of jobs in phase 1 among busy servers,
      b = number of jobs in phase 2 among busy servers.

    For level i <= n, the number of busy servers is i.
    For level i > n, the number of busy servers is n (queueing jobs are not part of the microstate).
    """
    busy = min(level, n)
    a = busy - j
    b = j
    return f"{a}{b}"


def _fmt_delta(num: int, den: int) -> str:
    """
    Pretty formatting for delta * num/den:
    - delta for 1
    - delta/3, 2delta/3, delta/2, ...
    """
    if den <= 0 or num <= 0:
        return ""
    g = math.gcd(num, den)
    num //= g
    den //= g
    if num == den:
        return r"$\delta$"
    if num == 1:
        return rf"$\delta/{den}$"
    return rf"${num}\delta/{den}$"


def _coords(level: int, j: int, n: int) -> tuple[float, float]:
    """
    Coordinates for microstate (level, j) where j is number of phase-2 jobs in service.
    For level <= n, number of microstates is level+1. For level > n, it is n+1.
    """
    m = min(level, n)
    x = (j - m / 2.0) * 1.60
    y = -float(level) * 1.55
    return x, y


NODE_RADIUS = 0.20


def _draw_node(ax, x: float, y: float, label: str, *, radius: float = NODE_RADIUS) -> None:
    ax.add_patch(
        plt.Circle(
            (x, y),
            radius=radius,
            fill=True,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.text(x, y, label, fontsize=14, ha="center", va="center", fontweight="medium", zorder=4)


# -----------------------------
# "Pretty" (colored) style nodes
# -----------------------------

LEVEL_COLORS = {
    0: "#A7D7FF",  # pastel blue
    1: "#7FE5DE",  # pastel turquoise
    2: "#C7A6FF",  # pastel violet
    3: "#FFBE7A",  # pastel orange
    4: "#FF9DB7",  # pastel pink
}


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0.0, min(1.0, x)) for x in rgb)
    return "#{:02x}{:02x}{:02x}".format(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _darken(hex_color: str, factor: float = 0.70) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    return _rgb01_to_hex((r * factor, g * factor, b * factor))


def _draw_box_node(
    ax,
    x: float,
    y: float,
    label: str,
    *,
    level: int,
    width: float = 1.15,
    height: float = 0.62,
    dashed: bool = False,
    fontsize: int = 14,
) -> None:
    fc = LEVEL_COLORS.get(level, "#FFFFFF")
    ec = _darken(fc, 0.60)
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor=fc,
        edgecolor=ec,
        linewidth=2.0,
        linestyle="--" if dashed else "-",
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight="medium",
        zorder=4,
    )


def _m(label: str) -> str:
    """Mathtext helper."""
    return f"${label}$"


def _delta_frac(num: int, den: int) -> str:
    """Return δ or δ·(num/den) as mathtext."""
    if num <= 0 or den <= 0:
        return ""
    g = math.gcd(num, den)
    num //= g
    den //= g
    if num == den:
        return _m(r"\delta")
    return _m(rf"\delta\cdot\frac{{{num}}}{{{den}}}")


def _delta_y(num: int, den: int, y: str) -> str:
    """Return δ·(num/den)·y as mathtext, collapsing 1 to δ·y."""
    if num <= 0 or den <= 0:
        return ""
    g = math.gcd(num, den)
    num //= g
    den //= g
    if num == den:
        return _m(rf"\delta {y}")
    return _m(rf"\delta\cdot\frac{{{num}}}{{{den}}}{y}")


def _delta_combo(a_num: int, a_den: int, b_num: int, b_den: int) -> str:
    """δ·(a_num/a_den*y1 + b_num/b_den*y2) in a compact mathtext form."""
    # assume inputs already simplified (we use 1/3,2/3 etc)
    return _m(rf"\delta\left(\frac{{{a_num}}}{{{a_den}}}y_1+\frac{{{b_num}}}{{{b_den}}}y_2\right)")


BOX_R = 0.36  # arrow shortening radius for box nodes (data units)


def _shorten_segment(
    x0: float, y0: float, x1: float, y1: float, r0: float, r1: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Move segment endpoints away from node centers, so arrows do not enter circles.
    Radii r0 and r1 are in data units.
    """
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L <= 1e-9:
        return (x0, y0), (x1, y1)
    ux = dx / L
    uy = dy / L
    return (x0 + r0 * ux, y0 + r0 * uy), (x1 - r1 * ux, y1 - r1 * uy)


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str | None,
    *,
    color: str = "black",
    lw: float = 1.0,
    ls: str = "-",
    rad: float = 0.0,
    label_offset: tuple[float, float] = (0.0, 0.0),
    fontsize: int = 10,
    start_radius: float = NODE_RADIUS,
    end_radius: float = NODE_RADIUS,
) -> None:
    # Shorten endpoints so arcs don't hit node circles.
    (sx, sy), (tx, ty) = _shorten_segment(x0, y0, x1, y1, start_radius, end_radius)
    patch = FancyArrowPatch(
        (sx, sy),
        (tx, ty),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        xm = (x0 + x1) / 2.0 + label_offset[0]
        ym = (y0 + y1) / 2.0 + label_offset[1]
        ax.text(
            xm,
            ym,
            label,
            fontsize=fontsize,
            color=color,
            ha="center",
            va="center",
            # Put labels above nodes to avoid being covered by circles.
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        )


def _draw_level_labels(ax, max_level: int) -> None:
    for i in range(max_level + 1):
        ax.text(-3.45, -i * 1.55, f"{i}", fontsize=12, ha="right", va="center")
    ax.text(-3.45, 1.10, "уровень $i$", fontsize=12, ha="right", va="center")


def _base_nodes(n: int, max_level: int) -> list[Node]:
    nodes: list[Node] = []
    for level in range(max_level + 1):
        m = min(level, n)
        for j in range(m + 1):
            nodes.append(Node(level=level, j=j))
    return nodes


def _setup_axes(max_level: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    ax.set_axis_off()
    _draw_level_labels(ax, max_level)
    ax.set_xlim(-3.9, 8.1)
    ax.set_ylim(-1.55 * (max_level + 0.9), 1.35)
    return fig, ax


def _draw_base_nodes(ax, n: int, max_level: int) -> None:
    for node in _base_nodes(n, max_level):
        x, y = _coords(node.level, node.j, n)
        # Level i is displayed on the left, so node label shows the microstate key (phase-1/phase-2 counts).
        key = _key(node.level, node.j, n)
        _draw_node(ax, x, y, rf"${key}$", radius=NODE_RADIUS)


def _draw_arrivals(ax, n: int, max_level: int) -> None:
    """Positive arrivals (matrix A)."""
    for level in range(0, max_level):
        m = min(level, n)
        for j in range(m + 1):
            x0, y0 = _coords(level, j, n)
            if level < n:
                x1, y1 = _coords(level + 1, j, n)
                _arrow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    r"$\lambda y_1$",
                    color="#1f77b4",
                    lw=1.4,
                    rad=-0.12,
                    label_offset=(-0.05, 0.20),
                )
                x2, y2 = _coords(level + 1, j + 1, n)
                _arrow(
                    ax,
                    x0,
                    y0,
                    x2,
                    y2,
                    r"$\lambda y_2$",
                    color="#1f77b4",
                    lw=1.4,
                    rad=0.12,
                    label_offset=(0.05, 0.20),
                )
            else:
                x1, y1 = _coords(level + 1, j, n)
                _arrow(ax, x0, y0, x1, y1, r"$\lambda$", color="#1f77b4", lw=1.4, rad=0.0, label_offset=(0.0, 0.22))


def _draw_service_base(ax, n: int, max_level: int) -> None:
    """Service completions (matrix B) for the base M/H2/n structure."""
    for level in range(1, max_level + 1):
        busy = min(level, n)
        m = min(level, n)
        for j in range(m + 1):
            x0, y0 = _coords(level, j, n)
            if level <= n:
                if busy - j > 0:
                    x1, y1 = _coords(level - 1, j, n)
                    _arrow(
                        ax,
                        x0,
                        y0,
                        x1,
                        y1,
                        rf"${busy - j}\mu_1$",
                        color="black",
                        lw=1.1,
                        rad=-0.08,
                        label_offset=(-0.12, -0.14),
                        fontsize=10,
                    )
                if j > 0:
                    x2, y2 = _coords(level - 1, j - 1, n)
                    _arrow(
                        ax,
                        x0,
                        y0,
                        x2,
                        y2,
                        rf"${j}\mu_2$",
                        color="black",
                        lw=1.1,
                        rad=0.08,
                        label_offset=(0.12, -0.14),
                        fontsize=10,
                    )
            else:
                # For i>n: after a departure the freed server is immediately occupied
                # by a queued job which chooses phase 1/2 with probabilities y1/y2.
                # This yields three possible downward transitions of the microstate index:
                #   j -> j     (new job phase 1 after phase-1 completion OR phase 2 after phase-2 completion)
                #   j -> j+1   (phase-1 completion, new job enters phase 2)
                #   j -> j-1   (phase-2 completion, new job enters phase 1)
                x_same, y_same = _coords(level - 1, j, n)
                _arrow(ax, x0, y0, x_same, y_same, None, color="black", lw=1.0, rad=0.00)
                if j < n:
                    x_up, y_up = _coords(level - 1, j + 1, n)
                    _arrow(ax, x0, y0, x_up, y_up, None, color="black", lw=1.0, rad=0.12)
                if j > 0:
                    x_dn, y_dn = _coords(level - 1, j - 1, n)
                    _arrow(ax, x0, y0, x_dn, y_dn, None, color="black", lw=1.0, rad=-0.12)


def _draw_rcs_negative_remove(ax, n: int, max_level: int) -> None:
    """Negative removals for RCS (REMOVE semantics): downward transitions in service."""

    def _maybe_frac_label(num: int, den: int) -> str | None:
        """Hide the trivial label δ, show only fractional δ/k etc."""
        s = _fmt_delta(num, den)
        return None if s == r"$\delta$" else s

    for level in range(1, max_level + 1):
        busy = min(level, n)
        m = min(level, n)
        for j in range(m + 1):
            x0, y0 = _coords(level, j, n)
            if busy - j > 0:
                x1, y1 = _coords(level - 1, j, n)
                dx = x1 - x0
                _arrow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    _maybe_frac_label(busy - j, busy) if level <= n else None,
                    color="#d62728",
                    lw=1.4,
                    ls="--",
                    rad=0.18,
                    # Put the "same-j" label a bit higher to avoid overlap with (j-1) label.
                    label_offset=((0.42 if dx > 0 else -0.42), 0.30),
                    fontsize=12,
                )
            if j > 0:
                x2, y2 = _coords(level - 1, j - 1, n)
                dx = x2 - x0
                _arrow(
                    ax,
                    x0,
                    y0,
                    x2,
                    y2,
                    _maybe_frac_label(j, busy) if level <= n else None,
                    color="#d62728",
                    lw=1.4,
                    ls="--",
                    rad=-0.18,
                    # Put the "j-1" label slightly lower to separate it from the (same-j) label.
                    label_offset=((0.52 if dx > 0 else -0.52), -0.06),
                    fontsize=12,
                )


def _plot_rcs_negative_arrivals(n: int = 3, max_level: int = 4) -> plt.Figure:
    """RCS negative arrivals: only transitions caused by negative arrivals (delta)."""
    fig, ax = _setup_axes(max_level)
    _draw_base_nodes(ax, n, max_level)
    _draw_rcs_negative_remove(ax, n, max_level)
    fig.tight_layout()
    return fig


def _plot_rcs_service(n: int = 3, max_level: int = 4) -> plt.Figure:
    """RCS service: only service completions (mu), without negative effects."""
    fig, ax = _setup_axes(max_level)
    _draw_base_nodes(ax, n, max_level)
    _draw_service_base(ax, n, max_level)
    fig.tight_layout()
    return fig


def _draw_disaster_nodes(ax, n: int, max_level: int) -> float:
    """Draw disaster states D_i; returns x-coordinate used for D_i nodes."""
    d_x = 6.5
    for level in range(1, max_level + 1):
        y = -float(level) * 1.55
        _draw_node(ax, d_x, y, rf"$D_{level}$", radius=NODE_RADIUS)
    return d_x


def _draw_disaster_transitions(ax, n: int, max_level: int, d_x: float) -> None:
    # transitions to D_i with rate delta (red dashed)
    for level in range(1, max_level + 1):
        m = min(level, n)
        for j in range(m + 1):
            x0, y0 = _coords(level, j, n)
            _arrow(
                ax,
                x0,
                y0,
                d_x,
                -float(level) * 1.55,
                None if j > 0 else r"$\delta$",
                color="#d62728",
                lw=1.4,
                ls="--",
                rad=0.08,
                label_offset=(0.0, 0.18),
                fontsize=12,
            )
    # return to (0,0) with rate gamma
    x00, y00 = _coords(0, 0, n)
    # Drawing gamma arrows from every D_i makes the diagram messy (many arcs intersect).
    # Instead, draw a single representative gamma transition and add a note "from any D_i".
    _arrow(
        ax,
        d_x,
        -1.0 * 1.55,
        x00,
        y00,
        r"$\gamma$",
        color="#d62728",
        lw=1.5,
        ls="-",
        rad=-0.26,
        label_offset=(0.35, 0.30),
        fontsize=12,
    )
    ax.text(
        d_x + 0.20,
        -1.0 * 1.55 + 0.35,
        r"из любого $D_i$",
        fontsize=11,
        color="#d62728",
        ha="left",
        va="bottom",
        zorder=2,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
    )


def _plot_disaster_negative_arrivals(n: int = 3, max_level: int = 4) -> plt.Figure:
    """Disaster negative arrivals: transitions to disaster states D_i and recovery to (0,0)."""
    fig, ax = _setup_axes(max_level)
    _draw_base_nodes(ax, n, max_level)
    d_x = _draw_disaster_nodes(ax, n, max_level)
    _draw_disaster_transitions(ax, n, max_level, d_x)
    fig.tight_layout()
    return fig


def _plot_disaster_service(n: int = 3, max_level: int = 4) -> plt.Figure:
    """Disaster service: only service completions (mu), without negative effects."""
    fig, ax = _setup_axes(max_level)
    _draw_base_nodes(ax, n, max_level)
    _draw_service_base(ax, n, max_level)
    fig.tight_layout()
    return fig


# -----------------------------
# Pretty diagrams requested in chat
# -----------------------------

Y_STEP_PRETTY = 2.0  # vertical spacing between levels (was 1.55; larger = more room for labels)


def _setup_axes_pretty(max_level: int) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_axis_off()
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-Y_STEP_PRETTY * (max_level + 0.8), 1.25)
    # Level labels on the left
    for i in range(max_level + 1):
        ax.text(-5.6, -i * Y_STEP_PRETTY, f"{i}", fontsize=16, ha="right", va="center", color="#111111")
    return fig, ax


def _rcs_pretty_pos(level: int, idx: int, count: int) -> tuple[float, float]:
    y = -float(level) * Y_STEP_PRETTY
    dx = 2.35
    if level == 4:
        dx = 2.65  # more space for long labels and crossings
    x = (idx - (count - 1) / 2.0) * dx
    return x, y


def _plot_rcs_negative_arrivals_pretty(n: int = 3, max_level: int = 4) -> plt.Figure:
    """
    Pretty RCS diagram: only transitions caused by negative arrivals (δ),
    levels 0..4 (with y1/y2 mix on 4->3).
    """
    fig, ax = _setup_axes_pretty(max_level)

    # Nodes: levels 0..2 use busy=level, levels 3..4 use busy=n=3 with b=0..3 ordering 30,21,12,03.
    nodes: dict[tuple[int, str], tuple[float, float]] = {}

    def add_level(level: int, labels: list[str]) -> None:
        for idx, lab in enumerate(labels):
            x, y = _rcs_pretty_pos(level, idx, len(labels))
            nodes[(level, lab)] = (x, y)
            w = 1.35 if level == 4 else 1.15
            _draw_box_node(ax, x, y, lab, level=level, width=w, height=0.62, dashed=False, fontsize=16)

    add_level(0, ["00"])
    add_level(1, ["10", "01"])
    add_level(2, ["20", "11", "02"])
    add_level(3, ["30", "21", "12", "03"])
    add_level(4, ["30", "21", "12", "03"])

    # Helper to get (a,b) from label.
    def ab(lab: str) -> tuple[int, int]:
        return int(lab[0]), int(lab[1])

    # Edges for levels 1..3: δ * a/k or δ * b/k.
    for level in (1, 2, 3):
        k = level
        labels = ["10", "01"] if level == 1 else (["20", "11", "02"] if level == 2 else ["30", "21", "12", "03"])
        for lab in labels:
            a, b = ab(lab)
            x0, y0 = nodes[(level, lab)]
            # remove phase-1 job: (a-1,b) on level-1
            if a > 0:
                to_lab = f"{a-1}{b}"
                x1, y1 = nodes[(level - 1, to_lab)]
                _arrow(
                    ax,
                    x0,
                    y0,
                    x1,
                    y1,
                    _delta_frac(a, k),
                    color="#333333",
                    lw=1.8,
                    rad=0.10,
                    label_offset=(0.0, 0.22),
                    fontsize=13,
                    start_radius=BOX_R,
                    end_radius=BOX_R,
                )
            # remove phase-2 job: (a,b-1) on level-1
            if b > 0:
                to_lab = f"{a}{b-1}"
                x2, y2 = nodes[(level - 1, to_lab)]
                _arrow(
                    ax,
                    x0,
                    y0,
                    x2,
                    y2,
                    _delta_frac(b, k),
                    color="#333333",
                    lw=1.8,
                    rad=-0.10,
                    label_offset=(0.0, -0.02),
                    fontsize=13,
                    start_radius=BOX_R,
                    end_radius=BOX_R,
                )

    # Edges for level 4 -> 3 with y1/y2.
    # Crossing arrows: spread labels noticeably left/right to avoid overlap (one left, one right).
    # 30 -> 30 (δ y1), 30 -> 21 (δ y2) — 30→21 crosses with 21→30
    x0, y0 = nodes[(4, "30")]
    x1, y1 = nodes[(3, "30")]
    _arrow(
        ax,
        x0,
        y0,
        x1,
        y1,
        _m(r"\delta y_1"),
        color="#333333",
        lw=1.8,
        rad=0.0,
        label_offset=(-0.5, 0.22),
        fontsize=12,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )
    x2, y2 = nodes[(3, "21")]
    _arrow(
        ax,
        x0,
        y0,
        x2,
        y2,
        _m(r"\delta y_2"),
        color="#333333",
        lw=1.8,
        rad=0.14,
        label_offset=(1.0, 0.18),
        fontsize=12,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )

    # 03 -> 12 (δ y1), 03 -> 03 (δ y2) — 03→12 crosses with 12→03
    x0, y0 = nodes[(4, "03")]
    x1, y1 = nodes[(3, "12")]
    _arrow(
        ax,
        x0,
        y0,
        x1,
        y1,
        _m(r"\delta y_1"),
        color="#333333",
        lw=1.8,
        rad=-0.14,
        label_offset=(-1.0, 0.18),
        fontsize=12,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )
    x2, y2 = nodes[(3, "03")]
    _arrow(
        ax,
        x0,
        y0,
        x2,
        y2,
        _m(r"\delta y_2"),
        color="#333333",
        lw=1.8,
        rad=0.0,
        label_offset=(0.5, 0.22),
        fontsize=12,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )

    # 21 -> 30 (δ*1/3*y1), 21 -> 21 (δ*(2/3*y1+1/3*y2)), 21 -> 12 (δ*2/3*y2)
    # 21→30 crosses 30→21: label left. 21→12 crosses 12→21: label right.
    x0, y0 = nodes[(4, "21")]
    x1, y1 = nodes[(3, "30")]
    _arrow(
        ax,
        x0,
        y0,
        x1,
        y1,
        _delta_y(1, 3, r"y_1"),
        color="#333333",
        lw=1.8,
        rad=-0.18,
        label_offset=(-1.0, 0.28),
        fontsize=11,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )
    x2, y2 = nodes[(3, "21")]
    _arrow(
        ax,
        x0,
        y0,
        x2,
        y2,
        _delta_combo(2, 3, 1, 3),
        color="#333333",
        lw=1.8,
        rad=0.0,
        label_offset=(0.0, 0.25),
        fontsize=10,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )
    x3, y3 = nodes[(3, "12")]
    _arrow(
        ax,
        x0,
        y0,
        x3,
        y3,
        _delta_y(2, 3, r"y_2"),
        color="#333333",
        lw=1.8,
        rad=0.18,
        label_offset=(1.0, 0.10),
        fontsize=11,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )

    # 12 -> 21 (δ*2/3*y1), 12 -> 12 (δ*(1/3*y1+2/3*y2)), 12 -> 03 (δ*1/3*y2)
    # 12→21 crosses 21→12: label left. 12→03 crosses 03→12: label right.
    x0, y0 = nodes[(4, "12")]
    x1, y1 = nodes[(3, "21")]
    _arrow(
        ax,
        x0,
        y0,
        x1,
        y1,
        _delta_y(2, 3, r"y_1"),
        color="#333333",
        lw=1.8,
        rad=-0.18,
        label_offset=(-1.0, 0.10),
        fontsize=11,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )
    x2, y2 = nodes[(3, "12")]
    _arrow(
        ax,
        x0,
        y0,
        x2,
        y2,
        _m(r"\delta\left(\frac{1}{3}y_1+\frac{2}{3}y_2\right)"),
        color="#333333",
        lw=1.8,
        rad=0.0,
        label_offset=(0.0, 0.25),
        fontsize=10,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )
    x3, y3 = nodes[(3, "03")]
    _arrow(
        ax,
        x0,
        y0,
        x3,
        y3,
        _delta_y(1, 3, r"y_2"),
        color="#333333",
        lw=1.8,
        rad=0.18,
        label_offset=(1.0, 0.28),
        fontsize=11,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )

    fig.tight_layout()
    return fig


def _plot_disaster_negative_arrivals_pretty(n: int = 3, max_level: int = 4) -> plt.Figure:
    """
    Pretty DISASTER diagram (clear semantics):
    - levels 0..4
    - δ transitions from each microstate to D on the same level
    - γ transitions D4->D3->D2->D1->00
    """
    fig, ax = _setup_axes_pretty(max_level)

    # Place D on the right to force fan-in arrows.
    d_x = 4.9
    row_y = {i: -i * Y_STEP_PRETTY for i in range(max_level + 1)}

    # 00 at top, centered a bit left from D to keep balance, but align with D for a clean γ arrow.
    x00, y00 = (d_x, row_y[0])
    _draw_box_node(ax, x00, y00, "00", level=0, width=1.15, height=0.62, dashed=False, fontsize=16)

    # Microstates by row:
    rows: dict[int, list[str]] = {
        1: ["10", "01"],
        2: ["20", "11", "02"],
        3: ["30", "21", "12", "03"],
        4: ["30", "21", "12", "03"],
    }

    def row_x_positions(level: int, count: int) -> list[float]:
        dx = 1.70 if count <= 3 else 1.55
        left = -4.1
        return [left + i * dx for i in range(count)]

    # Draw nodes for rows 1..4 and D nodes.
    micro_pos: dict[tuple[int, str], tuple[float, float]] = {}
    d_pos: dict[int, tuple[float, float]] = {}
    for level in range(1, max_level + 1):
        labs = rows[level]
        xs = row_x_positions(level, len(labs))
        y = row_y[level]
        for x, lab in zip(xs, labs):
            micro_pos[(level, lab)] = (x, y)
            _draw_box_node(ax, x, y, lab, level=level, width=1.15, height=0.62, dashed=False, fontsize=16)
        d_pos[level] = (d_x, y)
        _draw_box_node(ax, d_x, y, "D", level=level, width=1.15, height=0.62, dashed=True, fontsize=16)

    # δ arrows: micro -> D (same row), rightwards, slightly different curvature to avoid overlap.
    for level in range(1, max_level + 1):
        labs = rows[level]
        mid = (len(labs) - 1) / 2.0
        for i, lab in enumerate(labs):
            x0, y0 = micro_pos[(level, lab)]
            x1, y1 = d_pos[level]
            rad = (i - mid) * 0.08
            # Spread labels near D to avoid stacking.
            label_offset = (1.45, (i - mid) * 0.26)
            _arrow(
                ax,
                x0,
                y0,
                x1,
                y1,
                _m(r"\delta"),
                color="#333333",
                lw=1.8,
                rad=rad,
                label_offset=label_offset,
                fontsize=12,
                start_radius=BOX_R,
                end_radius=BOX_R,
            )

    # γ chain: D4->D3->D2->D1->00
    for level_from, level_to in ((4, 3), (3, 2), (2, 1)):
        x0, y0 = d_pos[level_from]
        x1, y1 = d_pos[level_to]
        _arrow(
            ax,
            x0,
            y0,
            x1,
            y1,
            _m(r"\gamma"),
            color="#333333",
            lw=1.8,
            rad=0.0,
            label_offset=(0.25, 0.05),
            fontsize=12,
            start_radius=BOX_R,
            end_radius=BOX_R,
        )
    x0, y0 = d_pos[1]
    _arrow(
        ax,
        x0,
        y0,
        x00,
        y00,
        _m(r"\gamma"),
        color="#333333",
        lw=1.8,
        rad=0.0,
        label_offset=(0.25, 0.05),
        fontsize=12,
        start_radius=BOX_R,
        end_radius=BOX_R,
    )

    fig.tight_layout()
    return fig


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir_rcs = base_dir / "negative_queues_figures" / "rcs" / "diagrams"
    out_dir_dis = base_dir / "negative_queues_figures" / "disaster" / "diagrams"
    out_dir_rcs.mkdir(parents=True, exist_ok=True)
    out_dir_dis.mkdir(parents=True, exist_ok=True)

    out_rcs_a = out_dir_rcs / "rcs_arrivals.png"
    out_rcs_b = out_dir_rcs / "rcs_service.png"
    out_dis_a = out_dir_dis / "disaster_arrivals.png"
    out_dis_b = out_dir_dis / "disaster_service.png"
    out_rcs_pretty = out_dir_rcs / "rcs_negative_arrivals_pretty.png"
    out_rcs_pretty_svg = out_dir_rcs / "rcs_negative_arrivals_pretty.svg"
    out_dis_pretty = out_dir_dis / "disaster_negative_arrivals_pretty.png"
    out_dis_pretty_svg = out_dir_dis / "disaster_negative_arrivals_pretty.svg"

    # Make the output consistent across environments.
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 12,
            "mathtext.fontset": "cm",
            "font.family": "serif",
        }
    )

    # For RCS we show exact δ-transitions only for i<=n; for i>n the article uses μ'-approximation.
    fig1 = _plot_rcs_negative_arrivals(n=3, max_level=3)
    fig1.savefig(out_rcs_a, bbox_inches="tight")
    plt.close(fig1)

    fig2 = _plot_rcs_service(n=3, max_level=4)
    fig2.savefig(out_rcs_b, bbox_inches="tight")
    plt.close(fig2)

    fig3 = _plot_disaster_negative_arrivals(n=3, max_level=4)
    fig3.savefig(out_dis_a, bbox_inches="tight")
    plt.close(fig3)

    fig4 = _plot_disaster_service(n=3, max_level=4)
    fig4.savefig(out_dis_b, bbox_inches="tight")
    plt.close(fig4)

    # Pretty diagrams used in the chat (colored, fixed layout, levels 0..4).
    fig5 = _plot_rcs_negative_arrivals_pretty(n=3, max_level=4)
    fig5.savefig(out_rcs_pretty, bbox_inches="tight")
    fig5.savefig(out_rcs_pretty_svg, bbox_inches="tight")
    plt.close(fig5)

    fig6 = _plot_disaster_negative_arrivals_pretty(n=3, max_level=4)
    fig6.savefig(out_dis_pretty, bbox_inches="tight")
    fig6.savefig(out_dis_pretty_svg, bbox_inches="tight")
    plt.close(fig6)

    print(f"Saved: {out_rcs_a}")
    print(f"Saved: {out_rcs_b}")
    print(f"Saved: {out_dis_a}")
    print(f"Saved: {out_dis_b}")
    print(f"Saved: {out_rcs_pretty}")
    print(f"Saved: {out_rcs_pretty_svg}")
    print(f"Saved: {out_dis_pretty}")
    print(f"Saved: {out_dis_pretty_svg}")


if __name__ == "__main__":
    main()
