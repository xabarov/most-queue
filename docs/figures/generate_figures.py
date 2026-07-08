#!/usr/bin/env python3
"""Generate schematic figures for docs/models.md.

Usage:
    python docs/figures/generate_figures.py          # regenerate all PNGs next to this script

Adding a figure for a new model: write a `fig_<name>()` function that returns a Figure,
register it in FIGURES, rerun the script and embed `figures/<name>.png` in docs/models.md.

Palette: colorblind-safe categorical set (validated, worst adjacent CVD dE=24.2).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# --- palette (light mode) ---
BLUE = "#2a78d6"  # customers / series 1
AQUA = "#1baf7a"  # servers / series 2
YELLOW = "#eda100"  # series 3
GREEN = "#008300"  # series 4
VIOLET = "#4a3aa7"  # series 5
RED = "#e34948"  # negative customers / series 6
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

OUT_DIR = Path(__file__).parent

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


# ---------------------------------------------------------------- helpers
def _clean_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_customer(ax, x, y, color=BLUE, r=0.16, label=None, alpha=1.0):
    ax.add_patch(Circle((x, y), r, color=color, alpha=alpha, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", zorder=4, fontweight="bold")


def draw_queue(ax, x, y, n_slots=5, occupied=3, w=0.42, h=0.44, occ_color=BLUE):
    """Horizontal queue: rightmost slots are closest to the server."""
    for i in range(n_slots):
        sx = x + i * w
        ax.add_patch(
            FancyBboxPatch(
                (sx, y - h / 2),
                w * 0.9,
                h,
                boxstyle="round,pad=0.02",
                fc="white",
                ec=MUTED,
                lw=1.1,
                zorder=2,
            )
        )
        if i >= n_slots - occupied:
            draw_customer(ax, sx + w * 0.45, y, color=occ_color, r=h * 0.32)


def draw_server(ax, x, y, r=0.34, color=AQUA, label="μ", sub=None, busy_color=None):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=4)
    if busy_color:
        draw_customer(ax, x, y + r + 0.18, color=busy_color, r=0.13)
    if sub:
        ax.text(x, y - r - 0.22, sub, ha="center", va="top", fontsize=8, color=INK2)


def draw_arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.6, style="-|>", ls: "str | tuple" = "-", zorder=2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            color=color,
            lw=lw,
            linestyle=ls,
            zorder=zorder,
        )
    )


def _title(ax, text, x=0.5):
    ax.set_title(text, fontsize=11.5, color=INK, pad=10, x=x)


# ---------------------------------------------------------------- figures
def fig_fifo_mmn():
    """M/M/c: arrivals -> queue -> c servers -> departures."""
    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=150)
    _clean_axes(ax, (-0.4, 10.6), (-1.5, 1.6))
    # arrivals
    ax.text(0.1, 0.72, "поток заявок, λ", fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.55, 1.1]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.45, 0, 2.25, 0)
    # queue
    draw_queue(ax, 2.45, 0, n_slots=5, occupied=3)
    ax.text(3.5, -0.62, "очередь (FIFO)", fontsize=9, color=INK2, ha="center")
    draw_arrow(ax, 4.75, 0, 5.55, 0)
    # servers
    for i, sy in enumerate([1.0, 0.0, -1.0]):
        draw_server(ax, 6.3, sy, label="μ", sub=None)
        draw_arrow(ax, 5.62, 0, 5.94, sy, lw=1.1)
        draw_arrow(ax, 6.66, sy, 7.5, 0, lw=1.1)
    ax.text(6.3, -1.45, "c приборов", fontsize=9, color=INK2, ha="center")
    # departures
    draw_arrow(ax, 7.55, 0, 8.4, 0)
    for i, xx in enumerate([8.7, 9.25, 9.8]):
        draw_customer(ax, xx, 0, color=GREEN, alpha=1.0 - 0.27 * i)
    ax.text(9.25, 0.6, "обслуженные", fontsize=9, color=INK2, ha="center")
    _title(ax, "Классическая СМО M/M/c: общая очередь и c одинаковых приборов")
    return fig


def _mg1_schedule(jobs, discipline):
    """Tiny deterministic single-server scheduler for the timeline figure.

    jobs: list of (name, arrival, size). Returns dict name -> list of (start, end) service pieces.
    """
    t, pieces = 0.0, {name: [] for name, _, _ in jobs}
    remaining = {name: size for name, _, size in jobs}
    arrival = {name: a for name, a, _ in jobs}
    done = set()
    current = None
    eps = 1e-9
    while len(done) < len(jobs):
        avail = [n for n in remaining if arrival[n] <= t + eps and n not in done]
        if not avail:
            t = min(a for n, a in arrival.items() if n not in done)
            continue
        if discipline == "FCFS":
            pick = min(avail, key=lambda n: arrival[n])
        elif discipline == "SJF":  # non-preemptive by original size
            pick = current if current in avail else min(avail, key=lambda n: remaining[n])
        else:  # SRPT, preemptive by remaining size
            pick = min(avail, key=lambda n: remaining[n])
        if discipline == "SJF" and current is None:
            current = pick
        # run until pick finishes or (for SRPT) next arrival
        future = [a for n, a in arrival.items() if a > t + eps and n not in done]
        horizon = min(future) if (discipline == "SRPT" and future) else float("inf")
        run = min(remaining[pick], horizon - t)
        if pieces[pick] and abs(pieces[pick][-1][1] - t) < eps:
            pieces[pick][-1] = (pieces[pick][-1][0], t + run)
        else:
            pieces[pick].append((t, t + run))
        remaining[pick] -= run
        t += run
        if remaining[pick] < eps:
            done.add(pick)
            if current == pick:
                current = None
    return pieces


def fig_disciplines_timeline():
    """FCFS vs SJF vs SRPT on the same job set."""
    jobs = [("A", 0.0, 5.0), ("B", 1.0, 2.0), ("C", 2.0, 1.0), ("D", 3.0, 3.0)]
    colors = {"A": BLUE, "B": AQUA, "C": YELLOW, "D": VIOLET}
    disciplines = ["FCFS", "SJF", "SRPT"]
    fig, axes = plt.subplots(len(disciplines), 1, figsize=(8.2, 4.6), dpi=150, sharex=True)
    for ax, disc in zip(axes, disciplines):
        pieces = _mg1_schedule(jobs, disc)
        finish = {n: max(e for _, e in p) for n, p in pieces.items()}
        for name, arr, _size in jobs:
            y = 0
            for s, e in pieces[name]:
                ax.barh(y, e - s, left=s, height=0.62, color=colors[name], edgecolor=SURFACE, linewidth=1.2)
                ax.text((s + e) / 2, y, name, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            ax.plot([arr], [0.55], marker="v", color=colors[name], markersize=7, clip_on=False)
        mean_t = sum(finish[n] - a for n, a, _ in jobs) / len(jobs)
        ax.set_ylabel(disc, rotation=0, ha="right", va="center", fontsize=10, color=INK)
        ax.text(11.15, 0, f"ср. время\nв системе: {mean_t:.2f}", fontsize=8.5, color=INK2, va="center")
        ax.set_yticks([])
        ax.set_xlim(0, 11.1)
        ax.set_ylim(-0.6, 0.9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("время", fontsize=9, color=INK2)
    fig.suptitle(
        "Один прибор, одни и те же заявки: A(размер 5), B(2), C(1), D(3) — метки ▼ это моменты прихода.\n"
        "SJF выбирает короткую заявку в момент освобождения прибора, SRPT может прервать текущую.",
        fontsize=9.5,
        color=INK2,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_priority():
    """Two priority classes, PR vs NP."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.1, 2.0))
    # two queues
    draw_queue(ax, 1.2, 1.0, n_slots=4, occupied=2, occ_color=RED)
    ax.text(0.9, 1.0, "класс 1\n(важные)", fontsize=8.5, color=INK2, ha="right", va="center")
    draw_queue(ax, 1.2, -1.0, n_slots=4, occupied=3, occ_color=BLUE)
    ax.text(0.9, -1.0, "класс 2\n(обычные)", fontsize=8.5, color=INK2, ha="right", va="center")
    draw_arrow(ax, 3.15, 1.0, 4.2, 0.15)
    draw_arrow(ax, 3.15, -1.0, 4.2, -0.15, color=MUTED)
    draw_server(ax, 4.7, 0, r=0.4, label="μ", busy_color=BLUE)
    draw_arrow(ax, 5.15, 0, 5.9, 0)
    ax.text(4.7, -0.85, "прибор занят\nобычной заявкой", fontsize=8, color=INK2, ha="center", va="top")
    # annotations PR / NP
    ax.text(
        6.3,
        1.15,
        "PR (прерывающий): важная заявка вытесняет\nобычную с прибора, та дообслужится позже",
        fontsize=9,
        color=INK,
        va="center",
    )
    draw_arrow(ax, 6.15, 1.15, 5.0, 0.35, color=RED, ls=(0, (4, 3)), lw=1.4)
    ax.text(
        6.3,
        -1.2,
        "NP (непрерывающий): важная заявка ждёт\nконца текущего обслуживания, но обгоняет очередь",
        fontsize=9,
        color=INK,
        va="center",
    )
    _title(ax, "Приоритетное обслуживание: отдельная очередь на каждый класс", x=0.44)
    return fig


def fig_vacations():
    """Server state cycle: serving -> cooling delay -> cooling -> warmup -> serving."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.3, 11.4), (-1.7, 1.5))
    states = [
        ("обслуживание", AQUA, "очередь не пуста —\nприбор работает"),
        ("задержка", YELLOW, "очередь опустела,\nещё ждём (delay)"),
        ("охлаждение /\nотпуск", VIOLET, "прибор выключен\n(cooling, vacation)"),
        ("прогрев", BLUE, "пришла заявка —\nвключаемся (warm-up)"),
    ]
    xs = [0.9, 3.7, 6.5, 9.3]
    for (name, color, sub), x in zip(states, xs):
        ax.add_patch(
            FancyBboxPatch((x - 1.12, -0.42), 2.24, 0.95, boxstyle="round,pad=0.06", fc=color, ec="none", zorder=3)
        )
        ax.text(x, 0.06, name, ha="center", va="center", fontsize=8.6, color="white", fontweight="bold", zorder=4)
        ax.text(x, -0.62, sub, ha="center", va="top", fontsize=8, color=INK2)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        draw_arrow(ax, x1 + 1.2, 0.1, x2 - 1.2, 0.1, lw=1.6)
    # loop back
    draw_arrow(ax, 9.3, 0.62, 9.3, 1.06, lw=1.4)
    ax.plot([0.9, 0.9, 9.3], [1.06, 1.06, 1.06], color=INK2, lw=1.4)
    ax.plot([0.9, 0.9], [1.06, 0.66], color=INK2, lw=1.4)
    draw_arrow(ax, 0.9, 0.8, 0.9, 0.6, lw=1.4)
    _title(ax, "Vacation-модели: жизненный цикл прибора (у конкретной модели — своё подмножество фаз)")
    return fig


def fig_negative():
    """RCS vs disaster."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=150)
    _clean_axes(ax, (-0.4, 11.4), (-2.3, 2.75))
    for y0, kind in [(1.15, "RCS"), (-1.15, "disaster")]:
        draw_queue(ax, 1.4, y0, n_slots=4, occupied=3)
        draw_server(ax, 4.6, y0, r=0.36, label="μ", busy_color=BLUE)
        draw_arrow(ax, 0.4, y0, 1.3, y0)
        draw_arrow(ax, 3.35, y0, 4.15, y0)
        draw_arrow(ax, 5.0, y0, 5.7, y0)
        # negative arrival
        draw_customer(ax, 6.6, y0 + 0.75, color=RED, r=0.15, label="−")
        if kind == "RCS":
            draw_arrow(ax, 6.45, y0 + 0.68, 4.85, y0 + 0.28, color=RED, lw=1.6, ls=(0, (4, 3)))
            ax.text(
                7.0,
                y0 + 0.72,
                "RCS: отрицательная заявка «выбивает»\nтолько ту, что на приборе",
                fontsize=9,
                color=INK,
                va="center",
            )
        else:
            draw_arrow(ax, 6.45, y0 + 0.62, 4.9, y0 + 0.25, color=RED, lw=1.6, ls=(0, (4, 3)))
            draw_arrow(ax, 6.42, y0 + 0.66, 2.6, y0 + 0.32, color=RED, lw=1.6, ls=(0, (4, 3)))
            ax.text(
                7.0,
                y0 + 0.72,
                "Disaster: катастрофа очищает всю систему —\nи очередь, и прибор",
                fontsize=9,
                color=INK,
                va="center",
            )
    ax.text(0.35, 2.45, "λ — обычные заявки,  λ⁻ — отрицательные (красные)", fontsize=9, color=INK2)
    _title(ax, "Отрицательные заявки: два сценария воздействия", x=0.45)
    return fig


def fig_fork_join():
    """Fork-Join (n,k)."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2), dpi=150)
    _clean_axes(ax, (-0.5, 11.3), (-1.9, 1.9))
    draw_customer(ax, 0.4, 0, r=0.22)
    ax.text(0.4, 0.5, "заявка", fontsize=9, color=INK2, ha="center")
    draw_arrow(ax, 0.7, 0, 1.55, 0)
    ax.text(1.95, 0.55, "fork:\nделим на части", fontsize=8.5, color=INK2, ha="center")
    ys = [1.15, 0.0, -1.15]
    for i, y in enumerate(ys):
        draw_arrow(ax, 1.75, 0, 2.9, y, lw=1.2)
        draw_customer(ax, 3.15, y, r=0.14, color=[BLUE, AQUA, YELLOW][i])
        draw_arrow(ax, 3.4, y, 4.15, y, lw=1.2)
        draw_server(ax, 4.6, y, r=0.3, label="μ")
        draw_arrow(ax, 4.95, y, 6.1, 0.18 * (1 if y > 0 else -1) if y != 0 else 0, lw=1.2)
    ax.add_patch(
        FancyBboxPatch((6.15, -0.55), 1.5, 1.1, boxstyle="round,pad=0.05", fc="white", ec=MUTED, lw=1.2, zorder=2)
    )
    ax.text(6.9, 0, "join:\nждём все\nчасти", ha="center", va="center", fontsize=8.5, color=INK)
    draw_arrow(ax, 7.75, 0, 8.6, 0)
    draw_customer(ax, 8.9, 0, r=0.22, color=GREEN)
    ax.text(9.0, 0.55, "готово, когда завершилась\nсамая медленная часть", fontsize=8.5, color=INK2, ha="center")
    _title(ax, "Fork-Join: заявка обслуживается по частям параллельно", x=0.45)
    return fig


def fig_batch():
    """Batch arrivals."""
    fig, ax = plt.subplots(figsize=(8.2, 2.6), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-1.5, 1.5))
    # groups arriving
    for gx, size in [(0.6, 3), (2.3, 1), (3.6, 2)]:
        for i in range(size):
            draw_customer(ax, gx + 0.34 * i, 0, r=0.15)
        ax.text(gx + 0.17 * (size - 1), -0.5, f"пачка ×{size}", fontsize=8, color=INK2, ha="center")
    draw_arrow(ax, 4.6, 0, 5.35, 0)
    draw_queue(ax, 5.55, 0, n_slots=4, occupied=2)
    draw_arrow(ax, 7.5, 0, 8.2, 0)
    draw_server(ax, 8.65, 0, r=0.34, label="μ")
    draw_arrow(ax, 9.05, 0, 9.8, 0)
    draw_customer(ax, 10.1, 0, color=GREEN, r=0.16)
    ax.text(
        2.2,
        1.1,
        "заявки приходят группами случайного размера (моменты прихода — пуассоновские)",
        fontsize=9,
        color=INK2,
        ha="left",
    )
    _title(ax, "Пакетное поступление M[X]/M/1: приходят пачками, обслуживаются по одной", x=0.46)
    return fig


def fig_impatience():
    """Impatient customers."""
    fig, ax = plt.subplots(figsize=(8.2, 2.8), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.0, 1.5))
    draw_arrow(ax, 0.3, 0, 1.1, 0)
    draw_queue(ax, 1.3, 0, n_slots=5, occupied=4)
    draw_arrow(ax, 3.7, 0, 4.45, 0)
    draw_server(ax, 4.9, 0, r=0.34, label="μ", busy_color=BLUE)
    draw_arrow(ax, 5.3, 0, 6.05, 0)
    draw_customer(ax, 6.35, 0, color=GREEN, r=0.16)
    # an impatient one leaves
    leave_x = 2.05
    draw_arrow(ax, leave_x, -0.32, leave_x, -1.15, color=RED, lw=1.6, ls=(0, (4, 3)))
    draw_customer(ax, leave_x, -1.35, color=RED, r=0.15)
    ax.text(
        3.0,
        -1.35,
        "не дождалась: у каждой заявки свой «запас терпения»,\nпо его истечении она уходит из очереди",
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(ax, "Нетерпеливые заявки (M/M/1+M): ожидание ограничено терпением клиента", x=0.45)
    return fig


def fig_engset():
    """Finite sources."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.6, 11.2), (-1.9, 1.9))
    ys = [1.2, 0.4, -0.4, -1.2]
    for i, y in enumerate(ys):
        ax.add_patch(
            FancyBboxPatch((0.0, y - 0.26), 1.25, 0.52, boxstyle="round,pad=0.04", fc="white", ec=MUTED, lw=1.1)
        )
        ax.text(0.62, y, f"источник {i + 1}", ha="center", va="center", fontsize=8, color=INK)
        draw_arrow(ax, 1.35, y, 2.6, 0.12 * (1 if y > 0 else -1), lw=1.0)
    ax.text(0.62, 1.75, "N источников (станков, абонентов)", fontsize=9, color=INK2, ha="left")
    draw_queue(ax, 2.9, 0, n_slots=3, occupied=1)
    draw_arrow(ax, 4.35, 0, 5.05, 0)
    draw_server(ax, 5.5, 0, r=0.34, label="μ")
    # return loop
    ax.plot([5.9, 7.0, 7.0, 0.62, 0.62], [0, 0, -1.75, -1.75, -1.5], color=INK2, lw=1.3)
    draw_arrow(ax, 0.62, -1.6, 0.62, -1.5, lw=1.3)
    ax.text(
        7.3,
        -0.9,
        "обслуженный источник возвращается «в работу»\nи лишь потом может снова прислать заявку:\nчем больше заявок внутри, тем слабее входящий поток",
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(ax, "Закрытая система (Engset): заявки порождает конечное число источников", x=0.42)
    return fig


def fig_loss():
    """Erlang B loss system: no queue, blocked customer is lost."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.4, 2.1))
    ax.text(0.1, 0.65, "поток заявок, λ", fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.6, 1.2]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.55, 0, 2.9, 0)
    ax.text(3.9, 1.75, "очереди нет: только n приборов", fontsize=9, color=INK2, ha="center")
    for sy in [1.05, 0.0, -1.05]:
        draw_server(ax, 3.9, sy, r=0.3, label="μ")
    draw_arrow(ax, 4.3, 0, 5.2, 0)
    draw_customer(ax, 5.5, 0, color=GREEN, r=0.16)
    # blocked customer
    draw_arrow(ax, 2.35, -0.2, 2.35, -1.5, color=RED, lw=1.6, ls=(0, (4, 3)))
    draw_customer(ax, 2.35, -1.75, color=RED, r=0.15)
    ax.text(
        3.1,
        -1.75,
        "все приборы заняты — заявка теряется (блокировка).\nДоля потерянных = формула Эрланга B",
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(ax, "Система с потерями M/M/n/0 (Erlang B): мест для ожидания нет", x=0.45)
    return fig


FIGURES = {
    "fifo_mmn": fig_fifo_mmn,
    "loss": fig_loss,
    "disciplines_timeline": fig_disciplines_timeline,
    "priority": fig_priority,
    "vacations": fig_vacations,
    "negative": fig_negative,
    "fork_join": fig_fork_join,
    "batch": fig_batch,
    "impatience": fig_impatience,
    "engset": fig_engset,
}


def main() -> None:
    """Regenerate every figure into the directory of this script."""
    for name, builder in FIGURES.items():
        fig = builder()
        out = OUT_DIR / f"{name}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
