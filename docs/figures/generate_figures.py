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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

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

# --- bilingual text: `_LANG` is flipped by main() for each render pass ---
_LANG = "en"


def t(en: str, ru: str) -> str:
    return ru if _LANG == "ru" else en


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
    ax.text(0.1, 0.72, t("arrival stream, λ", "поток заявок, λ"), fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.55, 1.1]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.45, 0, 2.25, 0)
    # queue
    draw_queue(ax, 2.45, 0, n_slots=5, occupied=3)
    ax.text(3.5, -0.62, t("queue (FIFO)", "очередь (FIFO)"), fontsize=9, color=INK2, ha="center")
    draw_arrow(ax, 4.75, 0, 5.55, 0)
    # servers
    for i, sy in enumerate([1.0, 0.0, -1.0]):
        draw_server(ax, 6.3, sy, label="μ", sub=None)
        draw_arrow(ax, 5.62, 0, 5.94, sy, lw=1.1)
        draw_arrow(ax, 6.66, sy, 7.5, 0, lw=1.1)
    ax.text(6.3, -1.45, t("c servers", "c приборов"), fontsize=9, color=INK2, ha="center")
    # departures
    draw_arrow(ax, 7.55, 0, 8.4, 0)
    for i, xx in enumerate([8.7, 9.25, 9.8]):
        draw_customer(ax, xx, 0, color=GREEN, alpha=1.0 - 0.27 * i)
    ax.text(9.25, 0.6, t("served", "обслуженные"), fontsize=9, color=INK2, ha="center")
    _title(
        ax,
        t(
            "Classic M/M/c queueing system: one shared queue and c identical servers",
            "Классическая СМО M/M/c: общая очередь и c одинаковых приборов",
        ),
    )
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
        ax.text(
            11.15,
            0,
            t(f"mean time\nin system: {mean_t:.2f}", f"ср. время\nв системе: {mean_t:.2f}"),
            fontsize=8.5,
            color=INK2,
            va="center",
        )
        ax.set_yticks([])
        ax.set_xlim(0, 11.1)
        ax.set_ylim(-0.6, 0.9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel(t("time", "время"), fontsize=9, color=INK2)
    fig.suptitle(
        t(
            "One server, the same jobs: A(size 5), B(2), C(1), D(3) — ▼ marks are arrival instants.\n"
            "SJF picks the shortest job when the server frees up, SRPT may preempt the current one.",
            "Один прибор, одни и те же заявки: A(размер 5), B(2), C(1), D(3) — метки ▼ это моменты прихода.\n"
            "SJF выбирает короткую заявку в момент освобождения прибора, SRPT может прервать текущую.",
        ),
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
    ax.text(0.9, 1.0, t("class 1\n(high)", "класс 1\n(важные)"), fontsize=8.5, color=INK2, ha="right", va="center")
    draw_queue(ax, 1.2, -1.0, n_slots=4, occupied=3, occ_color=BLUE)
    ax.text(
        0.9, -1.0, t("class 2\n(ordinary)", "класс 2\n(обычные)"), fontsize=8.5, color=INK2, ha="right", va="center"
    )
    draw_arrow(ax, 3.15, 1.0, 4.2, 0.15)
    draw_arrow(ax, 3.15, -1.0, 4.2, -0.15, color=MUTED)
    draw_server(ax, 4.7, 0, r=0.4, label="μ", busy_color=BLUE)
    draw_arrow(ax, 5.15, 0, 5.9, 0)
    ax.text(
        4.7,
        -0.85,
        t("server busy with\nan ordinary job", "прибор занят\nобычной заявкой"),
        fontsize=8,
        color=INK2,
        ha="center",
        va="top",
    )
    # annotations PR / NP
    ax.text(
        6.3,
        1.15,
        t(
            "PR (preemptive): a high-priority job pushes\nthe ordinary one off the server, it resumes later",
            "PR (прерывающий): важная заявка вытесняет\nобычную с прибора, та дообслужится позже",
        ),
        fontsize=9,
        color=INK,
        va="center",
    )
    draw_arrow(ax, 6.15, 1.15, 5.0, 0.35, color=RED, ls=(0, (4, 3)), lw=1.4)
    ax.text(
        6.3,
        -1.2,
        t(
            "NP (non-preemptive): a high-priority job waits\nfor the current service to end, but jumps the queue",
            "NP (непрерывающий): важная заявка ждёт\nконца текущего обслуживания, но обгоняет очередь",
        ),
        fontsize=9,
        color=INK,
        va="center",
    )
    _title(
        ax,
        t(
            "Priority service: a separate queue for each class",
            "Приоритетное обслуживание: отдельная очередь на каждый класс",
        ),
        x=0.44,
    )
    return fig


def fig_vacations():
    """Server state cycle: serving -> cooling delay -> cooling -> warmup -> serving."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.3, 11.4), (-1.7, 1.5))
    states = [
        (
            t("service", "обслуживание"),
            AQUA,
            t("queue non-empty —\nserver working", "очередь не пуста —\nприбор работает"),
        ),
        (
            t("delay", "задержка"),
            YELLOW,
            t("queue emptied,\nstill waiting (delay)", "очередь опустела,\nещё ждём (delay)"),
        ),
        (
            t("cooling /\nvacation", "охлаждение /\nотпуск"),
            VIOLET,
            t("server off\n(cooling, vacation)", "прибор выключен\n(cooling, vacation)"),
        ),
        (
            t("warm-up", "прогрев"),
            BLUE,
            t("a job arrived —\npowering up (warm-up)", "пришла заявка —\nвключаемся (warm-up)"),
        ),
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
    _title(
        ax,
        t(
            "Vacation models: the server's life cycle (each model uses its own subset of phases)",
            "Vacation-модели: жизненный цикл прибора (у конкретной модели — своё подмножество фаз)",
        ),
    )
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
                t(
                    'RCS: a negative customer "knocks out"\nonly the job currently on the server',
                    "RCS: отрицательная заявка «выбивает»\nтолько ту, что на приборе",
                ),
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
                t(
                    "Disaster: a disaster clears the whole system —\nboth the queue and the server",
                    "Disaster: катастрофа очищает всю систему —\nи очередь, и прибор",
                ),
                fontsize=9,
                color=INK,
                va="center",
            )
    ax.text(
        0.35,
        2.45,
        t(
            "λ — ordinary jobs,  λ⁻ — negative customers (red)",
            "λ — обычные заявки,  λ⁻ — отрицательные (красные)",
        ),
        fontsize=9,
        color=INK2,
    )
    _title(
        ax,
        t(
            "Negative customers: two impact scenarios",
            "Отрицательные заявки: два сценария воздействия",
        ),
        x=0.45,
    )
    return fig


def fig_fork_join():
    """Fork-Join (n,k)."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2), dpi=150)
    _clean_axes(ax, (-0.5, 11.3), (-1.9, 1.9))
    draw_customer(ax, 0.4, 0, r=0.22)
    ax.text(0.4, 0.5, t("job", "заявка"), fontsize=9, color=INK2, ha="center")
    draw_arrow(ax, 0.7, 0, 1.55, 0)
    ax.text(1.95, 0.55, t("fork:\nsplit into parts", "fork:\nделим на части"), fontsize=8.5, color=INK2, ha="center")
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
    ax.text(
        6.9,
        0,
        t("join:\nwait for all\nparts", "join:\nждём все\nчасти"),
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
    )
    draw_arrow(ax, 7.75, 0, 8.6, 0)
    draw_customer(ax, 8.9, 0, r=0.22, color=GREEN)
    ax.text(
        9.0,
        0.55,
        t("done once the slowest\npart finishes", "готово, когда завершилась\nсамая медленная часть"),
        fontsize=8.5,
        color=INK2,
        ha="center",
    )
    _title(
        ax,
        t(
            "Fork-Join: a job is served in parts in parallel",
            "Fork-Join: заявка обслуживается по частям параллельно",
        ),
        x=0.45,
    )
    return fig


def fig_batch():
    """Batch arrivals."""
    fig, ax = plt.subplots(figsize=(8.2, 2.6), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-1.5, 1.5))
    # groups arriving
    for gx, size in [(0.6, 3), (2.3, 1), (3.6, 2)]:
        for i in range(size):
            draw_customer(ax, gx + 0.34 * i, 0, r=0.15)
        ax.text(
            gx + 0.17 * (size - 1), -0.5, t(f"batch ×{size}", f"пачка ×{size}"), fontsize=8, color=INK2, ha="center"
        )
    draw_arrow(ax, 4.6, 0, 5.35, 0)
    draw_queue(ax, 5.55, 0, n_slots=4, occupied=2)
    draw_arrow(ax, 7.5, 0, 8.2, 0)
    draw_server(ax, 8.65, 0, r=0.34, label="μ")
    draw_arrow(ax, 9.05, 0, 9.8, 0)
    draw_customer(ax, 10.1, 0, color=GREEN, r=0.16)
    ax.text(
        2.2,
        1.1,
        t(
            "jobs arrive in groups of random size (arrival instants are Poisson)",
            "заявки приходят группами случайного размера (моменты прихода — пуассоновские)",
        ),
        fontsize=9,
        color=INK2,
        ha="left",
    )
    _title(
        ax,
        t(
            "Batch arrivals M[X]/M/1: arrive in batches, served one at a time",
            "Пакетное поступление M[X]/M/1: приходят пачками, обслуживаются по одной",
        ),
        x=0.46,
    )
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
        t(
            'gave up waiting: each job has its own "patience budget",\nonce it runs out the job leaves the queue',
            "не дождалась: у каждой заявки свой «запас терпения»,\nпо его истечении она уходит из очереди",
        ),
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(
        ax,
        t(
            "Impatient jobs (M/M/1+M): waiting is bounded by the customer's patience",
            "Нетерпеливые заявки (M/M/1+M): ожидание ограничено терпением клиента",
        ),
        x=0.45,
    )
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
        ax.text(0.62, y, t(f"source {i + 1}", f"источник {i + 1}"), ha="center", va="center", fontsize=8, color=INK)
        draw_arrow(ax, 1.35, y, 2.6, 0.12 * (1 if y > 0 else -1), lw=1.0)
    ax.text(
        0.62,
        1.75,
        t("N sources (machines, subscribers)", "N источников (станков, абонентов)"),
        fontsize=9,
        color=INK2,
        ha="left",
    )
    draw_queue(ax, 2.9, 0, n_slots=3, occupied=1)
    draw_arrow(ax, 4.35, 0, 5.05, 0)
    draw_server(ax, 5.5, 0, r=0.34, label="μ")
    # return loop
    ax.plot([5.9, 7.0, 7.0, 0.62, 0.62], [0, 0, -1.75, -1.75, -1.5], color=INK2, lw=1.3)
    draw_arrow(ax, 0.62, -1.6, 0.62, -1.5, lw=1.3)
    ax.text(
        7.3,
        -0.9,
        t(
            'a served source goes back "to work"\nand only then can send a job again:\nthe more jobs inside, the weaker the arrival stream',
            "обслуженный источник возвращается «в работу»\nи лишь потом может снова прислать заявку:\nчем больше заявок внутри, тем слабее входящий поток",
        ),
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(
        ax,
        t(
            "Closed system (Engset): jobs are generated by a finite number of sources",
            "Закрытая система (Engset): заявки порождает конечное число источников",
        ),
        x=0.42,
    )
    return fig


def fig_loss():
    """Erlang B loss system: no queue, blocked customer is lost."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.4, 2.1))
    ax.text(0.1, 0.65, t("arrival stream, λ", "поток заявок, λ"), fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.6, 1.2]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.55, 0, 2.9, 0)
    ax.text(
        3.9, 1.75, t("no queue: only n servers", "очереди нет: только n приборов"), fontsize=9, color=INK2, ha="center"
    )
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
        t(
            "all servers busy — the job is lost (blocking).\nLoss fraction = Erlang B formula",
            "все приборы заняты — заявка теряется (блокировка).\nДоля потерянных = формула Эрланга B",
        ),
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(
        ax,
        t(
            "Loss system M/M/n/0 (Erlang B): no room to wait",
            "Система с потерями M/M/n/0 (Erlang B): мест для ожидания нет",
        ),
        x=0.45,
    )
    return fig


def fig_m_g_inf():
    """M/G/inf: unlimited servers, no queue at all."""
    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-1.9, 1.9))
    ax.text(0.1, 0.65, t("arrival stream, λ", "поток заявок, λ"), fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.6, 1.2]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.55, 0, 2.7, 0)
    for row, y in enumerate([1.1, 0.0, -1.1]):
        for col in range(3):
            x = 3.4 + col * 1.0
            if row == 2 and col == 2:
                ax.text(x, y, "…", fontsize=14, color=MUTED, ha="center", va="center")
            else:
                busy = (row + col) % 2 == 0
                draw_server(ax, x, y, r=0.3, color=AQUA if busy else GRID, label="μ" if busy else "")
    draw_arrow(ax, 5.9, 0, 6.8, 0)
    draw_customer(ax, 7.1, 0, color=GREEN, r=0.16)
    ax.text(
        7.6,
        0.0,
        t(
            'there are "infinitely many" servers: each job\ngets its own immediately, no waiting.\nOn average a = λ·b₁ servers are busy\n(and it is Poisson for any service distribution b)',
            "приборов «бесконечно много»: каждый\nполучает свой сразу, ожидания нет.\nЗанято в среднем a = λ·b₁ приборов\n(и это Пуассон при любом распределении b)",
        ),
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(
        ax,
        t(
            "M/G/∞: how much resource is busy simultaneously",
            "M/G/∞: сколько ресурса занято одновременно",
        ),
        x=0.42,
    )
    return fig


def fig_ps():
    """Processor Sharing: server splits capacity equally."""
    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-1.7, 1.7))
    ax.text(0.1, 0.65, t("arrival stream, λ", "поток заявок, λ"), fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.6, 1.2]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.55, 0, 2.6, 0)
    ax.add_patch(
        FancyBboxPatch((2.8, -1.2), 3.4, 2.4, boxstyle="round,pad=0.08", fc="white", ec=MUTED, lw=1.4, zorder=2)
    )
    ax.text(4.5, 1.02, t("server (no queue)", "прибор (без очереди)"), fontsize=9, color=INK2, ha="center")
    for i, (y, color) in enumerate(zip([0.5, -0.1, -0.7], [BLUE, AQUA, VIOLET])):
        draw_customer(ax, 3.35, y, color=color, r=0.15)
        ax.add_patch(
            FancyBboxPatch((3.7, y - 0.11), 2.2, 0.22, boxstyle="round,pad=0.02", fc=GRID, ec="none", zorder=3)
        )
        ax.add_patch(
            FancyBboxPatch((3.7, y - 0.11), 2.2 / 3, 0.22, boxstyle="round,pad=0.02", fc=color, ec="none", zorder=4)
        )
        ax.text(6.0, y, "⅓ μ", fontsize=8.5, color=INK2, ha="left", va="center")
    draw_arrow(ax, 6.4, 0, 7.3, 0)
    draw_customer(ax, 7.6, 0, color=GREEN, r=0.16)
    ax.text(
        8.05,
        0.0,
        t(
            "k jobs in the system — each gets\n1/k of the capacity: nobody waits,\nbut everyone slows down by a factor of 1/(1−ρ)",
            "k заявок в системе — каждая получает\n1/k мощности: никто не ждёт,\nно все замедляются в 1/(1−ρ) раз",
        ),
        fontsize=9,
        color=INK,
        va="center",
        ha="left",
    )
    _title(
        ax,
        t(
            "M/G/1 PS: the processor is shared equally among all jobs",
            "M/G/1 PS: процессор делится поровну между всеми",
        ),
        x=0.44,
    )
    return fig


def fig_lcfs_pr():
    """LCFS-PR: preemptive stack."""
    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.0, 1.8))
    # stack
    labels = [
        t("3rd (in service)", "3-я (обслуживается)"),
        t("2nd (preempted)", "2-я (вытеснена)"),
        t("1st (preempted)", "1-я (вытеснена)"),
    ]
    colors = [YELLOW, AQUA, BLUE]
    for i, (lab, color) in enumerate(zip(labels, colors)):
        y = 0.8 - i * 0.75
        ax.add_patch(
            FancyBboxPatch((3.0, y - 0.28), 2.6, 0.56, boxstyle="round,pad=0.04", fc=color, ec="none", zorder=3)
        )
        ax.text(4.3, y, lab, fontsize=8.5, color="white", ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(4.3, 1.45, t("stack of jobs", "стек заявок"), fontsize=9, color=INK2, ha="center")
    # new arrival lands on top
    draw_customer(ax, 1.0, 0.8, color=RED, r=0.16)
    draw_arrow(ax, 1.25, 0.8, 2.9, 0.85, color=RED, lw=1.6, ls=(0, (4, 3)))
    ax.text(
        0.1,
        0.1,
        t(
            "a new job preempts\nthe current one and takes\nthe server immediately",
            "новая заявка вытесняет\nтекущую и сразу\nзанимает прибор",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="top",
    )
    # server on top element
    draw_server(ax, 6.6, 0.8, r=0.34, label="μ")
    draw_arrow(ax, 5.68, 0.8, 6.2, 0.8)
    draw_arrow(ax, 7.0, 0.8, 7.8, 0.8)
    draw_customer(ax, 8.1, 0.8, color=GREEN, r=0.15)
    ax.text(
        6.0,
        -1.0,
        t(
            "preempted jobs resume from the interruption point.\nMean sojourn time — same as PS: b₁/(1−ρ),\nbut distributed like a busy period: heavy tails",
            "вытесненные дообслуживаются с места прерывания (resume).\nСреднее время пребывания — как у PS: b₁/(1−ρ),\nно распределено как период занятости: хвосты тяжёлые",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    _title(ax, t("M/G/1 LCFS-PR: a preemptive stack", "M/G/1 LCFS-PR: прерывающий стек"), x=0.42)
    return fig


def fig_n_policy():
    """N-policy: server sleeps until N jobs accumulate."""
    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.0, 1.8))
    # left: accumulating, server off
    draw_queue(ax, 0.6, 0.6, n_slots=4, occupied=3)
    draw_server(ax, 3.3, 0.6, r=0.32, color=GRID, label="off")
    ax.text(
        1.55,
        1.35,
        t("server sleeps, jobs pile up…", "прибор спит, заявки копятся…"),
        fontsize=9,
        color=INK2,
        ha="center",
    )
    ax.text(1.55, -0.05, t("3 of N=4", "3 из N=4"), fontsize=8.5, color=INK2, ha="center")
    # arrow to right state
    draw_arrow(ax, 4.1, 0.6, 5.2, 0.6, lw=1.8)
    ax.text(4.65, 0.95, t("the N-th\narrived", "пришла\nN-я"), fontsize=8.5, color=INK, ha="center")
    # right: serving exhaustively
    draw_queue(ax, 5.5, 0.6, n_slots=4, occupied=4)
    draw_server(ax, 8.2, 0.6, r=0.32, label="μ", busy_color=BLUE)
    draw_arrow(ax, 8.6, 0.6, 9.4, 0.6)
    draw_customer(ax, 9.7, 0.6, color=GREEN, r=0.15)
    ax.text(
        5.4,
        -1.2,
        t(
            "once on, the server works until fully drained, then sleeps again.\nAverage price of saving on startups: (N−1)/(2λ) added to waiting",
            "включившись, прибор работает до полного опустошения, затем снова спит.\nСредняя плата за экономию включений: (N−1)/(2λ) к ожиданию",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    _title(
        ax,
        t(
            "M/G/1 under N-policy: turn on once N jobs have accumulated",
            "M/G/1 под N-policy: включаемся, когда накопилось N заявок",
        ),
        x=0.44,
    )
    return fig


def fig_unreliable():
    """Unreliable server: breakdown and repair on one job's timeline."""
    fig, ax = plt.subplots(figsize=(8.2, 2.4), dpi=150)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-1.6, 1.4)
    ax.axis("off")
    segments = [
        (0.5, 2.3, AQUA, t("service", "обслуживание")),
        (2.9, 1.8, RED, t("repair", "ремонт")),
        (4.8, 2.5, AQUA, t("resumed service", "дообслуживание")),
    ]
    for x0, width, color, lab in segments:
        ax.add_patch(FancyBboxPatch((x0, -0.3), width, 0.6, boxstyle="round,pad=0.03", fc=color, ec="none"))
        ax.text(x0 + width / 2, 0.0, lab, fontsize=8.5, color="white", ha="center", va="center", fontweight="bold")
    ax.annotate(
        t(
            "breakdown (rate ξ,\nonly under load)",
            "отказ (интенсивность ξ,\nтолько под нагрузкой)",
        ),
        xy=(2.9, 0.32),
        xytext=(2.0, 1.05),
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.4},
    )
    ax.annotate("", xy=(7.3, -0.75), xytext=(0.5, -0.75), arrowprops={"arrowstyle": "<|-|>", "color": INK2, "lw": 1.3})
    ax.text(
        3.9,
        -1.15,
        t(
            "completion time C = service + all its own repairs",
            "completion time C = обслуживание + все свои ремонты",
        ),
        fontsize=9,
        color=INK,
        ha="center",
    )
    ax.text(
        7.6,
        0.0,
        t(
            'system = ordinary M/G/1,\nwhere "service" is replaced by C',
            "система = обычная M/G/1,\nгде «обслуживание» заменено на C",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    _title(
        ax,
        t(
            "Unreliable server (Avi-Itzhak–Naor): breakdowns stick to the job",
            "Ненадёжный прибор (Avi-Itzhak–Naor): отказы приклеиваются к заявке",
        ),
        x=0.47,
    )
    return fig


def fig_slowdown():
    """Conditional slowdown E[T(x)]/x by discipline, computed by the library itself."""
    # local imports: the figure is DATA-DRIVEN — it runs most_queue calculators
    from most_queue.random.distributions import GammaDistribution  # pylint: disable=import-outside-toplevel
    from most_queue.theory.fifo.mg1 import MG1Calc  # pylint: disable=import-outside-toplevel
    from most_queue.theory.srpt import MG1FbCalc, MG1SrptCalc  # pylint: disable=import-outside-toplevel

    lam, mean, cv = 1.0, 0.7, 1.5
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean, cv)
    b = GammaDistribution.calc_theory_moments(gamma_params, 5)
    rho = lam * mean

    fcfs = MG1Calc()
    fcfs.set_sources(l=lam)
    fcfs.set_servers(b)
    w_fcfs = fcfs.get_w(3)[0]

    fb = MG1FbCalc()
    fb.set_sources(lam)
    fb.set_servers(gamma_params, "Gamma")
    fb.run()

    srpt = MG1SrptCalc()
    srpt.set_sources(lam)
    srpt.set_servers(gamma_params, "Gamma")
    srpt.run()

    import numpy as np  # pylint: disable=import-outside-toplevel

    xs = np.linspace(0.05, 4.0 * mean, 120)
    curves = {
        "FCFS": [(w_fcfs + x) / x for x in xs],
        "PS": [1.0 / (1.0 - rho) for _ in xs],
        "FB (LAS)": [fb.conditional_mean_response(float(x)) / x for x in xs],
        "SRPT": [srpt.conditional_mean_response(float(x)) / x for x in xs],
    }
    colors = {"FCFS": BLUE, "PS": AQUA, "FB (LAS)": YELLOW, "SRPT": VIOLET}

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    for name, ys in curves.items():
        ax.plot(xs, ys, color=colors[name], lw=2, label=name)
    ax.set_ylim(0, 12)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(
        0.35,
        11.2,
        t(
            "FCFS: short jobs suffer the most\n(curve goes off the chart, clipped)",
            "FCFS: короткие заявки страдают сильнее всех\n(кривая уходит вверх, обрезана)",
        ),
        fontsize=8.5,
        color=INK2,
        ha="left",
        va="top",
    )
    ax.set_xlim(0, 4.2 * mean)
    ax.set_xlabel(t("job size x", "размер заявки x"), fontsize=9, color=INK2)
    ax.set_ylabel(t("slowdown E[T(x)] / x", "замедление E[T(x)] / x"), fontsize=9, color=INK2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        t(
            "Who pays for the discipline: slowdown as a function of job size\n"
            f"(M/G/1, Gamma service, ρ={rho:.1f}, CV={cv}; computed by most_queue calculators)",
            "Кто платит за дисциплину: замедление заявки в зависимости от её размера\n"
            f"(M/G/1, Gamma-обслуживание, ρ={rho:.1f}, CV={cv}; посчитано калькуляторами most_queue)",
        ),
        fontsize=10,
        color=INK,
        pad=12,
    )
    return fig


def fig_retrial():
    """Retrial queue: blocked jobs join an orbit and retry."""
    fig, ax = plt.subplots(figsize=(8.2, 3.2), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.5, 1.9))
    ax.text(0.1, 0.65, t("arrival stream, λ", "поток заявок, λ"), fontsize=9, color=INK2, ha="left")
    for i, xx in enumerate([0.0, 0.6, 1.2]):
        draw_customer(ax, xx, 0, alpha=0.45 + 0.27 * i)
    draw_arrow(ax, 1.55, 0, 2.9, 0)
    draw_server(ax, 3.5, 0, r=0.36, label="μ", busy_color=BLUE)
    draw_arrow(ax, 3.95, 0, 4.9, 0)
    draw_customer(ax, 5.2, 0, color=GREEN, r=0.16)
    ax.text(3.5, 0.85, t("no queue", "очереди нет"), fontsize=8.5, color=INK2, ha="center")
    # orbit ellipse
    orbit = Ellipse((3.5, -1.55), 4.6, 1.3, fc="white", ec=MUTED, lw=1.3, ls=(0, (5, 3)))
    ax.add_patch(orbit)
    for xx in (2.3, 3.5, 4.7):
        draw_customer(ax, xx, -1.55, color=YELLOW, r=0.16)
    ax.text(3.5, -2.35, t("orbit", "орбита"), fontsize=9, color=INK2, ha="center")
    # blocked -> orbit
    draw_arrow(ax, 2.75, -0.25, 2.2, -1.15, color=RED, lw=1.5, ls=(0, (4, 3)))
    ax.text(0.35, -1.2, t("server busy —\nto orbit", "прибор занят —\nв орбиту"), fontsize=8.5, color=INK, ha="left")
    # retry -> server
    draw_arrow(ax, 4.75, -1.2, 3.75, -0.42, color=YELLOW, lw=1.5, ls=(0, (4, 3)))
    ax.text(
        5.6,
        -1.3,
        t(
            "each job in the orbit retries\nafter a random time (rate γ);\nif busy — back to orbit",
            "каждая заявка в орбите повторяет попытку\nчерез случайное время (интенсивность γ);\nзанято — снова в орбиту",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    _title(
        ax,
        t(
            "Retrial queue: instead of waiting — repeated retries",
            "Retrial-очередь: вместо ожидания — повторные попытки",
        ),
        x=0.44,
    )
    return fig


def fig_map_arrivals():
    """MAP: bursty correlated arrivals vs a renewal stream."""
    fig, ax = plt.subplots(figsize=(8.2, 2.8), dpi=150)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-2.1, 1.9)
    ax.axis("off")
    # renewal (top): evenly-ish spaced
    ax.text(
        0.1,
        1.45,
        t("renewal stream (interarrival times are independent):", "renewal-поток (интервалы независимы):"),
        fontsize=9,
        color=INK2,
        ha="left",
    )
    for xx in (0.7, 1.8, 2.7, 3.9, 4.9, 6.1, 7.0, 8.2, 9.3, 10.3):
        draw_customer(ax, xx, 0.8, r=0.12)
    ax.plot([0.4, 10.8], [0.8, 0.8], color=GRID, lw=1.0, zorder=1)
    # MAP (bottom): bursts
    ax.text(
        0.1,
        0.05,
        t(
            "MAP (e.g. MMPP): bursts and lulls at the same mean rate:",
            "MAP (например, MMPP): всплески и паузы при той же средней интенсивности:",
        ),
        fontsize=9,
        color=INK2,
        ha="left",
    )
    for xx in (0.7, 1.0, 1.3, 1.6, 2.0, 4.9, 5.2, 5.5, 5.9, 6.2, 9.4, 9.7, 10.0):
        draw_customer(ax, xx, -0.6, r=0.12, color=RED)
    ax.plot([0.4, 10.8], [-0.6, -0.6], color=GRID, lw=1.0, zorder=1)
    ax.text(
        0.1,
        -1.55,
        t(
            "interarrival times are correlated: a short one is more often followed by a short one. Means and CV may match\nthe renewal stream, but the queue under a MAP input is many times longer (see tutorials/map_ph_correlation.ipynb)",
            "интервалы коррелированы: за коротким чаще следует короткий. Средние и CV могут совпадать\nс renewal-потоком, но очередь при MAP-входе в разы длиннее (см. tutorials/map_ph_correlation.ipynb)",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    _title(
        ax,
        t(
            "Correlated input (MAP): what renewal models miss",
            "Коррелированный вход (MAP): то, чего не видят renewal-модели",
        ),
        x=0.45,
    )
    return fig


def _ph_phase(ax, x, y, label, color=AQUA, r=0.30):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold", zorder=4)


def _ph_exit(ax, x, y):
    ax.add_patch(
        FancyBboxPatch((x - 0.14, y - 0.14), 0.28, 0.28, boxstyle="round,pad=0.03", fc=GREEN, ec="none", zorder=3)
    )


def fig_ph_gallery():
    """Familiar distributions drawn as phase-type (PH) chains."""
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.4), dpi=150)
    for ax in axes.flat:
        ax.set_xlim(-0.3, 5.4)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # --- Exp(mu): a single phase
    ax = axes[0, 0]
    ax.set_title(t("Exponential = 1 phase", "Экспоненциальное = 1 фаза"), fontsize=9.5, color=INK)
    draw_arrow(ax, 0.2, 0, 1.0, 0)
    _ph_phase(ax, 1.4, 0, "μ")
    draw_arrow(ax, 1.75, 0, 2.6, 0)
    _ph_exit(ax, 2.9, 0)
    ax.text(0.2, 0.45, "α=[1]", fontsize=8, color=INK2)

    # --- Erlang-3: chain
    ax = axes[0, 1]
    ax.set_title(t("Erlang E₃ = a chain of 3 phases", "Эрланга E₃ = цепочка из 3 фаз"), fontsize=9.5, color=INK)
    draw_arrow(ax, 0.0, 0, 0.7, 0)
    for i in range(3):
        _ph_phase(ax, 1.1 + i * 1.3, 0, "μ")
        if i < 2:
            draw_arrow(ax, 1.45 + i * 1.3, 0, 1.9 + i * 1.3, 0)
    draw_arrow(ax, 4.05, 0, 4.7, 0)
    _ph_exit(ax, 5.0, 0)

    # --- H2: two parallel phases
    ax = axes[1, 0]
    ax.set_title(
        t("Hyperexponential H₂ = 2 parallel phases", "Гиперэкспоненциальное H₂ = 2 параллельные фазы"),
        fontsize=9.5,
        color=INK,
    )
    draw_arrow(ax, 0.1, 0, 0.9, 0.62, lw=1.3)
    draw_arrow(ax, 0.1, 0, 0.9, -0.62, lw=1.3)
    ax.text(0.35, 0.55, "p₁", fontsize=8, color=INK2)
    ax.text(0.35, -0.7, "1−p₁", fontsize=8, color=INK2)
    _ph_phase(ax, 1.3, 0.7, "μ₁")
    _ph_phase(ax, 1.3, -0.7, "μ₂")
    draw_arrow(ax, 1.65, 0.7, 2.6, 0.1, lw=1.3)
    draw_arrow(ax, 1.65, -0.7, 2.6, -0.1, lw=1.3)
    _ph_exit(ax, 2.9, 0)

    # --- Cox-2: chain with early exit
    ax = axes[1, 1]
    ax.set_title(
        t("Cox C₂ = a chain with early exit", "Кокса C₂ = цепочка с досрочным выходом"), fontsize=9.5, color=INK
    )
    draw_arrow(ax, 0.0, 0, 0.7, 0)
    _ph_phase(ax, 1.1, 0, "μ₁")
    draw_arrow(ax, 1.45, 0, 2.3, 0)
    ax.text(1.85, 0.22, "p₁", fontsize=8, color=INK2)
    _ph_phase(ax, 2.7, 0, "μ₂")
    draw_arrow(ax, 3.05, 0, 3.9, 0)
    _ph_exit(ax, 4.2, 0)
    draw_arrow(ax, 1.3, -0.35, 3.95, -1.05, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.15, -1.25, t('1−p₁ ("straight to exit")', "1−p₁ (сразу на выход)"), fontsize=8, color=INK2, ha="left")
    _ph_exit(ax, 4.2, -1.1)

    fig.suptitle(
        t(
            "PH distribution = time spent wandering through a chain of exponential phases until exit:\n"
            "start via vector α, transitions via matrix T. Everything familiar is a special case of PH(α, T)",
            "PH-распределение = время блуждания по цепочке экспоненциальных фаз до выхода:\n"
            "старт по вектору α, переходы по матрице T. Всё знакомое — частные случаи PH(α, T)",
        ),
        fontsize=10,
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def fig_mmpp():
    """MMPP mechanics: Poisson process modulated by a two-state chain."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.2, 1.9))
    # fast phase
    ax.add_patch(Circle((2.2, 0.2), 0.85, fc=BLUE, ec="none", zorder=3))
    ax.text(2.2, 0.36, t("phase 1", "фаза 1"), ha="center", fontsize=9.5, color="white", fontweight="bold", zorder=4)
    ax.text(2.2, -0.05, "λ₁ = 2.0", ha="center", fontsize=8.5, color="white", zorder=4)
    # slow phase
    ax.add_patch(Circle((7.4, 0.2), 0.85, fc=VIOLET, ec="none", zorder=3))
    ax.text(7.4, 0.36, t("phase 2", "фаза 2"), ha="center", fontsize=9.5, color="white", fontweight="bold", zorder=4)
    ax.text(7.4, -0.05, "λ₂ = 0.4", ha="center", fontsize=8.5, color="white", zorder=4)
    # switching arrows
    draw_arrow(ax, 3.15, 0.55, 6.45, 0.55, lw=1.5)
    draw_arrow(ax, 6.45, -0.15, 3.15, -0.15, lw=1.5)
    ax.text(4.8, 0.78, t("switching q₁₂", "переключение q₁₂"), fontsize=8.5, color=INK2, ha="center")
    ax.text(4.8, -0.45, "q₂₁", fontsize=8.5, color=INK2, ha="center")
    # emitted arrivals
    for xx in (1.5, 1.9, 2.3, 2.7):
        draw_customer(ax, xx, -1.45, r=0.11, color=BLUE)
    draw_arrow(ax, 2.2, -0.75, 2.2, -1.15, lw=1.2)
    ax.text(2.2, -1.85, t("jobs pour in often", "заявки сыплются часто"), fontsize=8.5, color=INK2, ha="center")
    for xx in (7.2, 7.8):
        draw_customer(ax, xx, -1.45, r=0.11, color=VIOLET)
    draw_arrow(ax, 7.4, -0.75, 7.4, -1.15, lw=1.2)
    ax.text(7.4, -1.85, t("jobs are rare", "заявки редки"), fontsize=8.5, color=INK2, ha="center")
    ax.text(
        9.0,
        0.2,
        t(
            "D₀ — phase transitions\nwithout a job,\nD₁ — with a job.\nMMPP: D₁ = diag(λᵢ)",
            "D₀ — переходы фаз\nбез заявки,\nD₁ — с заявкой.\nMMPP: D₁ = diag(λᵢ)",
        ),
        fontsize=9,
        color=INK,
        ha="left",
        va="center",
    )
    _title(
        ax,
        t(
            "MMPP — the simplest meaningful MAP: a Poisson process switched by a Markov chain",
            "MMPP — простейший содержательный MAP: Пуассон, переключаемый марковской цепью",
        ),
        x=0.46,
    )
    return fig


def _draw_station(ax, x, y, mu_label="μ", n_slots=3, occupied=2, sub=None, occ_color=BLUE):
    """Mini station: short queue + one server, centred at the server."""
    draw_queue(ax, x - n_slots * 0.42 - 0.25, y, n_slots=n_slots, occupied=occupied, occ_color=occ_color)
    draw_server(ax, x + 0.35, y, label=mu_label, sub=sub)


def fig_open_network():
    """Open network: routed stations, external arrivals and departures."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.0, 2.1))
    ax.text(0.0, 0.62, t("external flow, λ", "внешний поток, λ"), fontsize=9, color=INK2, ha="left")
    draw_customer(ax, 0.2, 0, alpha=0.7)
    draw_arrow(ax, 0.5, 0, 1.35, 0)
    _draw_station(ax, 2.8, 0, sub=t("node 1", "узел 1"))
    draw_arrow(ax, 3.35, 0.25, 4.55, 1.05)
    ax.text(3.9, 0.95, "p₁₂", fontsize=9, color=INK2)
    draw_arrow(ax, 3.35, -0.25, 4.55, -1.05)
    ax.text(3.9, -1.15, "p₁₃", fontsize=9, color=INK2)
    _draw_station(ax, 6.1, 1.05, sub=t("node 2", "узел 2"))
    _draw_station(ax, 6.1, -1.05, sub=t("node 3", "узел 3"))
    draw_arrow(ax, 6.7, 1.05, 8.0, 0.15)
    draw_arrow(ax, 6.7, -1.05, 8.0, -0.15)
    for i, xx in enumerate([8.4, 8.95, 9.5]):
        draw_customer(ax, xx, 0, color=GREEN, alpha=1.0 - 0.27 * i)
    ax.text(9.0, 0.6, t("served", "обслуженные"), fontsize=9, color=INK2, ha="center")
    _title(
        ax,
        t(
            "Open network: jobs are routed between stations by a transition matrix",
            "Открытая сеть: заявки движутся между узлами по матрице переходов",
        ),
    )
    return fig


def fig_closed_network():
    """Closed network: N jobs circulate between a delay node and a server."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=150)
    _clean_axes(ax, (-0.4, 10.6), (-2.3, 2.3))
    # delay node (terminals, think time)
    ax.add_patch(Ellipse((2.6, 1.0), 3.6, 1.7, fc="white", ec=MUTED, lw=1.2, zorder=1))
    for i, xx in enumerate([1.6, 2.4, 3.2]):
        draw_customer(ax, xx, 1.0, alpha=0.6 + 0.2 * i)
    ax.text(
        2.6,
        -0.05,
        t("N terminals, think time Z (delay node)", "N терминалов, время Z (delay-узел)"),
        fontsize=9,
        color=INK2,
        ha="center",
        va="top",
    )
    # server station
    _draw_station(ax, 6.7, -1.2, sub=t("server, b", "сервер, b"))
    # circulation arrows
    draw_arrow(ax, 4.3, 0.8, 5.6, -0.9)
    draw_arrow(ax, 7.4, -0.9, 4.5, 1.35)
    ax.text(
        7.0,
        0.6,
        t("fixed population N circulates", "фиксированная популяция N циркулирует"),
        fontsize=9,
        color=INK2,
        ha="left",
    )
    _title(
        ax,
        t(
            "Closed network: no external arrivals — N jobs loop forever (MVA / Buzen)",
            "Закрытая сеть: внешнего потока нет — N заявок циркулируют вечно (MVA / Бьюзен)",
        ),
    )
    return fig


def fig_g_network():
    """G-network: positive flow plus negative signals that remove customers."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.1, 1.9))
    ax.text(0.0, 0.62, t("positive flow, λ⁺", "позитивный поток, λ⁺"), fontsize=9, color=INK2, ha="left")
    draw_customer(ax, 0.2, 0, alpha=0.7)
    draw_arrow(ax, 0.5, 0, 1.35, 0)
    _draw_station(ax, 2.8, 0, sub=t("node 1", "узел 1"))
    draw_arrow(ax, 3.4, 0, 4.6, 0)
    ax.text(4.0, 0.2, "p⁺", fontsize=9, color=INK2)
    _draw_station(ax, 6.1, 0, sub=t("node 2", "узел 2"))
    draw_arrow(ax, 6.7, 0, 7.8, 0)
    draw_customer(ax, 8.15, 0, color=GREEN)
    # negative signal: routed from node 1 and external
    draw_arrow(ax, 3.3, -0.45, 4.75, -1.3, color=RED, ls=(0, (4, 3)))
    ax.text(3.7, -1.15, t("signal p⁻", "сигнал p⁻"), fontsize=9, color=RED)
    draw_arrow(ax, 4.9, -1.35, 5.7, -0.5, color=RED, ls=(0, (4, 3)))
    draw_arrow(ax, 7.3, -1.7, 6.3, -0.55, color=RED, ls=(0, (4, 3)))
    ax.text(7.45, -1.9, t("external negatives, λ⁻", "внешние негативные, λ⁻"), fontsize=9, color=RED, ha="left")
    ax.text(
        4.2,
        -1.85,
        t("a signal removes one customer", "сигнал удаляет одну заявку"),
        fontsize=9,
        color=INK2,
        ha="center",
    )
    _title(
        ax,
        t(
            "G-network (Gelenbe): negative signals kill customers — exact product form",
            "G-сеть (Геленбе): негативные сигналы удаляют заявки — точный product form",
        ),
    )
    return fig


def fig_bcmp():
    """BCMP: multi-class network with different station types."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.0, 1.9))
    ax.text(0.0, 0.92, t("class 1, λ₁", "класс 1, λ₁"), fontsize=9, color=BLUE, ha="left")
    draw_customer(ax, 0.2, 0.35, color=BLUE, alpha=0.8)
    ax.text(0.0, -1.0, t("class 2, λ₂", "класс 2, λ₂"), fontsize=9, color=YELLOW, ha="left")
    draw_customer(ax, 0.2, -0.5, color=YELLOW, alpha=0.9)
    draw_arrow(ax, 0.6, 0.35, 1.55, 0.05)
    draw_arrow(ax, 0.6, -0.5, 1.55, -0.15)
    # PS station with mixed classes
    draw_queue(ax, 1.75, 0, n_slots=3, occupied=2, occ_color=BLUE)
    draw_customer(ax, 2.6, 0, color=YELLOW, r=0.14)
    draw_server(ax, 3.35, 0, label="PS", sub=t("sharing", "разделение"))
    draw_arrow(ax, 3.85, 0, 4.8, 0)
    # FCFS station
    draw_queue(ax, 5.0, 0, n_slots=3, occupied=2, occ_color=YELLOW)
    draw_server(ax, 6.6, 0, label="μ", sub="FCFS")
    draw_arrow(ax, 7.1, 0, 8.05, 0)
    # IS station
    ax.add_patch(Ellipse((8.9, 0), 1.5, 1.1, fc="white", ec=MUTED, lw=1.2, zorder=1))
    draw_customer(ax, 8.6, 0, color=BLUE, r=0.14)
    draw_customer(ax, 9.2, 0, color=YELLOW, r=0.14)
    ax.text(8.9, -0.75, "IS", fontsize=9, color=INK2, ha="center")
    draw_arrow(ax, 9.7, 0, 10.5, 0)
    ax.text(
        5.6,
        -1.6,
        t(
            "per-class routing and service; product form by the BCMP theorem",
            "маршрутизация и обслуживание по классам; product form по теореме BCMP",
        ),
        fontsize=9,
        color=INK2,
        ha="center",
    )
    _title(
        ax,
        t(
            "BCMP network: several job classes over FCFS / PS / LCFS-PR / IS stations",
            "BCMP-сеть: несколько классов заявок и станции FCFS / PS / LCFS-PR / IS",
        ),
    )
    return fig


def fig_tandem_blocking():
    """Tandem with finite buffers: the middle node is full, upstream blocked."""
    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=150)
    _clean_axes(ax, (-0.4, 11.2), (-2.0, 1.7))
    ax.text(0.0, 0.62, "λ", fontsize=10, color=INK2, ha="left")
    draw_arrow(ax, 0.3, 0, 1.15, 0)
    # node 1: server holds a finished job (blocked)
    draw_queue(ax, 1.35, 0, n_slots=2, occupied=2)
    draw_server(ax, 2.6, 0, label="μ₁", busy_color=RED)
    ax.text(2.6, 0.95, t("blocked (BAS)", "блокирован (BAS)"), fontsize=9, color=RED, ha="center")
    draw_arrow(ax, 3.1, 0, 4.05, 0)
    # node 2: buffer full
    draw_queue(ax, 4.25, 0, n_slots=3, occupied=3)
    draw_server(ax, 5.9, 0, label="μ₂")
    ax.text(4.9, -0.75, t("buffer full, K₂", "буфер полон, K₂"), fontsize=9, color=INK2, ha="center")
    draw_arrow(ax, 6.4, 0, 7.35, 0)
    # node 3
    draw_queue(ax, 7.55, 0, n_slots=3, occupied=1)
    draw_server(ax, 9.2, 0, label="μ₃")
    draw_arrow(ax, 9.7, 0, 10.5, 0)
    ax.text(
        5.4,
        -1.55,
        t(
            "a finished job waits on the server until space frees downstream",
            "готовая заявка ждёт на приборе, пока не освободится место дальше",
        ),
        fontsize=9,
        color=INK2,
        ha="center",
    )
    _title(
        ax,
        t(
            "Tandem with finite buffers: blocking after service caps the throughput",
            "Тандем с конечными буферами: блокировка после обслуживания режет пропускную способность",
        ),
    )
    return fig


def fig_polling():
    """Polling: one server cyclically visits Q queues with switchover."""
    fig, ax = plt.subplots(figsize=(7.8, 3.4), dpi=150)
    _clean_axes(ax, (-0.4, 10.4), (-2.2, 2.2))
    ys = [1.5, 0.0, -1.5]
    for i, y in enumerate(ys):
        ax.text(-0.3, y + 0.42, f"λ{chr(0x2081 + i)}", fontsize=9, color=INK2)
        draw_arrow(ax, 0.0, y, 0.9, y, lw=1.2)
        draw_queue(ax, 1.1, y, n_slots=3, occupied=2 - (i == 1))
    draw_server(ax, 4.6, 0.0, r=0.42, label="μ")
    ax.text(4.6, -0.75, t("one server", "один сервер"), fontsize=9, color=INK2, ha="center")
    # cyclic visits with switchover
    draw_arrow(ax, 4.35, 0.42, 2.75, 1.4, color=VIOLET, ls=(0, (4, 3)))
    draw_arrow(ax, 2.75, -1.4, 4.35, -0.42, color=VIOLET, ls=(0, (4, 3)))
    draw_arrow(ax, 2.6, 1.25, 2.6, -1.25, color=VIOLET, ls=(0, (4, 3)))
    ax.text(
        3.1,
        1.85,
        t("cyclic visits, switchover r", "циклический обход, переключение r"),
        fontsize=9,
        color=VIOLET,
        ha="left",
    )
    draw_arrow(ax, 5.1, 0, 6.0, 0)
    for i, xx in enumerate([6.35, 6.9, 7.45]):
        draw_customer(ax, xx, 0, color=GREEN, alpha=1.0 - 0.27 * i)
    ax.text(
        9.0,
        0.0,
        t("exhaustive / gated\nper-queue discipline", "exhaustive / gated\nдисциплина по очереди"),
        fontsize=9,
        color=INK2,
        ha="center",
        va="center",
    )
    _title(
        ax,
        t(
            "Polling: one server tours Q queues; switchover makes it non-work-conserving",
            "Polling: сервер обходит Q очередей; переключение делает систему не work-conserving",
        ),
    )
    return fig


def fig_msj():
    """Multiserver job: one job occupies several servers at once."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0), dpi=150)
    _clean_axes(ax, (-0.4, 10.8), (-1.9, 1.8))
    ax.text(1.3, 1.15, t("jobs need k servers each", "заявкам нужно k серверов"), fontsize=9, color=INK2, ha="center")
    for xx, k, col in [(0.4, 1, BLUE), (1.3, 2, YELLOW), (2.2, 3, VIOLET)]:
        draw_customer(ax, xx, 0.55, color=col, r=0.2, label=str(k))
    draw_arrow(ax, 2.7, 0.55, 3.6, 0.2)
    for i in range(6):
        x = 4.3 + i * 1.05
        busy = i in (0, 1, 2)
        draw_server(ax, x, -0.35, color=YELLOW if busy else AQUA, label="μ")
    ax.add_patch(
        FancyBboxPatch((3.85, -0.95), 3.05, 1.2, boxstyle="round,pad=0.04", fc="none", ec=VIOLET, lw=1.8, zorder=4)
    )
    ax.text(
        5.35,
        -1.35,
        t("one job holds k = 3 servers simultaneously", "одна заявка держит k = 3 сервера одновременно"),
        fontsize=9,
        color=VIOLET,
        ha="center",
    )
    _title(
        ax,
        t(
            "Multiserver jobs: capacity is lost to packing — you can't just use c·μ",
            "Multiserver-job: ёмкость теряется на упаковке — просто c·μ использовать нельзя",
        ),
    )
    return fig


def fig_load_balancing():
    """Power-of-d dispatching over a pool of queues."""
    fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=150)
    _clean_axes(ax, (-0.4, 10.8), (-2.3, 2.3))
    draw_customer(ax, 0.3, 0, alpha=0.8)
    draw_arrow(ax, 0.6, 0, 1.4, 0)
    draw_server(ax, 1.9, 0, r=0.38, color=VIOLET, label="LB")
    ax.text(1.9, -0.75, t("dispatcher", "диспетчер"), fontsize=9, color=INK2, ha="center")
    ys = [1.6, 0.55, -0.55, -1.6]
    occupied = [3, 1, 2, 3]
    for y, occ in zip(ys, occupied):
        draw_queue(ax, 5.0, y, n_slots=4, occupied=occ)
        draw_server(ax, 7.15, y, r=0.28)
        draw_arrow(ax, 7.42, y, 8.1, y, lw=1.0)
    draw_arrow(ax, 2.3, 0.25, 4.8, 1.5, color=MUTED, ls=(0, (4, 3)))
    draw_arrow(ax, 2.3, 0.1, 4.8, 0.55, color=GREEN, lw=2.0)
    ax.text(3.15, 1.4, t("sampled, longer", "опрошена, длиннее"), fontsize=8.5, color=MUTED)
    ax.text(3.15, 0.06, t("sampled, shorter — join", "опрошена, короче — сюда"), fontsize=8.5, color=GREEN)
    ax.text(
        9.5,
        -1.55,
        t("N queues,\nonly d = 2 sampled", "N очередей,\nопрошены лишь d = 2"),
        fontsize=9,
        color=INK2,
        ha="center",
        va="center",
    )
    _title(
        ax,
        t(
            "Power of two choices: sample d random queues, join the shortest",
            "Power of two choices: опросить d случайных очередей и встать в кратчайшую",
        ),
    )
    return fig


def fig_time_varying():
    """Mt/M/c: congestion lags the time-varying load."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=150)
    x = np.linspace(0, 4 * 3.14159, 300)
    lam = 1.0 + 0.55 * np.sin(x)
    resp = 1.0 + 0.55 * np.sin(x - 0.7)
    ax.plot(x, lam, color=BLUE, lw=2, label=t("load λ(t)", "нагрузка λ(t)"))
    ax.plot(x, resp, color=RED, lw=2, ls="--", label=t("congestion (lags)", "перегрузка (отстаёт)"))
    peak = 3.14159 / 2
    ax.axvline(peak, color=MUTED, lw=1, ls=":")
    ax.axvline(peak + 0.7, color=MUTED, lw=1, ls=":")
    ax.annotate(
        t("lag", "запаздывание"),
        xy=(peak + 0.7, 1.62),
        xytext=(peak + 1.7, 1.72),
        fontsize=9,
        color=INK2,
        arrowprops={"arrowstyle": "->", "color": INK2},
    )
    ax.set_ylim(0.3, 1.95)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(t("time of day", "время суток"), fontsize=9, color=INK2)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _title(
        ax,
        t(
            "Time-varying Mₜ/M/c: peak congestion is NOT at peak demand (PSA vs MOL)",
            "Нестационарная Mₜ/M/c: пик перегрузки НЕ в пике спроса (PSA vs MOL)",
        ),
    )
    return fig


def fig_aoi():
    """Age of Information: sawtooth age process."""
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=150)
    deliveries = [1.6, 2.5, 4.4, 5.6, 7.2]
    ages_after = [0.7, 0.4, 0.9, 0.6, 0.5]
    xs, ys = [0.0], [0.8]
    tt, age = 0.0, 0.8
    for d, a in zip(deliveries, ages_after):
        xs.append(d)
        ys.append(age + (d - tt))
        xs.append(d)
        ys.append(a)
        tt, age = d, a
    xs.append(8.4)
    ys.append(age + 8.4 - tt)
    ax.plot(xs, ys, color=BLUE, lw=2)
    for d in deliveries:
        ax.axvline(d, color=GRID, lw=1)
        ax.plot([d], [0.06], marker="^", color=GREEN, markersize=7, clip_on=False)
    ax.axhline(1.35, color=RED, lw=1.4, ls="--")
    ax.text(8.3, 1.45, t("mean AoI", "средний AoI"), fontsize=9, color=RED, ha="right")
    ax.text(4.35, -0.34, t("updates delivered", "доставленные обновления"), fontsize=9, color=GREEN, ha="center")
    ax.set_ylim(0, 3.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(t("age of data", "возраст данных"), fontsize=9, color=INK2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _title(
        ax,
        t(
            "Age of Information: age grows between updates and resets on delivery",
            "Age of Information: возраст растёт между обновлениями и сбрасывается при доставке",
        ),
    )
    return fig


def fig_machine_repair():
    """Machine repair problem: finite park, spares, repair crew."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=150)
    _clean_axes(ax, (-0.4, 11.0), (-2.3, 2.2))
    # operating machines
    for xx in [0.6, 1.7, 2.8]:
        draw_server(ax, xx, 1.2, color=AQUA, label="M")
    ax.text(1.7, 0.45, t("M operating machines", "M работающих машин"), fontsize=9, color=INK2, ha="center")
    # warm spares
    for xx in [0.9, 2.0]:
        draw_server(ax, xx, -1.3, color=YELLOW, label="S")
    ax.text(1.45, -2.05, t("S warm spares", "S тёплых резервных"), fontsize=9, color=INK2, ha="center")
    # failure arrow to repair queue
    draw_arrow(ax, 3.35, 1.2, 4.7, 0.15, color=RED, ls=(0, (4, 3)))
    ax.text(3.55, 0.9, t("failures, ξ", "отказы, ξ"), fontsize=9, color=RED)
    # repair queue + crew
    draw_queue(ax, 4.9, 0, n_slots=3, occupied=2, occ_color=RED)
    for sy in (0.55, -0.55):
        draw_server(ax, 7.1, sy, r=0.3, color=VIOLET, label="η")
    ax.text(7.1, -1.25, t("R repairmen", "R ремонтников"), fontsize=9, color=INK2, ha="center")
    # repaired units return as spares
    draw_arrow(ax, 7.55, -0.6, 2.6, -1.5, color=GREEN, ls=(0, (4, 3)))
    ax.text(
        5.1,
        -1.75,
        t("repaired -> spare / operation", "починенные -> резерв / работа"),
        fontsize=9,
        color=GREEN,
        ha="center",
    )
    draw_arrow(ax, 2.0, -0.85, 2.7, 0.75, color=INK2, ls=(0, (2, 2)), lw=1.1)
    ax.text(
        9.4,
        0.7,
        t(
            "closed loop:\nfewer machines run —\nfewer failures happen",
            "замкнутый цикл:\nменьше машин работает —\nреже отказы",
        ),
        fontsize=8.5,
        color=INK2,
        ha="center",
        va="center",
    )
    _title(
        ax,
        t(
            "Machine repair problem: a finite park self-regulates through failures and repairs",
            "Machine repair problem: конечный парк саморегулируется отказами и ремонтами",
        ),
    )
    return fig


FIGURES = {
    "fifo_mmn": fig_fifo_mmn,
    "machine_repair": fig_machine_repair,
    "polling": fig_polling,
    "msj": fig_msj,
    "load_balancing": fig_load_balancing,
    "time_varying": fig_time_varying,
    "aoi": fig_aoi,
    "open_network": fig_open_network,
    "closed_network": fig_closed_network,
    "g_network": fig_g_network,
    "bcmp": fig_bcmp,
    "tandem_blocking": fig_tandem_blocking,
    "retrial": fig_retrial,
    "map_arrivals": fig_map_arrivals,
    "ph_gallery": fig_ph_gallery,
    "mmpp": fig_mmpp,
    "loss": fig_loss,
    "m_g_inf": fig_m_g_inf,
    "ps": fig_ps,
    "lcfs_pr": fig_lcfs_pr,
    "n_policy": fig_n_policy,
    "unreliable": fig_unreliable,
    "slowdown": fig_slowdown,
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
    """Regenerate every figure (English + Russian) into the directory of this script."""
    global _LANG
    for name, builder in FIGURES.items():
        for lang, suffix in (("en", ""), ("ru", ".ru")):
            _LANG = lang
            fig = builder()
            fig.savefig(OUT_DIR / f"{name}{suffix}.png", bbox_inches="tight")
            plt.close(fig)
            print(f"saved {name}{suffix}.png")


if __name__ == "__main__":
    main()
