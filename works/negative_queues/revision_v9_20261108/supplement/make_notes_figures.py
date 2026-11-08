"""Figures for ph_gamma_notes.md (reads notes-data.json, writes figures/notes_*.png).
Run from the repository root with PYTHONPATH=. after notes_data.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.linalg import expm
from scipy.special import beta as beta_fn, roots_jacobi
from scipy.stats import gamma as gamma_law

from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
B1, CV = 2.1, 0.8
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.linewidth": 0.8,
    "mathtext.fontset": "dejavuserif", "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
INK, BLUE, RED, GREY = "#000000", "#1f4e79", "#b03a2e", "#7f7f7f"
comma = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, linewidth=0.3, color="#bfbfbf")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(comma)


def ph_density(service, t):
    a, T, t0 = service.alpha, service.generator, service.exit
    return np.array([a @ expm(T * x) @ t0 for x in t])


def ph_cdf(service, t):
    a, T = service.alpha, service.generator
    return np.array([1 - a @ expm(T * x) @ np.ones(len(a)) for x in t])


def fig_density():
    shape = 1 / CV**2
    t = np.linspace(0.0, 8.0, 401)
    exact_pdf = gamma_law.pdf(t, shape, scale=B1 / shape)
    exact_cdf = gamma_law.cdf(t, shape, scale=B1 / shape)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    ax = axes[0]
    ax.plot(t, exact_pdf, color=INK, linewidth=1.6, label="гамма, CV = 0,8")
    for nodes, st in [(2, dict(color=RED, linestyle=":", marker="^")), (4, dict(color=BLUE, linestyle="--", marker="s")),
                      (8, dict(color=GREY, linestyle="-.", marker="o"))]:
        s = gamma_ph(B1, CV, nodes)
        ax.plot(t, ph_density(s, t), linewidth=1.1, markevery=40, markersize=4, markerfacecolor="white",
                label=f"PH, {s.size} фаз", **st)
    ax.set_xlabel("$t$")
    ax.set_ylabel("Плотность")
    ax.legend(frameon=False)
    tidy(ax)
    ax = axes[1]
    for nodes, st in [(2, dict(color=RED, linestyle=":")), (4, dict(color=BLUE, linestyle="--")),
                      (8, dict(color=GREY, linestyle="-."))]:
        s = gamma_ph(B1, CV, nodes)
        dev = np.abs(ph_cdf(s, t) - exact_cdf)
        ax.semilogy(t[1:], np.maximum(dev[1:], 1e-9), linewidth=1.2, label=f"{s.size} фаз", **st)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$|F_{\\mathrm{PH}}(t)-F(t)|$")
    ax.legend(frameon=False)
    tidy(ax)
    fig.savefig(FIG / "notes_density.png")
    plt.close(fig)


def fig_quadrature():
    shape = 1 / CV**2
    kappa = int(np.floor(shape))
    r = shape - kappa
    theta = B1 / shape
    x = np.linspace(0.002, 0.998, 500)
    dens = x ** (r - 1) * (1 - x) ** (-r) / beta_fn(r, 1 - r)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    ax.plot(x, dens, color=INK, linewidth=1.4, label="плотность $\\mathrm{Beta}(r,\\,1-r)$")
    for nodes, color, mk in [(4, BLUE, "s"), (8, RED, "o")]:
        with np.errstate(invalid="ignore", divide="ignore"):
            roots, weights = roots_jacobi(nodes, -r, r - 1)
        xs, ws = (roots + 1) / 2, weights / weights.sum()
        ax.vlines(xs, 0, ws * dens.max() * 1.6, color=color, linewidth=1.2, alpha=0.85)
        ax.plot(xs, ws * dens.max() * 1.6, linestyle="none", marker=mk, color=color, markerfacecolor="white",
                label=f"узлы и веса, $J={nodes}$ (веса в условном масштабе)")
    ax.set_ylim(0, dens.max() * 0.6)
    ax.set_xlabel("$x$ (условное среднее фазы равно $\\theta x$)")
    ax.set_ylabel("Плотность / вес")
    ax.set_title(f"$r={r:.4f}$, $\\kappa={kappa}$, $\\theta={theta:.4f}$".replace(".", ","), fontsize=10.5)
    ax.legend(frameon=False, loc="upper center")
    tidy(ax)
    fig.savefig(FIG / "notes_quadrature.png")
    plt.close(fig)


def fig_convergence(data):
    rows = data["convergence"]
    phases = [r["phases"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    ax = axes[0]
    ax.semilogy(phases, [r["cdf_deviation"] for r in rows], color=INK, marker="o", markerfacecolor="white",
                linewidth=1.2)
    ax.set_xlabel("Число фаз $\\kappa+J$")
    ax.set_ylabel("$\\max_t|F_{\\mathrm{PH}}(t)-F(t)|$")
    tidy(ax, "both")
    ax = axes[1]
    for mode, label, st in [("rcs", "удаление RCS", dict(color=BLUE, marker="s", linestyle="--")),
                            ("disaster", "очищение", dict(color=RED, marker="^", linestyle="-."))]:
        last = rows[-1][f"w1_{mode}"]
        vals = [abs(r[f"w1_{mode}"] / last - 1) * 100 for r in rows[:-1]]
        ax.semilogy(phases[:-1], np.maximum(vals, 1e-7), markerfacecolor="white", linewidth=1.2, label=label, **st)
    ax.set_xlabel("Число фаз $\\kappa+J$")
    ax.set_ylabel("$|w_1(J)/w_1(10)-1|$, %")
    ax.legend(frameon=False)
    tidy(ax, "both")
    fig.savefig(FIG / "notes_convergence.png")
    plt.close(fig)


def fig_h2(data):
    rows = data["h2"]
    cv = np.array([r["cv"] for r in rows])
    y1 = np.array([complex(*r["y1"]) for r in rows])
    mu1 = np.array([complex(*r["mu1"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    ax = axes[0]
    ax.plot(cv, y1.real, color=INK, linewidth=1.4, label="$\\mathrm{Re}\\,y_1$")
    ax.plot(cv, 1 - y1.real, color=GREY, linewidth=1.2, linestyle="--", label="$\\mathrm{Re}\\,y_2$")
    ax.axhline(0, color="#bfbfbf", linewidth=0.8)
    ax.axvline(1, color="#bfbfbf", linewidth=0.8)
    ax.set_xlabel("CV")
    ax.set_ylabel("Вес ветви")
    ax.legend(frameon=False)
    tidy(ax)
    ax = axes[1]
    ax.plot(cv, mu1.real, color=BLUE, linewidth=1.3, label="$\\mathrm{Re}\\,\\mu_1$")
    ax.plot(cv, np.abs(mu1.imag), color=RED, linewidth=1.3, linestyle="--", label="$|\\mathrm{Im}\\,\\mu_1|$")
    ax.axvline(1, color="#bfbfbf", linewidth=0.8)
    ax.set_xlabel("CV")
    ax.set_ylabel("Интенсивность")
    ax.legend(frameon=False)
    tidy(ax)
    fig.savefig(FIG / "notes_h2_params.png")
    plt.close(fig)


def fig_substitution(data):
    rows = data["substitution"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    for ax, key, ylabel in [(axes[0], "dw_percent", "$\\Delta w_1$, %"), (axes[1], "dv_percent", "$\\Delta v_1$, %")]:
        for mode, label, st in [("rcs", "удаление RCS", dict(color=BLUE, marker="s", linestyle="--")),
                                ("disaster", "очищение", dict(color=RED, marker="^", linestyle="-."))]:
            sel = [r for r in rows if r["mode"] == mode]
            ax.plot([r["cv"] for r in sel], [r[key] for r in sel], markerfacecolor="white", linewidth=1.2,
                    label=label, **st)
        ax.axhline(0, color="#bfbfbf", linewidth=0.8)
        ax.set_xlabel("CV")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(comma)
        ax.legend(frameon=False)
        tidy(ax)
    fig.savefig(FIG / "notes_substitution.png")
    plt.close(fig)


def fig_cost():
    path = ROOT / "notes-cost.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    for ax, mode, title in [(axes[0], "rcs", "удаление RCS"), (axes[1], "disaster", "очищение")]:
        sel = [r for r in rows if r["mode"] == mode]
        for key, label, st in [("tt", "ТТ, $H_2$ (2 фазы)", dict(color=INK, marker="o", linestyle="-")),
                               ("ph8", "PH, $\\kappa+8$ фаз", dict(color=BLUE, marker="s", linestyle="--")),
                               ("ph16", "PH, $\\kappa+16$ фаз", dict(color=RED, marker="^", linestyle="-."))]:
            ax.semilogy([r["cv"] for r in sel], [r[key]["seconds"] for r in sel], markerfacecolor="white",
                        linewidth=1.2, label=label, **st)
        ax.set_xlabel("CV")
        ax.set_ylabel("Время расчёта, с")
        ax.set_title(title, fontsize=10.5)
        ax.legend(frameon=False)
        tidy(ax, "both")
    fig.savefig(FIG / "notes_cost.png")
    plt.close(fig)


if __name__ == "__main__":
    data = json.loads((ROOT / "notes-data.json").read_text())
    fig_density()
    fig_quadrature()
    fig_convergence(data)
    fig_h2(data)
    fig_substitution(data)
    fig_cost()
    print("figures written to", FIG)
