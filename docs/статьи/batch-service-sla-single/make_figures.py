"""Generate the two figures for the batch-service SLA exact-tail article.

Personal article material (not part of the most-queue test/example suite).
Reuses the exact library calculators with the SAME parameters as the main
comparison script (examples/batch_service_sla_exact_tail_comparison.py) so
numbers in the figures match the numbers in the article tables exactly.

Print-oriented style: serif font (matches journal body text), no in-plot
title (the caption goes in the article text as "Рисунок N."), distinct
marker+linestyle per series so the figure survives black-and-white print.

    .venv/bin/python "docs/статьи/batch-service-sla/make_figures.py"
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc  # noqa: E402

ALPHA, TAU0, LAM = 0.1438, 1.8874, 0.8
ERLANG_K = 10
QUEUE_TRUNCATION = 400
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[3]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "lines.markeredgewidth": 0.8,
        "mathtext.fontset": "dejavuserif",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

# Print-safe encoding: distinct marker + linestyle per series, not color alone.
SERIES_STYLE = {
    "erlang": {"marker": "o", "linestyle": "-", "color": "#000000", "label": "$E_k$, CV = 0,316"},
    "h2": {"marker": "s", "linestyle": "--", "color": "#595959", "label": "$H_2$, CV = 2,0"},
}
BOUND_STYLE = {"linestyle": ":", "color": "#000000", "label": "граница Inoue (2021), $b_{\\max}=\\infty$"}


def erlang_rate_fn(size):
    """Per-phase Erlang rate giving mean batch time alpha*size + tau0 at k=ERLANG_K phases."""
    return ERLANG_K / (ALPHA * size + TAU0)


def figure_mean_wait_vs_bmax():
    """Fig. 1: E[W] vs b_max for both variance regimes, with Inoue's bound as a reference line."""
    comparison_path = REPO_ROOT / "works" / "batch_service_sla_exact_tail" / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    b_max = [r["b_max"] for r in comparison["erlang_low_variance"]]
    erlang_w = [r["exact_mean_w"] for r in comparison["erlang_low_variance"]]
    h2_w = [r["exact_mean_w"] for r in comparison["h2_high_variance"]]
    bound = comparison["inoue_2021_bound_e_w"]

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
    ax.plot(b_max, erlang_w, markersize=5, linewidth=1.3, **SERIES_STYLE["erlang"])
    ax.plot(b_max, h2_w, markersize=5, linewidth=1.3, **SERIES_STYLE["h2"])
    ax.axhline(bound, linewidth=1.1, **BOUND_STYLE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(b_max)
    ax.set_xticklabels([str(b) for b in b_max])
    ax.set_xlabel(r"$b_{\max}$")
    ax.set_ylabel(r"$E[W]$")
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)
    ax.grid(True, which="major", axis="y", linewidth=0.3, color="#bfbfbf")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_mean_wait_vs_bmax.png")
    fig.savefig(FIG_DIR / "fig1_mean_wait_vs_bmax.pdf")
    plt.close(fig)


def figure_tail_curves():
    """Fig. 2: exact P(W>t) tail curves (Erlang case) for a few b_max values, log-y."""
    b_max_values = (2, 8, 32)
    t_grid = np.linspace(0.0, 15.0, 31)
    styles = [
        {"marker": "o", "linestyle": "-", "color": "#000000"},
        {"marker": "s", "linestyle": "--", "color": "#595959"},
        {"marker": "^", "linestyle": "-.", "color": "#a6a6a6"},
    ]

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
    for b_max, style in zip(b_max_values, styles):
        calc = BulkServiceErlangCalc(a=1, b=b_max, k=ERLANG_K, queue_truncation=QUEUE_TRUNCATION)
        calc.set_sources(LAM)
        calc.set_servers(erlang_rate_fn)
        tail = np.array([calc.get_tail(t) for t in t_grid])
        tail = np.clip(tail, 1e-20, None)  # avoid log(0) for the plot only
        ax.plot(
            t_grid,
            tail,
            markevery=5,
            markersize=4.5,
            linewidth=1.3,
            label=rf"$b_{{\max}}={b_max}$",
            **style,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P(W>t)$")
    ax.legend(frameon=False, handlelength=2.4)
    ax.grid(True, which="major", axis="y", linewidth=0.3, color="#bfbfbf")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_exact_tail_curves.png")
    fig.savefig(FIG_DIR / "fig2_exact_tail_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_mean_wait_vs_bmax()
    figure_tail_curves()
    print("wrote figures to", FIG_DIR)
