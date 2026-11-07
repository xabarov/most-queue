"""Render the revised article's figures/tables from the frozen experiment JSON."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATA = REPO / "docs/research"
FIG = ROOT / "figures"
TABLE = ROOT / "latex/tables"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "figure.constrained_layout.use": True,
    }
)
COLORS = ["#246b8e", "#e4a647", "#ad4a4a"]
FAMILIES = {"E3": r"$E_3$", "H2": r"$H_2$"}
SERIES = {"patience": "Нетерпение", "load": "Нагрузка", "decision": "Выбор параметров"}


def russian_cell(value):
    """Russian decimal commas and conventional scientific notation, display only."""
    value = str(value)
    scientific = re.fullmatch(r"(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)", value)
    if scientific:
        if float(scientific[1]) == 0:
            return "$0$"
        mantissa = scientific[1].replace(".", "{,}")
        return "$" + mantissa + r"\cdot10^{" + str(int(scientific[2])) + "}$"
    return re.sub(r"(?<=\d)\.(?=\d)", ",", value)


def load(panel):
    return json.loads((DATA / f"bulk-sla-revision-{panel}-2026.json").read_text())["cases"]


def save(fig, name):
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "linear":
                axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"revision_{name}.{suffix}", dpi=190)
    plt.close(fig)


def table(name, columns, headers, rows):
    text = [r"\begin{tabular}{" + columns + "}", r"\hline", " & ".join(headers) + r" \\", r"\hline"]
    text += [" & ".join(map(russian_cell, row)) + r" \\" for row in rows]
    text += [r"\hline", r"\end{tabular}"]
    (TABLE / f"{name}.tex").write_text("\n".join(text) + "\n")


def ci(value, index=None):
    lo, hi = value["lower"], value["upper"]
    if index is not None:
        lo, hi = lo[index], hi[index]
    return f"[{lo:.4f}; {hi:.4f}]"


def outcomes(patience):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), sharey=True)
    for ax, family in zip(axes, ("E3", "H2")):
        rows = [r for r in patience if r["case"]["family"] == family]
        x = np.array([r["case"]["gamma"] for r in rows])
        ys = [[r["selected"]["outcomes"][0][key] for r in rows] for key in ("on_time", "late", "abandoned")]
        ax.stackplot(
            x,
            *ys,
            colors=COLORS,
            labels=("Начало обслуживания в срок", "Начало обслуживания после срока", "Уход"),
            alpha=0.85,
        )
        observed = [r["simulation"]["t95"]["on_time"]["mean"][0] for r in rows]
        err = [
            (r["simulation"]["t95"]["on_time"]["upper"][0] - r["simulation"]["t95"]["on_time"]["lower"][0]) / 2
            for r in rows
        ]
        ax.errorbar(x, observed, yerr=err, fmt="ko", ms=3, capsize=3, label="ИМ, ДИ уровня 95%")
        ax.set(xlabel=r"Интенсивность ухода $\gamma$, 1/ед. времени", title=FAMILIES[family], ylim=(0, 1))
    axes[0].set_ylabel("Вероятность исхода")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    save(fig, "outcomes")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for family, color in zip(("E3", "H2"), COLORS):
        rows = [r for r in patience if r["case"]["family"] == family]
        x = [r["case"]["gamma"] for r in rows]
        axes[0].plot(x, [r["selected"]["moments"][0] for r in rows], "o-", color=color, label=FAMILIES[family])
        axes[1].plot(x, [r["selected"]["tails"][0] for r in rows], "o-", color=color, label=FAMILIES[family])
    for ax in axes:
        ax.set_xlabel(r"Интенсивность ухода $\gamma$, 1/ед. времени")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Условное среднее время\nожидания, ед. времени")
    axes[1].set_ylabel(r"$P(W>5\mid S)$")
    save(fig, "conditional")


def diagrams():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    states = {
        "s": (3.5, 3),
        "arrival": (7.5, 4.2),
        "death": (7.5, 2.8),
        "busy": (7.5, 1.25),
        "idle": (3.5, 0.3),
        "phase": (0.6, 3),
    }
    labels = {
        "s": r"$(i,\varphi,j)$",
        "arrival": r"$(i,\varphi,j+1)$",
        "death": r"$(i,\varphi,j-1)$",
        "busy": r"$(m,\psi,j-m)$",
        "idle": r"$I_j$",
        "phase": r"$(i,\psi,j)$",
    }
    for key, (x, y) in states.items():
        ax.text(
            x, y, labels[key], ha="center", va="center", bbox=dict(boxstyle="round,pad=.5", fc="#e8f1f5", ec=COLORS[0])
        )

    def arrow(target, label, xytext):
        ax.annotate("", xy=states[target], xytext=states["s"], arrowprops=dict(arrowstyle="->", shrinkA=34, shrinkB=42))
        ax.text(*xytext, label, ha="center", fontsize=9)

    arrow("arrival", r"$\lambda$", (5.1, 4.0))
    arrow("death", r"$j\gamma$", (5.45, 2.45))
    arrow("phase", r"$(T_i)_{\varphi\psi}$", (1.9, 3.35))
    arrow("busy", r"$j\geq a$: завершение", (6.0, 0.45))
    arrow("idle", r"$j<a$: завершение", (1.7, 1.6))
    ax.text(4.9, 4.9, r"$m=\min(b,j)$; новая фаза $\psi\sim\beta_m$", ha="center", fontsize=11)
    save(fig, "queue")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    ax.text(4.7, 4.5, r"Состояние в период ожидания: $(R,i,\varphi,K)$", ha="center", fontsize=12)
    ax.text(1.25, 3.35, r"$R\to R-1$: $R\gamma$", ha="center", bbox=dict(boxstyle="round", fc="#e8f1f5"))
    ax.text(
        4.75,
        3.35,
        r"$K\to K+1$: $\lambda$" + "\n" + r"$K\to K-1$: $K\gamma$",
        ha="center",
        bbox=dict(boxstyle="round", fc="#e8f1f5"),
    )
    ax.text(8.25, 3.35, "Уход меченой\n" + r"заявки: $\gamma$", ha="center", bbox=dict(boxstyle="round", fc="#f4dfdf"))
    ax.text(4.7, 2.4, "При завершении текущего пакета", ha="center", fontsize=11)
    for x, text in [
        (1.3, r"$R\geq b$" + "\n" + r"$R\to R-b$" + "\nновый пакет размера " + r"$b$"),
        (4.75, r"$R<b,\ R+1+K\geq a$" + "\nначало обслуживания\nмеченой заявки"),
        (8.2, r"$R<b,\ R+1+K<a$" + "\nпростой и накопление"),
    ]:
        ax.annotate("", xy=(x, 1.65), xytext=(4.7, 2.15), arrowprops=dict(arrowstyle="->"))
        ax.text(
            x,
            0.7,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=.4", fc="#e8f1f5", ec=COLORS[0]),
        )
    save(fig, "tagged")


def decisions(rows):
    counters = {}
    minima = []
    for metric, targets in (("conditional", (0.01, 0.001, 0.0001)), ("all_arrivals", (0.1, 0.05, 0.01))):
        for baseline in ("ordinary_gamma_tails", "atom_gamma_tails"):
            count = Counter()
            for row in rows:
                if "selected" not in row:
                    continue
                s = row["selected"]
                for j in range(3):
                    exact, fitted = s["tails"][j], s[baseline][j]
                    if metric == "all_arrivals":
                        exact = 1 - s["outcomes"][j]["on_time"]
                        fitted = 1 - s["served"] * (1 - fitted)
                    for eps in targets:
                        if abs(exact - eps) <= 1e-7 or abs(fitted - eps) <= 1e-7:
                            count["indeterminate"] += 1
                        elif fitted <= eps < exact:
                            count["false_accept"] += 1
                        elif exact <= eps < fitted:
                            count["false_reject"] += 1
                        else:
                            count["agree"] += 1
            counters[f"{metric}_{baseline}"] = dict(count)
        for family in ("E3", "H2"):
            for patience in (0.0, 0.1):
                for threshold in (1, 2, 3, 4):
                    candidates = [
                        r
                        for r in rows
                        if r["case"]["family"] == family
                        and r["case"]["gamma"] == patience
                        and r["case"]["a"] == threshold
                        and "selected" in r
                    ]
                    for j, deadline in enumerate((5, 10, 20)):
                        for target in targets:
                            chosen = {}
                            for key in ("tails", "ordinary_gamma_tails", "atom_gamma_tails"):

                                def fails(row):
                                    s = row["selected"]
                                    tail = s[key][j]
                                    return tail if metric == "conditional" else 1 - s["served"] * (1 - tail)

                                chosen[key] = next((r["case"]["b"] for r in candidates if fails(r) <= target), None)
                            minima.append(
                                dict(
                                    metric=metric,
                                    family=family,
                                    gamma=patience,
                                    a=threshold,
                                    deadline=deadline,
                                    target=target,
                                    minima=chosen,
                                )
                            )
    selected = [r for r in rows if r["case"]["family"] == "E3" and r["case"]["gamma"] == 0 and r["case"]["a"] == 1]
    table_rows = []
    for j, deadline in enumerate((5, 10, 20)):
        for eps in (0.01, 0.001, 0.0001):
            table_minima = [
                next((r["case"]["b"] for r in selected if r["selected"][key][j] <= eps), "---")
                for key in ("tails", "ordinary_gamma_tails", "atom_gamma_tails")
            ]
            table_rows.append([deadline, f"{eps:g}"] + table_minima)
    table("caps", "rrrrr", ["$D$", r"$\varepsilon$", "Расчёт", "Гамма", "Атом + гамма"], table_rows)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, metric in zip(axes, ("conditional", "all_arrivals")):
        values = np.full((4, 15), np.nan)
        for row in rows:
            c = row["case"]
            if c["family"] == "E3" and c["gamma"] == 0.1 and "selected" in row:
                s = row["selected"]
                failure = s["tails"][1] if metric == "conditional" else 1 - s["outcomes"][1]["on_time"]
                target = 0.001 if metric == "conditional" else 0.01
                values[c["a"] - 1, c["b"] - 2] = 1 if failure <= target else 0
        ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(1.5, 16.5, 0.5, 4.5),
            cmap=ListedColormap(["#f0d1c9", "#66a3aa"]),
            vmin=0,
            vmax=1,
        )
        ax.set_xticks(range(2, 17, 2))
        ax.set_yticks(range(1, 5))
        ax.set_xlabel(r"Максимальный размер пакета $b$")
        ax.set_title(
            r"$\overline{F}_S(D)\leq 0{,}001$" if metric == "conditional" else r"$1-S(D)\leq 0{,}01$", fontsize=11
        )
    axes[0].set_ylabel(r"Порог запуска $a$")
    save(fig, "feasible")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharex=True)
    for ax, family in zip(axes, ("E3", "H2")):
        for cap, color in zip((3, 5, 8, 16), ("#246b8e", "#a55632", "#469180", "#776497")):
            row = next(r for r in rows if r["case"] == dict(a=1, b=cap, family=family, gamma=0.0, **{"lambda": 0.8}))
            ax.semilogy((5, 10, 20), np.maximum(1e-5, row["selected"]["tails"]), "o-", color=color, label=f"$b={cap}$")
        for level in (0.01, 0.001, 0.0001):
            ax.axhline(level, color=".65", lw=0.7, ls=":")
        ax.set(xlabel="Предельное время ожидания\n" + r"$D$, ед. времени", title=FAMILIES[family], ylim=(1e-4, 1))
        ax.set_xticks((5, 10, 20))
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$P(W>D)$")
    save(fig, "tails")
    return counters, minima


def main():
    FIG.mkdir(exist_ok=True)
    TABLE.mkdir(exist_ok=True)
    patience, loads, decision, inoue = [load(p) for p in ("patience", "load", "decision", "inoue")]
    assert list(map(len, (patience, loads, decision, inoue))) == [12, 36, 228, 10]
    outcomes(patience)
    diagrams()
    counts, minima = decisions(decision)
    validations = []
    misses = []
    for r in patience:
        c, s, sim = r["case"], r["selected"], r["simulation"]["t95"]
        if c["gamma"] in (0, 0.1, 0.5):
            validations.append(
                [
                    FAMILIES[c["family"]],
                    f"{c['gamma']:g}",
                    f"{s['abandoned']:.4f}",
                    ci(sim["abandoned"]),
                    f"{s['outcomes'][0]['on_time']:.4f}",
                    ci(sim["on_time"], 0),
                ]
            )
        for key, pred in (
            ("mean_wait", s["moments"][0]),
            ("abandoned", s["abandoned"]),
            ("on_time", s["outcomes"][0]["on_time"]),
        ):
            low, high = sim[key]["lower"], sim[key]["upper"]
            if key == "on_time":
                low, high = low[0], high[0]
            if not low - 1e-12 <= pred <= high + 1e-12:
                misses.append(dict(case=c, metric=key, prediction=pred, interval=[low, high]))
    table(
        "validation",
        "lrrrrr",
        ["Закон", r"$\gamma$", "Расчёт: $p_A$", "ИМ: 95\\% ДИ", "Расчёт: $S(5)$", "ИМ: 95\\% ДИ"],
        validations,
    )
    inoue_rows = []
    for r in inoue:
        if r["case"]["b"] in (4, 8, None):
            inoue_rows.append(
                [
                    r["case"]["lambda"],
                    r["case"]["b"] or r"$\infty$",
                    ci(r["simulation"]["t95"]["mean_sojourn"]),
                    f"{r['unbounded_mean_sojourn']:.4f}" if "unbounded_mean_sojourn" in r else "---",
                    f"{r['inoue_sojourn_upper_bound']:.4f}" if "inoue_sojourn_upper_bound" in r else "---",
                ]
            )
    table(
        "inoue", "rrrrr", [r"$\lambda$", "$b$", "ИМ: $E[V]$, 95\\% ДИ", "$E[V]$, расчёт", "Оценка сверху"], inoue_rows
    )
    rows = []
    for key, value in counts.items():
        rows.append(
            [
                "Условный" if key.startswith("conditional") else "Безусловный",
                "Гамма" if "ordinary" in key else "Атом + гамма",
                value.get("false_accept", 0),
                value.get("false_reject", 0),
                value.get("indeterminate", 0),
            ]
        )
    table(
        "decisions",
        r"p{.17\linewidth}p{.21\linewidth}p{.17\linewidth}p{.15\linewidth}p{.16\linewidth}",
        ["Критерий", "Аппроксимация", "Ошибочное признание допустимости", "Ошибочное отклонение", "Неразличимо"],
        rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, family in zip(axes, ("E3", "H2")):
        for load_value, color in zip((0.3, 0.6, 0.85), COLORS):
            selected = [
                r
                for r in loads
                if r["case"]["family"] == family
                and abs(r["case"]["lambda"] * (0.1438 * 8 + 1.8874) / 8 - load_value) < 1e-10
                and "selected" in r
            ]
            if selected:
                ax.plot(
                    [r["case"]["a"] for r in selected],
                    [r["selected"]["moments"][0] for r in selected],
                    "o-",
                    color=color,
                    label="$r=" + f"{load_value:g}".replace(".", "{,}") + "$",
                )
        ax.set(xlabel=r"Порог запуска $a$", title=FAMILIES[family], xticks=(1, 2, 4))
        ax.grid(alpha=0.2)
        ax.legend()
        if family == "H2":
            ax.text(
                0.05,
                0.62,
                r"$r=0{,}85$: критерий" + "\nсходимости не выполнен",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
            )
    axes[0].set_ylabel("Среднее время ожидания, ед. времени")
    save(fig, "threshold")
    diagnostics = []
    for name, records in (("patience", patience), ("load", loads), ("decision", decision)):
        successful = [r["selected"] for r in records if "selected" in r]
        diagnostics.append(
            [
                SERIES[name],
                f"{len(successful)}/{len(records)}",
                max(s["N"] for s in successful),
                f"{max(s['queue_boundary_mass'] for s in successful):.1e}",
                f"{max(abs(s['abandon_flow_residual']) for s in successful):.1e}",
            ]
        )
    table(
        "diagnostics",
        r"p{.20\linewidth}p{.14\linewidth}p{.10\linewidth}p{.19\linewidth}p{.23\linewidth}",
        ["Серия расчётов", "Критерий выполнен", r"$N_{\max}$", "Вероятность границы, макс.", "Невязка баланса, макс."],
        diagnostics,
    )
    local_rows = []
    for name, records in (("patience", patience), ("decision", decision)):
        selected = [r["selected"] for r in records if "selected" in r]
        local_rows.append(
            [
                SERIES[name],
                max(s["behind_cap"] for s in selected),
                f"{max(s['initial_omitted_mass'] for s in selected):.2e}",
            ]
            + [f"{max(s['local_moment_bounds'][i] for s in selected):.2e}" for i in range(3)]
        )
    table(
        "local_bounds",
        "lrrrrr",
        ["Серия расчётов", r"$C_{\max}$", r"$\delta_{\max}$", "$e_0$", "$e_1$", "$e_2$"],
        local_rows,
    )
    summary = {
        "decision_counts": counts,
        "configuration_minima": minima,
        "patience_pointwise_t95_misses": misses,
        "statuses": {
            name: dict(Counter(r.get("status", "simulation") for r in records))
            for name, records in zip(("patience", "load", "decision", "inoue"), (patience, loads, decision, inoue))
        },
        "data_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DATA.glob("bulk-sla-revision-*-2026.json"))
        },
    }
    (ROOT / "revision_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
