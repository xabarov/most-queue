"""Assemble the separately versioned manuscript and tables from recorded evidence."""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent


def number(x):
    return f"{x:.4f}".replace(".", ",")


def table(rows, keys, title, label):
    header = [
        r"\begin{table}[!htbp]",
        r"\centering\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{${keys[2]}$}} & \multicolumn{{3}}{{c}}{{${keys[3]}$}} \\",
        r"Вариант & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% \\",
        r"\midrule",
    ]
    for cv in [0.8, 1.2]:
        header += [rf'\multicolumn{{7}}{{c}}{{$CV={str(cv).replace(".",",")}$}} \\']
        for row in rows:
            if row["cv"] != cv:
                continue
            labels = {
                ("rcs", "remove"): "RCS, удаление",
                ("disaster", "remove"): "Очищение",
                ("rcs", "restart"): "RCS, новый $B$",
                ("disaster", "restart"): "Общий, новый $B$",
                ("rcs", "fixed"): "RCS, прежний $B$",
                ("disaster", "fixed"): "Общий, прежний $B$",
            }
            cells = [labels[row["mode"], row["scenario"]]]
            for key in keys[:2]:
                c = row["calculation"][key]
                if key in ["w", "v"]:
                    c = c[0]
                s = row["simulation"][key]
                cells += [
                    number(c),
                    f'{number(s["mean"])} ± {number(s["half_ci95"])}',
                    f'{100*(c/s["mean"]-1):.2f}'.replace(".", ","),
                ]
            header += [" & ".join(cells) + r" \\"]
    header += [r"\bottomrule", r"\end{tabular}", r"\caption{" + title + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(header)


def figures():
    spec = importlib.util.spec_from_file_location("draw", OLD / "word/redraw_figure1.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    d.ROOT = ROOT / "figures"
    d.ROOT.mkdir(exist_ok=True)
    d.rcs()
    fig, ax = d.canvas(8.3)
    ys = {0: 0.55, 1: 2.1, 2: 3.85, 3: 5.6, 4: 7.35}
    dx = 13.8
    d.label(ax, 0.4, 0.25, "$i$", 11)
    d.label(ax, 0.4, ys[0], "0", 11)
    d.node(ax, dx, ys[0], "00")
    for i in range(1, 5):
        d.label(ax, 0.4, ys[i], str(i), 11)
        d.node(ax, dx, ys[i], rf"$\mathcal{{D}}_{i}$", dashed=True)
        d.arrow(ax, (dx, ys[i] - 0.30), (dx, ys[i - 1] + 0.30))
        d.label(ax, dx + 0.45, (ys[i] + ys[i - 1]) / 2, r"$\gamma$", 12)
        count = min(i, 3) + 1
        xs = [3.2 + 1.65 * j for j in range(count)]
        xleft, xright = xs[0] - 0.7, xs[-1] + 0.7
        ax.add_patch(
            FancyBboxPatch(
                (xleft, ys[i] - 0.4),
                xright - xleft,
                0.8,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                edgecolor=d.INK,
                facecolor="white",
                linewidth=0.8,
            )
        )
        for j, x in enumerate(xs):
            d.label(ax, x, ys[i], f"{min(i,3)-j}{j}", 12)
        d.arrow(ax, (xright + 0.05, ys[i] - 0.16), (dx - 0.60, ys[i - 1] + 0.1))
        d.label(ax, (xright + dx) / 2 - 0.1, (ys[i] + ys[i - 1]) / 2 - 0.4, r"$\delta$", 12)
        d.EDGES["disaster"] += [
            dict(source=[i, j], target=[i - 1, "D"] if i > 1 else [0, 0], rate="delta") for j in range(count)
        ]
        d.EDGES["disaster"].append(dict(source=[i, "D"], target=[i - 1, "D"] if i > 1 else [0, 0], rate="gamma"))
    d.save(fig, "figure1b_disaster")
    (d.ROOT / "transitions.json").write_text(json.dumps(d.EDGES, ensure_ascii=False, indent=2))


def main():
    data = json.loads((ROOT / "experiment-results.json").read_text())["results"]
    tables = [
        table(
            [r for r in data if r["scenario"] != "fixed"],
            ("w", "v", "w_1", "v_1"),
            r"Средние ожидание и пребывание при $\delta=0{,}3$. ДИ --- полуширина 95\%-го доверительного интервала; «общий» означает одновременный повтор на всех занятых каналах.",
            "tab:results",
        ),
        table(
            [r for r in data if r["scenario"] == "fixed"],
            ("w", "v", "w_1", "v_1"),
            r"Средние при повторе с сохранённой длительностью, $\delta=0{,}02$.",
            "tab:results_fixed",
        ),
        table(
            data,
            ("sw", "sv", r"\sigma_W", r"\sigma_V"),
            r"Стандартные отклонения ожидания и пребывания. Интенсивности соответствуют таблицам 1 и 2.",
            "tab:sko",
        ),
    ]
    (ROOT / "generated_tables_fixed.tex").write_text("\n\n".join(tables) + "\n")
    preamble = (OLD / "negative_queues_takahasi_takami.tex").read_text().split(r"\begin{abstract}")[0]
    preamble = preamble.replace(r"\begin{samepage}", "").replace(
        r"\begin{document}", r"\emergencystretch=3em" + "\n" + r"\begin{document}"
    )
    body = (ROOT / "article_body.tex").read_text().replace(r"\end{samepage}", "")
    body = body.replace(r"\).", r".\)")  # keep sentence punctuation with inline mathematics
    (ROOT / "negative_queues_takahasi_takami.tex").write_text(preamble + body)
    bib = (OLD / "citations.bib").read_text()
    start = bib.index("@misc{mostqueue2026,")
    bib = (
        bib[:start]
        + """@misc{mostqueue2026,
  author = {Khabarov, Roman},
  title = {Most-Queue: queueing theory calculations and simulation in Python},
  howpublished = {\\url{https://github.com/xabarov/most-queue}},
  note = {Дата обращения: 07.11.2026}
}\n"""
    )
    bib += """\n@article{pal2017,
  author = {Pal, Arnab and Reuveni, Shlomi},
  title = {First Passage under Restart},
  journal = {Physical Review Letters},
  year = {2017}, volume = {118}, number = {3}, pages = {030603},
  doi = {10.1103/PhysRevLett.118.030603}
}\n"""
    # Verified against publisher records. The old Gelenbe/Schassberger entry
    # combined a nonexistent title with the wrong journal, pages and given name.
    bib = re.sub(
        r"@article\{gelenbe1992,.*?\n\}",
        """@article{gelenbe1992,
  author = {Gelenbe, Erol and Schassberger, Rolf},
  title = {Stability of Product Form {G}-Networks},
  journal = {Probability in the Engineering and Informational Sciences},
  year = {1992}, volume = {6}, number = {3}, pages = {271--276},
  doi = {10.1017/S0269964800002539}
}""",
        bib,
        flags=re.S,
    )

    # The local 2018 file is an author typesetting draft (blank ISBN).
    # Published edition: Lan, 2019, ISBN 978-5-8114-3462-6.
    def book_year(m):
        return (
            m[0]
            .replace("{2018}", "{2019}")
            .replace("  language", "  isbn = {978-5-8114-3462-6},\n  pagetotal = {512},\n  language")
        )

    bib = re.sub(r"@book\{ryzhikov2018ru,.*?\n\}", book_year, bib, flags=re.S)
    cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", body) for key in group.split(",")}
    bib = (
        "\n\n".join(
            entry
            for entry in re.findall(r"@\w+\{.*?\n\}", bib, re.S)
            if entry.split("{", 1)[1].split(",", 1)[0] in cited
        )
        + "\n"
    )
    (ROOT / "citations.bib").write_text(bib)
    figures()
    print(ROOT / "negative_queues_takahasi_takami.tex")


if __name__ == "__main__":
    main()
