"""v9: result tables, table manifest and summary numbers from saved JSON.

Sources (no new calculations here):
  ../revision_v7_20261107/experiment-results.json — restarts with new length,
      original 8-run simulation of the main tables, extended experiment;
  tt-results.json — removal and clearing by Takahashi-Takami + tagged equations;
  corrected-results.json — retained work with the quadrature correction;
  long_sim/*.json — 32 x 2e6 simulation series (v9).
Writes tables_v9.tex, table-manifest.json (with tab:methods), summary-v9.json.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V7 = json.loads((ROOT.parent / "revision_v7_20261107/experiment-results.json").read_text())
TT = json.loads((ROOT / "tt-results.json").read_text())
COR = json.loads((ROOT / "corrected-results.json").read_text())
LONG = {p.stem: json.loads(p.read_text())["summary"] for p in (ROOT / "long_sim").glob("*.json")}
ORDER = ["RCS, удаление", "Очищение", "RCS, новая длительность", "Все каналы, новая длительность"]
ORDER_FIXED = ["RCS, сохранённая длительность", "Все каналы, сохранённая длительность"]
WIDTHS = [4.16, 1.61, 2.83, 1.28, 1.61, 2.83, 1.28]


def num(x, d=4):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", "−")


def delta(calc, sim):
    return 100 * (calc / sim - 1)


def cell(calc, sim, half):
    return [num(calc), f"{num(sim)} ± {num(half)}", num(delta(calc, sim), 2)]


def main_rows():
    """Rows of tables 2-4: label, cv, calc dict (w,v,sw,sv), sim dict, method, sim series."""
    rows = []
    tt = {(r["label"], r["cv"]): r for r in TT["main"]}
    cor = {(r["label"], r["cv"]): r for r in COR["main"]}
    for item in V7["results"]:
        key = (item["label"], item["cv"])
        if key in tt:
            t = tt[key]["tt"]
            calc = {"w": t["w"][0], "v": t["v"][0], "sw": t["sw"], "sv": t["sv"]}
            sim = {k: item["simulation"][k] for k in ("w", "v", "sw", "sv")}
            rows.append((item["label"], item["cv"], calc, sim, "TT", "8x2e5"))
        elif key in cor:
            c = cor[key]["corrected"]
            calc = {"w": c["w"][0], "v": c["v"][0], "sw": math.sqrt(c["w"][1] - c["w"][0] ** 2),
                    "sv": math.sqrt(c["v"][1] - c["v"][0] ** 2)}
            name = "main-fixed-" + item["mode"] + f"-cv{item['cv']}"
            rows.append((item["label"], item["cv"], calc, LONG[name], "retained+correction", "32x2e6"))
        else:
            c = item["calculation"]
            calc = {"w": c["w"][0], "v": c["v"][0], "sw": c["sw"], "sv": c["sv"]}
            sim = {k: item["simulation"][k] for k in ("w", "v", "sw", "sv")}
            rows.append((item["label"], item["cv"], calc, sim, "joint PH", "8x2e5"))
    return rows


def table(rows, labels, first, second, caption, label, keys=("w", "v")):
    body, expected = [], []
    for cv in (0.8, 1.2):
        body.append(r"\multicolumn{7}{l}{Коэффициент вариации: " + num(cv, 1) + "} \\\\")
        for lab in labels:
            r = next(x for x in rows if x[0] == lab and x[1] == cv)
            calc, sim = r[2], r[3]
            cells = [lab] + cell(calc[keys[0]], sim[keys[0]]["mean"], sim[keys[0]]["half_ci95"]) \
                + cell(calc[keys[1]], sim[keys[1]]["mean"], sim[keys[1]]["half_ci95"])
            expected.append(cells)
            body.append(" & ".join(cells) + r" \\")
    tex = "\n".join([
        r"\begin{table}[!htbp]",
        r"\centering\small\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}p{4.16cm}p{1.61cm}p{2.83cm}p{1.28cm}p{1.61cm}p{2.83cm}p{1.28cm}@{}}",
        r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{{first}}} & \multicolumn{{3}}{{c}}{{{second}}} \\",
        r"Вариант & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"])
    return tex, {"id": label, "widths_cm": WIDTHS, "header_rows": 2, "expected_rows": expected, "font_pt": 10}


def law_table():
    rows, expected = [], []
    for r in TT["law"]:
        if not (r["n"] == 3 and abs(r["rho"] - 0.7) < 1e-9):
            continue
        t, g = r["tt"], r["gamma_ph16"]
        lab = ("RCS, удаление" if r["mode"] == "rcs" else "Очищение")
        cells = [num(r["cv"], 1), lab, num(t["w"][0]), num(g["w"][0]), num(delta(t["w"][0], g["w"][0]), 2),
                 num(t["v"][0]), num(g["v"][0]), num(delta(t["v"][0], g["v"][0]), 2)]
        expected.append(cells)
        rows.append(" & ".join(cells) + r" \\")
    tex = "\n".join([
        r"\begin{table}[!htbp]",
        r"\centering\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}p{1.0cm}p{3.2cm}p{1.6cm}p{1.6cm}p{1.3cm}p{1.6cm}p{1.6cm}p{1.3cm}@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$w_1$} & \multicolumn{3}{c}{$v_1$} \\",
        r"$\mathrm{CV}$ & Вариант & $H_2$ & Гамма & $\Delta$, \% & $H_2$ & Гамма & $\Delta$, \% \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Погрешность замены гамма-распределения двухфазным законом с теми же тремя моментами: "
        r"\(n=3\), \(\rho=0{,}7\), \(\delta=0{,}3\). Столбец «\(H_2\)» --- метод Такахаси--Таками, "
        r"«Гамма» --- положительное фазовое представление с 16 узлами квадратуры. "
        r"\(\Delta=100(H_2-\text{гамма})/\text{гамма}\).}",
        r"\label{tab:law}", r"\end{table}"])
    return tex, {"id": "tab:law", "widths_cm": [1.0, 3.2, 1.6, 1.6, 1.3, 1.6, 1.6, 1.3], "header_rows": 2,
                 "expected_rows": expected, "font_pt": 10}


def extended_summary():
    tt = {r["case"]["id"]: r for r in TT["extended"]}
    cor = {r["case"]["id"]: r for r in COR["extended"]}
    out, misses, maxima = [], [], {"w": 0, "v": 0, "sw": 0, "sv": 0}
    for item in V7["extended"]:
        cid = item["case"]["id"]
        if cid in tt:
            t = tt[cid]
            calc = {"w": t["tt"]["w"][0], "v": t["tt"]["v"][0], "sw": t["tt"]["sw"], "sv": t["tt"]["sv"]}
            sim = t["simulation"]
            source = "TT, 8x5e5"
        elif cid in cor:
            ls = cor[cid]["long_simulation"]
            calc = {k: ls[k]["calculation"] for k in ls}
            sim = {k: {"mean": ls[k]["simulation"], "half_ci95": ls[k]["half_ci95"]} for k in ls}
            source = "retained+correction, 32x2e6"
        elif cid in LONG:
            calc = {k: item["comparison"][k]["calculation"] for k in ("w", "v", "sw", "sv")}
            sim = {k: LONG[cid][k] for k in ("w", "v", "sw", "sv")}
            source = "joint PH, 32x2e6"
        else:
            calc = {k: item["comparison"][k]["calculation"] for k in ("w", "v", "sw", "sv")}
            sim = {k: {"mean": item["comparison"][k]["simulation"], "half_ci95": item["comparison"][k]["half_ci95"]}
                   for k in ("w", "v", "sw", "sv")}
            source = "joint PH, 8x5e5"
        row = {"id": cid, "case": item["case"], "source": source}
        for k in ("w", "v", "sw", "sv"):
            d = delta(calc[k], sim[k]["mean"])
            inside = abs(calc[k] - sim[k]["mean"]) <= sim[k]["half_ci95"]
            row[k] = {"calculation": calc[k], "simulation": sim[k]["mean"], "half_ci95": sim[k]["half_ci95"],
                      "difference_percent": d, "inside_ci95": inside}
            maxima[k] = max(maxima[k], abs(d))
            if not inside:
                misses.append((cid, k))
        out.append(row)
    return out, maxima, misses


def main():
    rows = main_rows()
    t2, m2 = table(rows, ORDER, "$w_1$", "$v_1$",
                   r"Средние при $\delta=0{,}3$. Удаление и очищение рассчитаны методом Такахаси--Таками, повторы --- "
                   r"совместной фазовой моделью. ДИ --- полуширина 95\%-го доверительного интервала по восьми прогонам. "
                   r"«Все каналы» означает одновременный повтор на занятых каналах.", "tab:results")
    t3, m3 = table(rows, ORDER_FIXED, "$w_1$", "$v_1$",
                   r"Средние при повторе с сохранённой длительностью, $\delta=0{,}02$; расчёт с поправкой квадратуры. "
                   r"ДИ --- по 32 прогонам длиной $2\cdot10^6$ уходов.", "tab:results_fixed")
    t4, m4 = table(rows, ORDER + ORDER_FIXED, r"$\sigma_W$", r"$\sigma_V$",
                   r"Стандартные отклонения ожидания и пребывания. Варианты и серии ИМ те же, что в таблицах~\ref{tab:results} и~\ref{tab:results_fixed}.",
                   "tab:sd", keys=("sw", "sv"))
    t5, m5 = law_table()
    (ROOT / "tables_v9.tex").write_text("\n\n".join(["% tab:results", t2, "% tab:results_fixed", t3, "% tab:sd", t4,
                                                    "% tab:law", t5]) + "\n")
    old = json.loads((ROOT.parent / "revision_v8_20261107/table-manifest.json").read_text())
    methods = next(x for x in old if x["id"] == "tab:methods")
    (ROOT / "table-manifest.json").write_text(json.dumps([methods, m2, m3, m4, m5], ensure_ascii=False, indent=1))
    ext, maxima, misses = extended_summary()
    main_max = {k: max(abs(delta(r[2][k], r[3][k]["mean"])) for r in rows) for k in ("w", "v", "sw", "sv")}
    main_misses = [(r[0], r[1], k) for r in rows for k in ("w", "v", "sw", "sv")
                   if abs(r[2][k] - r[3][k]["mean"]) > r[3][k]["half_ci95"]]
    law_max = {}
    for r in TT["law"]:
        for k in ("w", "v"):
            key = f"cv{r['cv']}-{k}"
            law_max[key] = max(law_max.get(key, 0), abs(delta(r["tt"][k][0], r["gamma_ph16"][k][0])))
    summary = {"main_max_abs_percent": main_max, "main_misses": main_misses,
               "extended_max_abs_percent": maxima, "extended_misses": misses, "extended": ext,
               "law_max_abs_percent_all_conditions": law_max,
               "correction_shift_w1_percent": {(r.get("label") or r["case"]["id"]) + f" cv{r.get('cv') or r['case']['cv']}":
                                               r["correction"]["relative_shift_w1_percent"] for r in COR["main"] + COR["extended"]},
               "cells": sum(len(row) - 1 for m in (m2, m3, m4, m5) for row in m["expected_rows"])}
    (ROOT / "summary-v9.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "extended"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
