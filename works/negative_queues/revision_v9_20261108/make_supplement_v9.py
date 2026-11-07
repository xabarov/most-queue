"""v9: supplement LaTeX source and table manifest from saved JSON.

Unchanged v8 tables (finite grids, sensitivity, cost, positive PH, Erlang-4) are
copied verbatim; new tables are generated from tt-results.json,
corrected-results.json, summary-v9.json, diagnostics/*.json and long_sim/*.json.
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V8S = ROOT.parent / "revision_v8_20261107/supplement/negative_queues_takahasi_takami.tex"
V8M = json.loads((ROOT.parent / "revision_v8_20261107/supplement/table-manifest.json").read_text())
V7 = json.loads((ROOT.parent / "revision_v7_20261107/experiment-results.json").read_text())
TT = json.loads((ROOT / "tt-results.json").read_text())
COR = json.loads((ROOT / "corrected-results.json").read_text())
SUM = json.loads((ROOT / "summary-v9.json").read_text())
TTCHK = json.loads((ROOT / "diagnostics/tt_vs_sparse.json").read_text())
LONG = {p.stem: json.loads(p.read_text()) for p in (ROOT / "long_sim").glob("*.json")}
MAIN = json.loads((ROOT / "negative_queues_takahasi_takami.tex").read_text() if False else "null")
MANIFEST = []
W7 = [4.16, 1.61, 2.83, 1.28, 1.61, 2.83, 1.28]
WE = [1.5, 1.9, 3.4, 1.4, 1.9, 3.4, 1.4]


def num(x, d=4):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", "−")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return rf"\({m.replace('.', '{,}')}\cdot10^{{{int(e)}}}\)"


def rendered(cell):
    m = re.fullmatch(r"\\\((\d+)\{,\}(\d+)\\cdot10\^\{(-?\d+)\}\\\)", cell)
    return f"{m[1]},{m[2]}⋅10{m[3].replace('-', '−')}" if m else cell


def table(label, widths, header, rows, caption, header_rows=1, font=10):
    spec = "".join(f"p{{{w}cm}}" for w in widths)
    body = []
    expected = []
    for r in rows:
        if isinstance(r, str):
            body.append(r)
        else:
            body.append(" & ".join(r) + r" \\")
            expected.append([rendered(c) for c in r])
    MANIFEST.append({"id": label, "widths_cm": widths, "header_rows": header_rows, "expected_rows": expected,
                     "font_pt": font})
    return "\n".join([r"\begin{table}[!htbp]", r"\centering\small\setlength{\tabcolsep}{2pt}",
                      rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule", *header, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\end{table}"])


def copy_v8(label):
    text = V8S.read_text()
    block = re.search(r"\\begin\{table\}(?:(?!\\end\{table\}).)*?\\label\{" + re.escape(label) + r"\}\s*\\end\{table\}",
                      text, re.S)[0]
    spec = json.loads(json.dumps(next(m for m in V8M if m["id"] == label)))
    fix = lambda v: re.sub(r"(?<![\w,])-(?=\d)", "−", v)
    spec["expected_rows"] = [[fix(c) for c in row] for row in spec["expected_rows"]]
    MANIFEST.append(spec)
    block = re.sub(r"(?<=& )-(?=\d)", "−", block)
    return block.replace("$CV$", r"$\mathrm{CV}$").replace("\\(CV", "\\(\\mathrm{CV}")


def triple(calc, mean, half):
    return [num(calc), f"{num(mean)} ± {num(half)}", num(100 * (calc / mean - 1), 2)]


HEAD7 = [r" & \multicolumn{3}{c}{$w_1$} & \multicolumn{3}{c}{$v_1$} \\",
         r"Вариант & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% \\"]
HEAD7S = [r" & \multicolumn{3}{c}{$\sigma_W$} & \multicolumn{3}{c}{$\sigma_V$} \\",
          r"Вариант & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% & Расчёт & ИМ $\pm$ ДИ & $\Delta$, \% \\"]


def tt_tables():
    rows = []
    for r in TTCHK:
        mech = "RCS, удаление" if r["mode"] == "rcs" else "Очищение"
        y1 = complex(*r["h2"]["y"][0])
        kind = "компл." if abs(y1.imag) > 1e-12 else ("отриц. вес" if y1.real > 1 else "положит.")
        rows.append([num(r["cv"], 1), mech, kind, sci(r["max_rel_moments"]), sci(r["balance_residual"])])
    t1 = table("tab:ttcheck", [1.2, 4.2, 2.6, 3.6, 3.6],
               [r"$\mathrm{CV}$ & Механизм & Параметры $H_2$ & Отн. разность моментов & Невязка баланса \\"], rows,
               r"Сверка метода Такахаси--Таками с прямым разреженным решением тех же уравнений: \(n=3\), \(\lambda=1\), "
               r"\(\delta=0{,}3\), \(N=200\), \(H=60\). Разность --- максимум по \(w_1,w_2,v_1,v_2\).")
    rows = []
    for r in TT["law"]:
        t, g = r["tt"], r["gamma_ph16"]
        mech = "RCS" if r["mode"] == "rcs" else "Очищ."
        rows.append([num(r["cv"], 1), str(r["n"]), num(r["rho"], 2), num(r["delta"], 2), mech,
                     num(t["w"][0]), num(100 * (t["w"][0] / g["w"][0] - 1), 2),
                     num(t["v"][0]), num(100 * (t["v"][0] / g["v"][0] - 1), 3)])
    t2 = table("tab:lawfull", [1.0, 0.8, 1.2, 1.2, 1.5, 2.0, 2.0, 2.0, 2.0],
               [r"$\mathrm{CV}$ & $n$ & $\rho$ & $\delta$ & Мех. & $w_1$ ($H_2$) & $\Delta_W$, \% & $v_1$ ($H_2$) & $\Delta_V$, \% \\"],
               rows, r"Погрешность двухфазного закона во всех рассмотренных режимах относительно положительного фазового "
                     r"представления гамма-распределения с 16 узлами.")
    return t1, t2


def correction_tables():
    rows = []
    for r in COR["main"] + COR["extended"]:
        name = (r["label"].replace(", сохранённая длительность", "") + ", " + num(r["cv"], 1)) if "label" in r else \
            f"Условие {int(r['case']['id'][4:]) + 1}"
        c = r["correction"]
        rows.append([name, num(r["raw_v8"]["w"][0]), num(c["delta0_extrapolated"]["w"][0]),
                     num(c["delta0_reference"]["w"][0]), num(c["relative_shift_w1_percent"], 3),
                     num(r["corrected"]["w"][0])])
    t1 = table("tab:correction", [4.4, 2.2, 2.2, 2.2, 1.9, 2.2],
               [r"Вариант & $w_1$ без попр. & $w_1(0)$, модель & $w_1(0)$, эталон & Сдвиг, \% & $w_1$ с попр. \\"], rows,
               r"Поправка квадратуры для сохранённой работы. \(w_1(0)\) --- среднее ожидание при \(\delta=0\) на той же сетке "
               r"(модель) и для исходного закона (эталон). Условия 9--12 --- из расширенного эксперимента.")
    v = COR["validation_n2"]
    rows = [[str(j), num(v["J_sequence_delta0.04"][str(j)], 5), num(v["corrected_from_J"][str(j)], 5)] for j in (3, 4, 5, 6)]
    t2 = table("tab:jseq", [1.5, 4.0, 4.0], [r"$J$ & $w_1$ без поправки & $w_1$ с поправкой \\"], rows,
               r"Проверка поправки при \(n=2\), \(\rho=0{,}5\), \(\mathrm{CV}=0{,}8\), RCS, \(\delta=0{,}04\), "
               r"\(K=8\), 12, 16. Предел последовательности без поправки по геометрической экстраполяции "
               rf"равен {num(v['geometric_limit'], 5)}.")
    rows = []
    for r in COR["main"]:
        lab = r["label"].replace(", сохранённая длительность", "") + ", " + num(r["cv"], 1)
        o = r["original_simulation"]
        rows.append([lab] + triple(o["w"]["calculation"], o["w"]["simulation"], o["w"]["half_ci95"])
                    + triple(o["v"]["calculation"], o["v"]["simulation"], o["v"]["half_ci95"]))
    t3 = table("tab:oldsim", W7, HEAD7, rows,
               r"Расчёт с поправкой и прежняя серия ИМ для сохранённой работы (8 прогонов, исключение \(2\cdot10^4\) "
               r"и наблюдение \(2\cdot10^5\) уходов).", header_rows=2)
    return t1, t2, t3


def extended_tables():
    ext = {r["id"]: r for r in SUM["extended"]}
    rows_m, rows_s, rows_p = [], [], []
    for item in V7["extended"]:
        cid = item["case"]["id"]
        k = str(int(cid[4:]) + 1)
        r = ext[cid]
        rows_m.append([k] + triple(r["w"]["calculation"], r["w"]["simulation"], r["w"]["half_ci95"])
                      + triple(r["v"]["calculation"], r["v"]["simulation"], r["v"]["half_ci95"]))
        rows_s.append([k] + triple(r["sw"]["calculation"], r["sw"]["simulation"], r["sw"]["half_ci95"])
                      + triple(r["sv"]["calculation"], r["sv"]["simulation"], r["sv"]["half_ci95"]))
        c = item["case"]
        mech = {("remove", "rcs"): "RCS, удаление", ("remove", "disaster"): "Очищение",
                ("restart", "rcs"): "RCS, новая длительность", ("restart", "disaster"): "Все каналы, новая длительность",
                ("fixed", "rcs"): "RCS, сохранённая длительность", ("fixed", "disaster"): "Все каналы, сохранённая длительность"}[(c["action"], c["mode"])]
        method = "ТТ" if r["source"].startswith("TT") else ("СФМ+попр." if "correction" in r["source"] else "СФМ")
        series = "32×2 млн" if "32x" in r["source"] else "8×0,5 млн"
        rows_p.append([k, mech, str(c["n"]), num(c["rho"], 2), num(c["cv"], 1), num(c["delta"], 2), method, series])
    tp = table("tab:parameters", [0.9, 5.2, 0.7, 1.2, 1.0, 1.2, 2.5, 2.2],
               [r"№ & Механизм & $n$ & $\rho$ & $\mathrm{CV}$ & $\delta$ & Расчёт & Серия ИМ \\"], rows_p,
               r"Дополнительные условия, метод расчёта (ТТ --- Такахаси--Таками, СФМ --- совместная фазовая модель) "
               r"и серия ИМ. Номер сохраняется во всех следующих таблицах.")
    tm = table("tab:studymeans", WE, [h.replace("Вариант &", "№ &") for h in HEAD7], rows_m, r"Средние в дополнительных условиях.", header_rows=2)
    ts = table("tab:studystd", WE, [h.replace("Вариант &", "№ &") for h in HEAD7S], rows_s, r"Стандартные отклонения в дополнительных условиях.", header_rows=2)
    return tp, tm, ts


def main():
    text = V8S.read_text()
    head = text[: text.index(r"\section{Назначение и исходные данные}")]
    t_tt, t_law = tt_tables()
    t_cor, t_j, t_old = correction_tables()
    t_par, t_mean, t_std = extended_tables()
    grids, sens, cost, ph, erl = (copy_v8(x) for x in ("tab:grids", "tab:sensitivity", "tab:cost", "tab:ph", "tab:erlang"))
    order = [t_tt, t_law, grids, t_cor, t_j, t_old, sens, t_par, t_mean, t_std, cost, ph, erl]
    ids = ["tab:ttcheck", "tab:lawfull", "tab:grids", "tab:correction", "tab:jseq", "tab:oldsim", "tab:sensitivity",
           "tab:parameters", "tab:studymeans", "tab:studystd", "tab:cost", "tab:ph", "tab:erlang"]
    manifest = [next(m for m in MANIFEST if m["id"] == i) for i in ids]
    longmax = SUM["extended_max_abs_percent"]
    body = rf"""\section{{Назначение и исходные данные}}
Приложение сопровождает версию 9 статьи «Расчёт многоканальных СМО с отрицательными заявками и повторным обслуживанием». Оно содержит сверку метода Такахаси--Таками, погрешность двухфазного закона во всех режимах, поправку квадратуры для сохранённой работы, прежние серии имитационного моделирования и расширенный эксперимент. Везде ДИ означает полуширину 95\%-го доверительного интервала по распределению Стьюдента, а \(\Delta=100(\text{{расчёт}}-\text{{ИМ}})/\text{{ИМ}}\).

\section{{Метод Такахаси--Таками}}
Стационарные вероятности двухфазной модели рассчитывались итерациями Такахаси--Таками в комплексной арифметике (библиотека Most-Queue, допуск \(10^{{-12}}\)), моменты --- по уравнениям меченой заявки. Те же уравнения решены прямым разреженным методом (таблица~\ref{{tab:ttcheck}}). При \(\mathrm{{CV}}=0{{,}}8\) параметры \(H_2\) вещественны, но один вес отрицателен; при \(\mathrm{{CV}}=0{{,}}6\) и \(0{{,}}5\) параметры комплексно-сопряжённые. Мнимые части итоговых характеристик не превышают \(10^{{-7}}\).

{t_tt}

Библиотечный расчёт моментов численным дифференцированием ПЛС даёт то же среднее ожидание. Для пребывания при RCS он использует постоянную интенсивность \(\delta/m\), где \(m\) --- число занятых каналов в момент начала обслуживания; при \(\mathrm{{CV}}=1{{,}}2\) это завышает \(v_1\) с \(1{{,}}92203\) до \(1{{,}}92229\), при \(\mathrm{{CV}}=0{{,}}5\) --- с \(2{{,}}15436\) до \(2{{,}}15508\).

{t_law}

\section{{Сохранённая работа}}
В основных примерах \(J=4\), \(N=100\); для RCS \(H=24\). Экстраполяция по \(1/K\) при \(K=8\), 12 и 16 имеет веса 2, \(-9\) и 8 и применяется к начальным моментам; в расширенном эксперименте \(J=3\), \(K=4\), 8 и 12 с весами \(1/2\), \(-4\) и \(9/2\). Таблица~\ref{{tab:grids}} содержит конечные сетки до поправки.

{grids}

Поправка квадратуры вычисляется по системе без воздействий на тех же сетках \(J,K\) и с тем же усечением \(N\) (таблица~\ref{{tab:correction}}). Эталон при \(\delta=0\) --- положительное фазовое представление гамма-распределения с 16 узлами или само \(H_2\). Поправка всегда положительна: квадратура с конечным \(J\) занижает время ожидания. Обоснованность поправки проверена по последовательности \(J=3,\ldots,6\) (таблица~\ref{{tab:jseq}}): значения без поправки сходятся к пределу снизу, значения с поправкой практически от \(J\) не зависят.

{t_cor}

{t_j}

Основная таблица статьи использует серию из 32 прогонов длиной \(2\cdot10^6\) уходов. Прежняя серия приведена в таблице~\ref{{tab:oldsim}}; её интервалы примерно в четыре раза шире.

{t_old}

Сопоставление более грубой тройки фаз, числа классов и другого аргумента экстраполяции относится к результату без поправки:

{sens}

\section{{Расширенный эксперимент}}
В каждом условии \(b_1=2{{,}}1\), \(\lambda=\rho n/b_1\). Исходный закон при \(\mathrm{{CV}}=0{{,}}8\) --- гамма-распределение, при \(\mathrm{{CV}}=1{{,}}2\) --- положительный \(H_2\). Удаление и очищение рассчитаны методом Такахаси--Таками, повторы --- совместной фазовой моделью (9 положительных фаз при \(\mathrm{{CV}}=0{{,}}8\)), сохранённая работа --- с поправкой квадратуры. В условиях 1--7 выполнено по 8 прогонов с исключением первых 50\,000 и наблюдением 500\,000 уходов. Для условий 8--12 добавлены серии из 32 прогонов длиной \(2\cdot10^6\) уходов: в прежней серии отдельные оценки этих условий выходили за интервалы. Прежние результаты условий 8--12 сохранены в исходных данных. Все средние и стандартные отклонения находятся внутри соответствующих интервалов; максимальные абсолютные расхождения средних ожидания и пребывания равны {num(longmax['w'], 2)}\% и {num(longmax['v'], 2)}\%, стандартных отклонений --- {num(longmax['sw'], 2)}\% и {num(longmax['sv'], 2)}\%.

{t_par}

{t_mean}

{t_std}

{cost}

Память указана по максимальному объёму оперативной памяти, занятой отдельным процессом (RSS); время --- по полному времени выполнения расчёта при одновременной работе нескольких процессов. Прочерк означает, что отдельное увеличение предела для данного условия не выполнялось.

\section{{Положительные фазовые распределения}}
Для гамма-распределения при \(\mathrm{{CV}}=0{{,}}8\) использованы квадратуры с \(J=2\), 4 и 8, соответствующие 3, 5 и 9 фазам. На контрольной сетке аргумента максимальное отклонение функции распределения от исходного гамма-распределения составляет соответственно \(6{{,}}33\cdot10^{{-3}}\), \(8{{,}}08\cdot10^{{-4}}\) и \(9{{,}}54\cdot10^{{-5}}\). Это дискретная проверка, а не доказанная равномерная граница.

{ph}

{erl}

Двухфазный закон по трём моментам при \(\mathrm{{CV}}=0{{,}}8\) имеет вещественные параметры с отрицательным весом. Его ПЛС совпадает с ПЛС положительного распределения Кокса
\[
\boldsymbol\alpha=(1,0),\qquad
S=\begin{{pmatrix}}-0{{,}}674119662&0{{,}}511518043\\0&-1{{,}}230642243\end{{pmatrix}},
\]
с вероятностью перехода во вторую фазу 0{{,}}758794120. Поэтому при \(\mathrm{{CV}}=0{{,}}8\) расчёт методом Такахаси--Таками относится к вероятностному закону, а не только к алгебраическому приближению.

\section{{Воспроизводимость и границы проверки}}
Исходные JSON-файлы содержат неокруглённые начальные моменты, параметры, отсечения, время и память, файлы прогонов --- начальные состояния генераторов случайных чисел. Длинные серии версии 9 используют базы 2026110810000--2026110816000 с шагом 1000 по вариантам и 8000, 9000 для условий 9 и 8 соответственно; номер прогона прибавляется к базе. Имитатор проверен при \(\delta=0\) на точном решении \(M/G/2\) с гамма-распределением.

Набор из 75 автоматических тестов охватывает известные частные случаи, сверку метода Такахаси--Таками с прямым решением и независимые реализации. Он не устанавливает равномерную точность для всех нагрузок и законов обслуживания. Разности сеток и поправка квадратуры являются численной диагностикой, а не доказанной оценкой погрешности.
"""
    (ROOT / "supplement/negative_queues_takahasi_takami.tex").write_text(head + body + "\n\\end{document}\n")
    (ROOT / "supplement/table-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    refs = {i: [f"tab_{k + 1}", str(k + 1), ""] for k, i in enumerate(ids)}
    refs["fig:convergence"] = ["fig_1", "1", ""]
    (ROOT / "supplement/references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2) + "\n")
    print("supplement tables:", len(manifest), "cells:", sum(len(r) - 1 for m in manifest for r in m["expected_rows"]))


if __name__ == "__main__":
    main()
