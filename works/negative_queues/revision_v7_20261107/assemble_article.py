"""Create the v7 manuscript and supplement from immutable calculation outputs."""

import copy
import json
import re
import shutil
from pathlib import Path
import numpy as np
from works.negative_queues.revision_v7_20261107.summarize_step2 import point

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "refinement_20261107"
SUP = ROOT / "supplement"


def read(path):
    return json.loads(path.read_text())


def number(x, digits=4):
    return f"{x:.{digits}f}".replace(".", ",")


def value(row, key):
    return row[key][0] if key in ["w", "v"] else row[key]


def title(mode, action):
    if action == "remove":
        return "RCS, удаление" if mode == "rcs" else "Очищение"
    return (
        ("RCS" if mode == "rcs" else "Все каналы")
        + ", "
        + ("новая длительность" if action == "restart" else "сохранённая длительность")
    )


def compare(calc, sim):
    return {
        k: dict(
            calculation=value(calc, k),
            simulation=sim[k]["mean"],
            half_ci95=sim[k]["half_ci95"],
            difference_percent=100 * (value(calc, k) / sim[k]["mean"] - 1),
            inside_ci95=bool(abs(value(calc, k) - sim[k]["mean"]) <= sim[k]["half_ci95"] + 1e-12),
        )
        for k in ["w", "v", "sw", "sv", "q"]
    }


def metric_cells(comparison, keys):
    result = []
    for k in keys:
        row = comparison[k]
        result.extend(
            [
                number(row["calculation"]),
                number(row["simulation"]) + " ± " + number(row["half_ci95"]),
                number(row["difference_percent"], 2),
            ]
        )
    return result


def table(caption, label, headers, rows, widths, manifest, header_rows=1, grouped=False):
    n = len(widths)
    spec = "@{}" + "".join("p{" + str(round(w * 0.945, 2)) + "cm}" for w in widths) + "@{}"
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering\\small\\setlength{\\tabcolsep}{2pt}",
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
    ]
    lines += [" & ".join(h) + r" \\" for h in headers]
    lines.append(r"\midrule")
    expected = []
    for row in rows:
        if isinstance(row, str):
            lines.append(r"\multicolumn{" + str(n) + r"}{c}{" + row + r"} \\")
        else:
            lines.append(" & ".join(row) + r" \\")
            expected.append(row)
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    )
    manifest.append(dict(id=label, widths_cm=widths, header_rows=header_rows, expected_rows=expected, font_pt=10))
    return "\n".join(lines) + "\n\n"


def seven(caption, label, rows, keys, manifest, bycv=False):
    symbol = {"w": r"$w_1$", "v": r"$v_1$", "sw": r"$\sigma_W$", "sv": r"$\sigma_V$"}
    heads = [
        ["", r"\multicolumn{3}{c}{" + symbol[keys[0]] + "}", r"\multicolumn{3}{c}{" + symbol[keys[1]] + "}"],
        ["Вариант", "Расчёт", r"ИМ $\pm$ ДИ", r"$\Delta$, \%", "Расчёт", r"ИМ $\pm$ ДИ", r"$\Delta$, \%"],
    ]
    out = []
    previous = None
    for row in rows:
        if bycv and row["cv"] != previous:
            out.append("$CV=" + number(row["cv"], 1) + "$")
            previous = row["cv"]
        out.append([row["label"]] + metric_cells(row["comparison"], keys))
    return table(caption, label, heads, out, [4.4, 1.7, 3, 1.35, 1.7, 3, 1.35], manifest, header_rows=2)


def main(main_only=False):
    SUP.mkdir(exist_ok=True)
    baseline = read(OLD / "experiment-results.json")
    refinement = read(ROOT / "step2-results.json")["results"]
    ph = read(ROOT / "step4-results.json")
    data = []
    for old in baseline["results"]:
        row = copy.deepcopy(old)
        cv, mode, action = (row[k] for k in ["cv", "mode", "scenario"])
        if action == "fixed":
            row["calculation"] = next(r["calculation"] for r in refinement if r["cv"] == cv and r["mode"] == mode)
        else:
            row["calculation"] = read(ROOT / "step4" / f"base-{action}-{mode}-cv{cv}-j{8 if cv<1 else 0}-n240-b80.json")
        row["comparison"] = compare(row["calculation"], row["simulation"])
        row["label"] = title(mode, action)
        row.pop("truncation_change", None)
        row["method"] = (
            "joint retained-work PH; raw-moment extrapolation J4, K8/12/16"
            if action == "fixed"
            else (
                "full tagged PH equations; positive Gamma PH with 9 phases"
                if cv < 1
                else "full tagged PH equations; original positive H2 law"
            )
        )
        data.append(row)
    data.sort(
        key=lambda r: (
            r["cv"],
            ["remove", "restart", "fixed"].index(r["scenario"]),
            ["rcs", "disaster"].index(r["mode"]),
        )
    )
    study = [] if main_only else read(ROOT / "step3-results.json")
    for row in study:
        new = next((p for p in ph["extended"] if p["case"]["id"] == row["case"]["id"]), None)
        if new:
            row["h2_comparison"] = row["comparison"]
            row["comparison"] = new["comparison"]
            row["calculation"] = new["calculation"]
        row["label"] = str(int(row["case"]["id"][-2:]) + 1)
    payload = dict(
        **{k: v for k, v in baseline.items() if k != "results"},
        results=data,
        extended=study,
        retained_work_validation=refinement,
        positive_ph=ph,
        version="v7",
        extended_pending=main_only,
        original_simulation_unchanged=True,
        original_simulation_file=str(OLD / "experiment-results.json"),
        new_simulation_files=[
            str(ROOT / name)
            for name in ["step1-replications.jsonl", "step3-replications.jsonl", "step4-replications.jsonl"]
        ],
    )
    (ROOT / "experiment-results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "generated_tables_fixed.tex").write_text("")
    (SUP / "generated_tables_fixed.tex").write_text("")
    shutil.copy2(ROOT / "citations.bib", SUP / "citations.bib")
    (ROOT / "figures").mkdir(exist_ok=True)
    for path in (OLD / "figures").glob("figure1*"):
        shutil.copy2(path, ROOT / "figures" / path.name)
    (SUP / "figures").mkdir(exist_ok=True)
    for path in (ROOT / "figures").glob("convergence.*"):
        shutil.copy2(path, SUP / "figures" / path.name)
    main_manifest = [dict(id="tab:methods", widths_cm=[3.9, 6.5, 6.1], header_rows=1, expected_rows=[], font_pt=10)]
    table1 = seven(
        r"Средние при $\delta=0{,}3$. ДИ --- полуширина приближённого95\%-го доверительного интервала по восьми исходным прогонам. Все каналы означает одновременный повтор на занятых каналах.",
        "tab:results",
        [r for r in data if r["scenario"] != "fixed"],
        ["w", "v"],
        main_manifest,
        True,
    )
    table2 = seven(
        r"Средние при повторе с сохранённой длительностью, $\delta=0{,}02$. После прерывания заново выполняется первоначально выбранная работа; используются те же исходные имитационные оценки, что и в предыдущей версии.",
        "tab:results_fixed",
        [r for r in data if r["scenario"] == "fixed"],
        ["w", "v"],
        main_manifest,
        True,
    )
    main_manifest[1]["widths_cm"] = [4.4, 1.7, 3, 1.35, 1.7, 3, 1.35]
    maxw = max(abs(r["comparison"]["w"]["difference_percent"]) for r in data)
    maxv = max(abs(r["comparison"]["v"]["difference_percent"]) for r in data)
    misses = [
        (r["label"], r["cv"], k) for r in data for k in ["w", "v", "sw", "sv"] if not r["comparison"][k]["inside_ci95"]
    ]
    stage_misses = [
        (r["case"]["id"], k) for r in study for k in ["w", "v", "sw", "sv"] if not r["comparison"][k]["inside_ci95"]
    ]
    (ROOT / "coverage-summary.json").write_text(
        json.dumps(dict(main_misses=misses, extended_misses=stage_misses), ensure_ascii=False, indent=2) + "\n"
    )
    results = r"""\section{Численная проверка}\label{sec:results}
\subsection{Основные примеры и протокол}
В основных примерах \(n=3\), \(\lambda=1\), \(b_1=2{,}1\), \(\rho=0{,}7\). При удалении и новой длительности \(\delta=0{,}3\); при сохранённой длительности \(\delta=0{,}02\). Рассматриваются \(CV=0{,}8\) и \(CV=1{,}2\). В первом случае исходный закон --- гамма-распределение; во втором --- положительное \(H_2\), согласованное с первыми тремя моментами гамма-распределения. Таким образом, сравнение при \(CV=1{,}2\) не относится к исходному гамма-закону.

Таблицы~\ref{tab:results} и~\ref{tab:results_fixed} сохраняют исходные имитационные оценки: по восемь независимых прогонов на каждую из12 комбинаций, прогрев20\,000 уходов и наблюдение не менее200\,000 уходов, включая удаления. Полуширины95\%-х интервалов получены по распределению Стьюдента с семью степенями свободы. Относительное расхождение определяется как \(\Delta=100(\text{расчёт}-\text{ИМ})/\text{ИМ}\); оно не является известной точной погрешностью относительно стационарного значения. Стандартные отклонения и дополнительные результаты приведены в сопроводительном приложении.

Для Gamma при \(CV=0{,}8\) в вариантах удаления и новой длительности использована положительная модель с9 фазами; проверены также3 и5 фаз. Увеличение числа фаз с5 до9 изменило среднее ожидание в этих четырёх базовых примерах не более чем на0{,}00058\%. Проверены пределы \((N,H)=(160,40)\) и \((240,80)\). Эти сравнения контролируют численную чувствительность, включая точность решения линейных систем.

При сохранённой работе основной результат получен при \(J=4\) экстраполяцией начальных моментов по \(1/K\) на сетках \(K=8,12,16\). Веса равны \(2,-9,8\). Отдельно сопоставлены \(J=3\), более грубая тройка \(K=4,8,12\), экстраполяция по \(1/\sqrt K\) и увеличение пределов очереди. Для \(CV=0{,}8\) добавлена проверка \(J=5\) на умеренных сетках. Все исходные конечные сетки сохранены; ни число фаз, ни веса не выбирались по близости к имитации.

"""
    results += table1 + table2
    results += f"Наибольшее абсолютное расхождение средних в основных таблицах составляет {number(maxw,2)}\\% для ожидания и {number(maxv,2)}\\% для пребывания. "
    results += (
        "Все приведённые средние и стандартные отклонения находятся внутри соответствующих исходных интервалов. "
        if not misses
        else f"Число выходов средних и стандартных отклонений за исходные интервалы равно {len(misses)}; они сохранены в таблицах. "
    )
    results += r"""Попадание в отдельный доверительный интервал не доказывает точность метода и не обеспечивает одновременного покрытия всех характеристик.

\subsection{Независимые проверки и причины расхождений}
Для RCS с сохранённой работой выполнены ещё16 прогонов независимо написанного событийного имитатора: по восемь на каждое значение \(CV\), с прогревом50\,000 и наблюдением500\,000 уходов. Для общих прерываний используются отдельно сохранённые независимые прогоны с тем же объёмом наблюдения. При RCS и \(CV=0{,}8\) уточнённые оценки \(w_1=1{,}03729\) и \(v_1=3{,}16805\) лежат ниже нижних границ дополнительных95\%-х интервалов примерно на0{,}00022 и0{,}00009 соответственно. Увеличение числа классов с4 до5 на сетке \(K=8\) меняет среднее примерно на0{,}04\%. Оба выхода за интервалы сохранены; дополнительная квадратура не использовалась для выбора более близкого к имитации числа.

Общие прерывания требуют совместного описания каналов. В сохранённом независимом эксперименте при \(CV=1{,}2\) среднее ожидание составляет \(2{,}229\pm0{,}050\) для общих прерываний и \(2{,}049\pm0{,}034\) для независимых эффективных обслуживаний с тем же законом времени завершения одной заявки. Следовательно, уточнение только индивидуальных моментов не воспроизводит зависимость каналов. В вариантах RCS полные уравнения дополнительно сохраняют изменение риска во время обслуживания и связь ожидания с последующим пребыванием.

Расширенный эксперимент содержит12 новых сочетаний параметров: \(n=2,3,4\), номинальную нагрузку от0{,}4 до1{,}2 и разные интенсивности воздействий. На каждое сочетание приходится восемь независимых прогонов с прогревом50\,000 и наблюдением500\,000 уходов. Все прогоны сохранены. Для повторов с сохранённой работой использованы \(J=3\), \(K=4,8,12\), а в более нагруженных случаях дополнительно увеличен предел очереди. Условия, результаты, изменения при уточнении, время и пиковая память приведены в сопроводительном приложении. Времена измерены при совместном использовании вычислительной машины; это характеристика выполненных расчётов, а не сравнительный тест аппаратуры.

Положительная9-фазная аппроксимация устранила два выхода за интервалы в дополнительных Gamma-примерах, наблюдавшихся при прежнем двухфазном приближении. Для точного Erlang-4, соответствующего \(CV=0{,}5\), проведены ещё16 независимых прогонов; средние и стандартные отклонения в обоих вариантах новой длительности согласуются с интервалами. При общем повторе исходного \(H_2\), \(n=3\), \(\rho=0{,}85\), \(\delta=0{,}6\), сохранились выходы среднего ожидания, среднего пребывания и стандартного отклонения ожидания за отдельные интервалы. Расчёт этого случая совпал с независимо реализованным двухфазным решателем; уточнение границ не устранило расхождение. Такой результат совместим со статистической неопределённостью отдельных интервалов, но сам по себе не устанавливает её как единственную причину.

При RCS с сохранённой работой, \(n=2\), \(\rho=0{,}5\), \(CV=0{,}8\), \(\delta=0{,}04\), оба средних также ниже отдельных интервалов: расхождения составляют \(-1{,}09\%\) и \(-0{,}35\%\). Дополнительные сетки \(J=4,5\), \(K=4,8,12,16\) выявили вклад квадратуры: увеличение \(J\) с3 до5 меняет экстраполированное среднее ожидание примерно на0{,}23\%, но оба выхода сохраняются. Основной результат \(J=3\) оставлен в таблицах. Всего в расширенном эксперименте пять из48 средних и стандартных отклонений лежат вне соответствующих интервалов. Максимальные абсолютные расхождения средних ожидания и пребывания составляют1{,}36\% и0{,}68\%; для стандартных отклонений максимум равен2{,}24\%. Уточнение сеток ограничивает возможный вклад отдельных численных ошибок, но не даёт полного объяснения оставшегося расхождения.

Численные проверки включают закон Литтла, уравнения баланса, вырождение в известные одноканальные и экспоненциальные модели, эквивалентность независимых фазовых реализаций и контроль отсечений. Набор из49 автоматических проверок пройден. Тесты не заменяют исследования сходимости для иных законов обслуживания, тяжёлых хвостов и режимов, близких к границе устойчивости.

\section{Заключение}
Построен единый фазовый расчёт многоканальных систем с выборочным удалением, очищением и двумя вариантами повторного обслуживания. Уравнения помеченной заявки учитывают текущее число занятых каналов и общий момент прерывания. Для сохранённой работы построена совместная модель с непрерывными координатами и её конечная фазовая аппроксимация. Положительное представление гамма-распределения позволяет уточнять обслуживание при \(CV<1\), сохраняя вероятностный смысл переходов.

Работоспособность расчёта проверена на указанных параметрах, независимых реализациях и известных предельных случаях. Раздельный контроль выявляет вклад аппроксимации обслуживания, фазовой дискретизации, усечений и неопределённости имитации. Разности сеток остаются диагностикой; строгая общая оценка ошибки стационарных моментов здесь не устанавливается. Рост числа фаз и каналов существенно увеличивает пространство состояний.

Программные материалы, параметры и исходные прогоны приложены к рабочей версии статьи. Библиотека Most-Queue \cite{mostqueue2026} используется для вспомогательных операций с распределениями и для сопоставления с прежними расчётами; новые исследовательские программы включены в сопровождающий комплект.
"""
    oldtext = (OLD / "article_body.tex").read_text()
    clearing = oldtext.split(r"\subsection{Мгновенное очищение системы}", 1)[1].split(
        r"\section{Расчёт временных характеристик}", 1
    )[0]
    clearing = clearing.replace("Для алгебраической аппроксимации эта операция применяется к алгебраическим весам.", "")
    appendix = (
        r"""\clearpage
\section*{Приложение. Двухфазная схема очищения}
Для положительного \(H_2\) фаза \(\ell\in\{1,2\}\) выбирается с вероятностью \(y_\ell\) и имеет интенсивность завершения \(\mu_\ell\). Состояние \((i,j)\) означает \(i\) заявок в системе и \(j\) занятых каналов в фазе2; остальные занятые каналы находятся в фазе1. Блок \(B_i\) задаёт переходы на предыдущий уровень, \(C_i\) --- внутренние переходы, \(D_i\) --- диагональ суммарных интенсивностей выхода. При \(m=\min(i,n)\) завершения в фазах1 и2 имеют суммарные интенсивности \((m-j)\mu_1\) и \(j\mu_2\). Если очередь непуста, после завершения новая заявка сразу выбирает фазу по \((y_1,y_2)\). Эти переходы определяют базовый блок обслуживания; до введения катастроф базовая диагональ равна
\[
D_i^{\mathrm{base}}(j,j)=\lambda+(m-j)\mu_1+j\mu_2.
\]
Для RCS отрицательное воздействие равновероятно освобождает один из занятых каналов. Схемы на рисунке~\ref{fig:negative_transitions} показывают только отрицательные воздействия.

"""
        + clearing
    )
    # The appendix has the only two figure panels, retaining their large original artwork.
    core = (ROOT / "article_core_draft.tex").read_text().replace("со шкалой", "с масштабом")
    full_prefix = (OLD / "negative_queues_takahasi_takami.tex").read_text().split(r"\begin{abstract}", 1)[0]
    text = full_prefix + core + "\n" + results + "\n" + appendix + "\n\\printbibliography\n\\end{document}\n"
    text = re.sub(r"(?<=[А-Яа-яЁё])(?=\d)", " ", text)
    (ROOT / "negative_queues_takahasi_takami.tex").write_text(text)
    (ROOT / "article_body.tex").write_text(core + "\n" + results + "\n" + appendix)
    (ROOT / "table-manifest.json").write_text(json.dumps(main_manifest, ensure_ascii=False, indent=2) + "\n")
    refs = {
        "tab:methods": ["tab_1", "1", ""],
        "tab:results": ["tab_2", "2", ""],
        "tab:results_fixed": ["tab_3", "3", ""],
        "fig:rcs_transitions": ["fig_1", "1", "а"],
        "fig:disaster_transitions": ["fig_1", "1", "б"],
        "fig:negative_transitions": ["fig_1", "1", ""],
        "sec:times": ["sec_4", "4", ""],
        "sec:results": ["sec_5", "5", ""],
    }
    (ROOT / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2) + "\n")
    if not main_only:
        make_supplement(data, study, refinement, ph)


def make_supplement(data, study, refinement, ph):
    manifest = []
    body = r"""\section{Назначение и исходные данные}
Приложение сопровождает версию7 статьи «Расчёт многоканальных СМО с отрицательными заявками». Оно содержит дополнительные характеристики, сетки и параметры проверки. Основные таблицы статьи сохраняют прежние96 имитационных оценок; новые независимые серии анализируются отдельно. Везде ДИ означает полуширину приближённого95\%-го интервала, а \(\Delta=100(\text{расчёт}-\text{ИМ})/\text{ИМ}\). Все числа сформированы из сохранённых JSON-результатов, без ручного округления промежуточных моментов.

"""
    body += seven(
        "Стандартные отклонения основных12 примеров. Имитация совпадает с основной статьёй.",
        "tab:std",
        data,
        ["sw", "sv"],
        manifest,
        True,
    )
    body += r"""\section{Уточнение сохранённой работы}
В основных примерах \(J=4\), \(N=100\); для RCS \(H=24\). Экстраполяция \(K=8,12,16\) по \(1/K\) имеет веса \(2,-9,8\) и применяется к начальным моментам. Таблица конечных сеток позволяет отделить сами расчёты от экстраполированного результата. Малые отличия при увеличении \(N,H\) включают остаточную ошибку линейного решателя. Увеличение числа состояний само по себе не является доказательством сходимости стационарных моментов.

"""
    rows = []
    for mode in ["rcs", "disaster"]:
        for cv in [0.8, 1.2]:
            for k in [4, 8, 12, 16]:
                r = point(mode, cv, 4, k)
                rows.append(
                    [
                        ("RCS" if mode == "rcs" else "Все") + ", " + number(cv, 1),
                        str(k),
                        number(r["w"][0], 6),
                        number(r["w"][1], 6),
                        number(r["seconds"], 1),
                    ]
                )
    body += table(
        r"Конечные сетки: вариант и \(CV\), число фаз на класс, начальные моменты ожидания, время расчёта в секундах.",
        "tab:grids",
        [["Вариант, $CV$", "$K$", "$w_1$", "$w_2$", "Время, с"]],
        rows,
        [4.7, 1.2, 3.5, 3.5, 2.6],
        manifest,
    )
    rows = [
        dict(
            label=("RCS" if r["mode"] == "rcs" else "Все каналы") + ", CV=" + number(r["cv"], 1),
            comparison={
                k: dict(
                    calculation=value(r["calculation"], k),
                    simulation=v["mean"],
                    half_ci95=v["half_ci95"],
                    difference_percent=v["difference_percent"],
                    inside_ci95=v["inside_ci95"],
                )
                for k, v in r["comparison"].items()
            },
        )
        for r in refinement
    ]
    body += seven(
        "Экстраполяция и отдельная независимая имитация сохранённой работы.", "tab:extrap", rows, ["w", "v"], manifest
    )
    sensitivity = read(ROOT / "step2-results.json")
    assert not sensitivity["pending_checks"]
    body += r"""При увеличении стационарного предела общих прерываний с \(N=100\) до160 на сетке \(CV=1{,}2,J=4,K=16\) среднее ожидание меняется на \(6{,}00\cdot10^{-6}\), второй начальный момент --- на \(6{,}57\cdot10^{-4}\) (около0{,}0032\%). Для RCS увеличение \(N\) или \(H\) на контрольной сетке меняет первые два момента менее чем на \(1{,}3\cdot10^{-6}\). Это изменения конечных сеток; веса экстраполяции могут усиливать остаточную ошибку. Результаты обеих точностей линейного решателя сохранены отдельно.

"""
    body += "Сопоставление более грубой тройки фаз, числа классов и другого аргумента экстраполяции:\n\n"
    rows = []
    for r in refinement:
        z = r["sensitivity_percent"]
        rows.append(
            [
                ("RCS" if r["mode"] == "rcs" else "Все") + ", " + number(r["cv"], 1),
                number(z["w"]["coarser"], 4),
                number(z["w"]["fewer_classes"], 4),
                number(z["w"]["sqrt_extrapolation"], 4),
                number(z["sw"]["sqrt_extrapolation"], 4),
            ]
        )
    body += table(
        r"Относительные изменения, \%, относительно основной экстраполяции. Они не являются строгими границами ошибки.",
        "tab:sensitivity",
        [["Вариант, $CV$", "$w_1$: грубее", "$w_1$: $J=3$", r"$w_1$: $1/\sqrt K$", r"$\sigma_W$: $1/\sqrt K$"]],
        rows,
        [4.7, 2.95, 2.95, 2.95, 2.95],
        manifest,
    )
    body += r"""\begin{figure}[!htbp]
\centering
\begin{subfigure}{0.98\linewidth}
\centering
\maybeincludegraphics[width=0.98\linewidth]{figures/convergence.png}
\caption{Конечные сетки и экстраполяции.}
\end{subfigure}
\caption{Сходимость среднего ожидания при сохранённой работе. Полосы показывают интервалы отдельной независимой имитации. Коэффициенты экстраполяции не используют имитационные значения.}
\label{fig:convergence}
\end{figure}
"""
    body += r"""\section{Расширенный эксперимент}
В каждом условии \(b_1=2{,}1\), \(\lambda=\rho n/b_1\). Исходный закон при \(CV=0{,}8\) --- Gamma, при \(CV=1{,}2\) --- положительный \(H_2\), согласованный с первыми тремя Gamma-моментами. Для каждого условия выполнены8 прогонов с прогревом50\,000 и наблюдением500\,000 уходов. При сохранённой работе основная сетка \(J=3\), \(K=4,8,12\); для двух каналов \(N=120,H=32\), для трёх \(N=240,H=40\). Для общих прерываний свёртка с временем собственной работы устраняет необходимость в \(H\). Средняя сетка более нагруженных случаев дополнительно проверена при \(N=400\).

"""
    params = []
    for row in study:
        c = row["case"]
        params.append(
            [
                row["label"],
                title(c["mode"], c["action"]),
                str(c["n"]),
                number(c["rho"], 2),
                number(c["cv"], 1),
                number(c["delta"], 2),
            ]
        )
    body += table(
        "Дополнительные условия. Номер сохраняется во всех следующих таблицах.",
        "tab:parameters",
        [["№", "Механизм", "$n$", r"$\rho$", "$CV$", r"$\delta$"]],
        params,
        [0.8, 8.1, 1.1, 2, 2, 2],
        manifest,
    )
    body += seven(
        "Средние в дополнительных условиях. Для Gamma в вариантах удаления и новой длительности использовано9 фаз.",
        "tab:studymeans",
        study,
        ["w", "v"],
        manifest,
    )
    body += seven(
        "Стандартные отклонения в дополнительных условиях. Выходы за интервалы не исключались из результатов.",
        "tab:studystd",
        study,
        ["sw", "sv"],
        manifest,
    )
    rows = []
    for row in study:
        rows.append(
            [
                row["label"],
                number(row["total_calculation_seconds"], 1),
                number(row["peak_rss_kib"] / 1024, 1),
                (
                    number(max(abs(x) for v in row["truncation_changes"].values() for x in v), 6)
                    if row["truncation_changes"]
                    else "—"
                ),
            ]
        )
    body += table(
        "Стоимость исходных сеток расширенного эксперимента и максимальное абсолютное изменение начальных моментов при проверке отсечений. Для Gamma время положительного PH-уточнения сохранено отдельно в исходных данных.",
        "tab:cost",
        [["№", "Время сеток, с", "Пиковая память, МиБ", "Изменение при уточнении"]],
        rows,
        [1, 4.5, 5, 6],
        manifest,
    )
    body += r"""Память указана по пиковому RSS отдельного процесса; время --- по настенным часам, при одновременной работе нескольких процессов. Прочерк означает, что отдельное увеличение предела для данного условия не выполнялось; нулевое значение после округления означает малое измеренное изменение. Выбор прямого или итерационного решателя зависит от размера системы, поэтому время не обязано монотонно расти с пределом очереди. Проверка половин прогонов и закона Литтла сохранена для каждого условия. Отдельный95\%-й интервал не гарантирует одновременного покрытия всех строк и характеристик.

В условии8 средние ожидания и пребывания, а также стандартное отклонение ожидания находятся вне интервалов; независимый двухфазный решатель воспроизводит фазовый расчёт. В условии9 вне интервалов находятся оба средних. Следующая дополнительная проверка условия9 выполнена после обнаружения расхождения; исходная строка расширенного эксперимента сохранена. Все приведённые ниже средние остаются ниже соответствующих интервалов: \(0{,}71427\pm0{,}00520\) и \(2{,}91397\pm0{,}00629\).

"""
    extra = read(ROOT / "case08-refinement-results.json")["results"]
    rows = [
        [
            str(r["classes"]),
            ", ".join(map(str, r["stages"])),
            number(r["calculation"]["w"][0], 6),
            number(r["calculation"]["v"][0], 6),
        ]
        for r in extra
    ]
    body += table(
        r"Дополнительная чувствительность условия9 по числу классов и фаз. Экстраполируются начальные моменты по \(1/K\).",
        "tab:case09",
        [["$J$", "$K$", "$w_1$", "$v_1$"]],
        rows,
        [1.5, 5, 5, 5],
        manifest,
    )
    body += r"""

\section{Положительные фазовые распределения}
Для Gamma при \(CV=0{,}8\) использованы квадратуры с \(J=2,4,8\), соответствующие3, 5 и9 фазам. На контрольной сетке аргумента максимальное отклонение функции распределения от исходной Gamma составляет соответственно0{,}00633; 0{,}000808 и0{,}0000954. Это дискретная проверка, не доказанная равномерная граница. Первые \(2J-1\) моментов сохраняются по точности квадратуры.

"""
    rows = []
    for row in ph["baseline"]:
        rows.append(
            [title(row["mode"], row["action"])]
            + [number(row["estimates"][str(j)]["w"], 6) for j in [2, 4, 8]]
            + [number(row["j4_to_j8_percent"]["w"], 6)]
        )
    body += table(
        r"Среднее ожидание при3, 5 и9 положительных фазах и изменение между5 и9 фазами, \%.",
        "tab:ph",
        [["Механизм", "3 фазы", "5 фаз", "9 фаз", r"$\Delta$, \%"]],
        rows,
        [6.1, 2.7, 2.7, 2.7, 2.3],
        manifest,
    )
    rows = [
        dict(label="RCS" if r["case"]["mode"] == "rcs" else "Все каналы", comparison=r["comparison"])
        for r in ph["exact_erlang"]
    ]
    body += seven(
        r"Точный Erlang-4: \(CV=0{,}5\), \(n=3\), \(\lambda=0{,}6\), \(\delta=0{,}3\), новая длительность. Отдельные8 прогонов на вариант.",
        "tab:erlang",
        rows,
        ["w", "v"],
        manifest,
    )
    body += r"""Для прежнего рационального двухфазного ПЛС при \(CV=0{,}8\) найдено положительное представление Кокса:
\[
\boldsymbol\alpha=(1,0),\qquad
S=\begin{pmatrix}-0{,}674119662&0{,}511518043\\0&-1{,}230642243\end{pmatrix}.
\]
Вероятность перехода во вторую фазу равна0{,}758794120. Эквивалентность проверена по ПЛС и четырём вариантам очереди. Это объясняет допустимость самого прежнего рационального закона, но не превращает его в точную Gamma.

\section{Воспроизводимость и границы проверки}
Исходные JSON-файлы содержат неокруглённые начальные моменты, параметры, отсечения, время и память. Файлы отдельных прогонов содержат все начальные состояния генераторов случайных чисел. Для RCS с сохранённой работой используется база202611072000 с шагом100 между значениями \(CV\); для12 дополнительных условий --- 202611073000 с шагом100 по условию; для Erlang-4 --- 202611075000 с шагом100 по варианту. Внутри каждой серии8 прогонов используются смещения0--7. Прогоны не исключались по результату.

Уравнения и независимый событийный имитатор расположены в сопровождающем каталоге расчётов. Полный список команд приведён в README. Сохраняются обе фазовые реализации, результаты уточнений и прежняя версия статьи. Контроль49 тестами охватывает известные частные случаи и независимые реализации; он не устанавливает равномерную точность для всех нагрузок и законов обслуживания. Экстраполяции на сохранённой работе требуют отдельной оценки при иных параметрах.
"""
    preamble = (
        (OLD / "negative_queues_takahasi_takami.tex").read_text().split(r"\begin{document}", 1)[0]
        + r"\begin{document}"
        + "\n"
        + r"\begin{center}\Large\bfseries Дополнительные материалы к статье о СМО с отрицательными заявками\end{center}"
        + "\n"
    )
    text = preamble + body + "\n\\end{document}\n"
    text = re.sub(r"(?<=[А-Яа-яЁё])(?=\d)", " ", text)
    (SUP / "negative_queues_takahasi_takami.tex").write_text(text)
    (SUP / "table-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    refs = {row["id"]: [f"tab_{i+1}", str(i + 1), ""] for i, row in enumerate(manifest)}
    refs["fig:convergence"] = ["fig_1", "1", ""]
    (SUP / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--main-only", action="store_true")
    main(parser.parse_args().main_only)
