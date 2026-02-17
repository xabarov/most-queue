"""
Build LaTeX tables for the "repeat without resampling" (fixed service per job) discipline.

The main paper tables (tab:results, tab:sko) are shown for the base example with delta=0.3.
For fixed-service restarts, such delta values often violate moment existence / stability, so we
export a separate table at a smaller delta.

Run from repo root:
  ./.venv/bin/python works/negative_queues/export_fixed_service_tables.py --delta 0.05 --out works/negative_queues/generated_tables_fixed.tex
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from works.negative_queues.work_mgn_negatives import (
    DisasterScenario,
    NegativeServiceType,
    RcsScenario,
    collect_calc_results,
    collect_sim_results,
    gamma_moments_by_mean_and_cv,
    read_parameters_from_yaml,
)


def rel_err_signed(calc: float, sim: float) -> float:
    if sim == 0:
        return 0.0 if calc == 0 else float("inf")
    return 100.0 * (calc - sim) / sim


def std_from_moments(m: list[float]) -> float:
    if len(m) < 2:
        return 0.0
    var = float(m[1]) - float(m[0]) ** 2
    return float(np.sqrt(max(0.0, var)))


def fmt_num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def fmt_err(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.2f}"


def build_tables(*, delta: float, num_jobs: int, cvs: tuple[float, ...]) -> str:
    base_qp = read_parameters_from_yaml("works/negative_queues/base_parameters.yaml")
    qp = dict(base_qp)
    qp["arrival_rate"] = dict(base_qp["arrival_rate"])
    qp["arrival_rate"]["negative"] = float(delta)
    qp["num_of_jobs"] = int(num_jobs)

    # Only the two fixed-service restart scenarios.
    scenarios = [
        (
            "RCS (повтор без пересемпл.)",
            NegativeServiceType.RCS,
            True,
            True,
            RcsScenario.REQUEUE_NO_RESAMPLING,
            DisasterScenario.CLEAR_SYSTEM,
        ),
        (
            "Катастрофы (повтор без пересемпл.)",
            NegativeServiceType.DISASTER,
            True,
            True,
            RcsScenario.REMOVE,
            DisasterScenario.REQUEUE_ALL_NO_RESAMPLING,
        ),
    ]

    # Collect (calc, sim) per scenario per CV.
    rows: dict[float, list[tuple[str, dict, dict]]] = {cv: [] for cv in cvs}
    for cv in cvs:
        for label, discipline, requeue, repeat_wo_resampling, rcs_sc, dis_sc in scenarios:
            n = int(qp["channels"]["base"])
            service_mean = n * float(qp["utilization"]["base"]) / float(qp["arrival_rate"]["positive"])
            b = gamma_moments_by_mean_and_cv(service_mean, float(cv))

            calc = collect_calc_results(
                n=n,
                qp=qp,
                b=b,
                discipline=discipline,
                requeue_on_negative=requeue,
                repeat_without_resampling=repeat_wo_resampling,
            )
            sim = collect_sim_results(
                n=n,
                qp=qp,
                b=b,
                discipline=discipline,
                rcs_scenario=rcs_sc,
                disaster_scenario=dis_sc,
            )
            rows[cv].append((label, asdict(calc), asdict(sim)))

    # Build LaTeX tables (booktabs style) as a single fragment.
    lines: list[str] = []
    lines.append("% -----------------------------------------------------------------------------")
    lines.append("% Auto-generated tables for fixed-service restarts (repeat without resampling).")
    lines.append("% -----------------------------------------------------------------------------")

    # Table: w1, v1
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{$w_1$} & \multicolumn{3}{c}{$v_1$} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"Дисциплина & Числ & ИМ & Отн.\% & Числ & ИМ & Отн.\% \\")
    lines.append(r"\midrule")
    for i, cv in enumerate(cvs):
        cv_caption = (
            r"\multicolumn{7}{@{}l}{\textit{Коэффициент вариации} $CV=0{,}8$} \\"
            if i == 0
            else r"\multicolumn{7}{@{}l}{\textit{Коэффициент вариации} $CV=1{,}2$} \\"
        )
        lines.append(cv_caption)
        for label, c, s in rows[cv]:
            w_calc = float(c["w"][0])
            w_sim = float(s["w"][0])
            v_calc = float(c["v"][0])
            v_sim = float(s["v"][0])
            w_err = rel_err_signed(w_calc, w_sim)
            v_err = rel_err_signed(v_calc, v_sim)
            lines.append(
                " & ".join(
                    [
                        label,
                        fmt_num(w_calc),
                        fmt_num(w_sim),
                        fmt_err(w_err),
                        fmt_num(v_calc),
                        fmt_num(v_sim),
                        fmt_err(v_err),
                    ]
                )
                + r" \\"
            )
        if i < len(cvs) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{Первые начальные моменты времени ожидания \(w_1\) и пребывания \(v_1\) для дисциплины «повтор без пересемплирования» при фиксированных \(n=3\), \(\lambda_{{\text{{pos}}}}=1{{,}}0\), \(\delta={delta:.2f}\), \(\rho=0{{,}}7\). Числ --- численный расчёт, ИМ --- имитационное моделирование, Отн.\% --- относительная погрешность.}}"
    )
    lines.append(r"\label{tab:results_fixed}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LaTeX tables for fixed-service restarts.")
    parser.add_argument("--delta", type=float, default=0.05, help="Negative rate delta for the tables.")
    parser.add_argument("--jobs", type=int, default=200_000, help="Number of jobs for simulation.")
    parser.add_argument("--out", type=str, default="", help="Write output to this .tex file (optional).")
    args = parser.parse_args()

    text = build_tables(delta=float(args.delta), num_jobs=int(args.jobs), cvs=(0.8, 1.2))
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
