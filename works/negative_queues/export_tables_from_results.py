"""
Export LaTeX table fragments (booktabs) from results.json produced by work_mgn_negatives.py.

Modes:
1) W/V table: reads utilization results, --wv (default).
2) SKO table: reads coefs results, outputs sigma_W and sigma_V (--sko).
3) Legacy: state probabilities and moments (--legacy).

Run from repo root:
  ./.venv/bin/python works/negative_queues/export_tables_from_results.py [--wv|--sko|--legacy]
"""

import argparse
import json
import os
import sys


def variance_from_moments(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    return v[1] - v[0] ** 2


def skewness_from_moments(v: list[float]) -> float:
    if len(v) < 3:
        return 0.0
    var = variance_from_moments(v)
    if var <= 0:
        return 0.0
    mu = v[0]
    mu3 = v[2] - 3 * v[1] * mu + 2 * mu**3
    return mu3 / (var**1.5)


def kurtosis_from_moments(v: list[float]) -> float:
    if len(v) < 4:
        return 0.0
    var = variance_from_moments(v)
    if var <= 0:
        return 0.0
    mu = v[0]
    mu4 = v[3] - 4 * v[2] * mu + 6 * v[1] * mu**2 - 3 * mu**4
    return mu4 / (var**2) - 3.0


def rel_err(a: float, b: float) -> float:
    if b == 0:
        return 0.0 if a == 0 else 100.0
    return 100.0 * abs(a - b) / abs(b)


# Four scenarios: (subdir, LaTeX row label)
SCENARIOS = [
    ("rcs/remove", "RCS (удаление)"),
    ("rcs/requeue", "RCS (повторное обслуж.)"),
    ("disaster/clear", "Катастрофы (очищение)"),
    ("disaster/requeue", "Катастрофы (повторное обслуж.)"),
]


def export_wv_table(base: str, target_rho: float = 0.7) -> None:
    """Print one LaTeX table: rows = 4 scenarios, cols = E[W] (Числ, ИМ, %), E[V] (Числ, ИМ, %)."""
    fig_base = os.path.join(base, "negative_queues_figures")
    rows = []
    for subdir, label in SCENARIOS:
        path = os.path.join(fig_base, subdir, "utilization", "results.json")
        if not os.path.isfile(path):
            print(f"% {path} not found, skip {label}", file=sys.stderr)
            rows.append((label, None))
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        util = data["utilization_factor"]
        idx = min(range(len(util)), key=lambda i: abs(util[i] - target_rho))
        calc_list = data["calc"]
        sim_list = data["sim"]
        c, s = calc_list[idx], sim_list[idx]

        def first_moment(x: list, i: int = 0) -> float:
            val = x[i]
            return float(val[0]) if isinstance(val, list) else float(val)

        w_calc = first_moment(c["w"])
        w_sim = first_moment(s["w"])
        v_calc = first_moment(c["v"])
        v_sim = first_moment(s["v"])
        w_err = rel_err(w_sim, w_calc)
        v_err = rel_err(v_sim, v_calc)
        rows.append((label, (w_calc, w_sim, w_err, v_calc, v_sim, v_err)))

    print("% ----- Table E[W] and E[V] for 4 scenarios (utilization point nearest to rho=0.7) -----")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lrrrrrr@{}}")
    print(r"\toprule")
    print(
        r"Сценарий & $\mathbb{E}[W]$ (Числ) & $\mathbb{E}[W]$ (ИМ) & Отн.\% & $\mathbb{E}[V]$ (Числ) & $\mathbb{E}[V]$ (ИМ) & Отн.\% \\"
    )
    print(r"\midrule")
    for label, vals in rows:
        if vals is None:
            print(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
        else:
            w_calc, w_sim, w_err, v_calc, v_sim, v_err = vals
            print(f"{label} & {w_calc:.4g} & {w_sim:.4g} & {w_err:.2f} & {v_calc:.4g} & {v_sim:.4g} & {v_err:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Средние времена ожидания $\mathbb{E}[W]$ и пребывания $\mathbb{E}[V]$: численный расчёт (Числ), имитационное моделирование (ИМ), относительная погрешность в \%.}"
    )
    print(r"\end{table}")


def std_from_moments(moments: list) -> float:
    """Standard deviation from raw moments: sqrt(E[X^2] - E[X]^2)."""
    if len(moments) < 2:
        return 0.0
    m1 = float(moments[0]) if isinstance(moments[0], (int, float)) else float(moments[0][0])
    m2 = float(moments[1]) if isinstance(moments[1], (int, float)) else float(moments[1][0])
    var = max(0.0, m2 - m1**2)
    return var**0.5


def export_sko_table(base: str, target_cvs: tuple[float, ...] = (0.8, 1.2)) -> None:
    """Print LaTeX table: same structure as Table 1, but for sigma_W and sigma_V.
    Rows = 4 scenarios x 2 CV blocks, cols = sigma_W (Числ, ИМ, %), sigma_V (Числ, ИМ, %).
    Uses coefs results."""
    fig_base = os.path.join(base, "negative_queues_figures")
    all_data: list[tuple[str, list[tuple[float, float, float, float, float, float] | None]]] = []
    for subdir, label in SCENARIOS:
        path = os.path.join(fig_base, subdir, "coefs", "results.json")
        if not os.path.isfile(path):
            print(f"% {path} not found, skip {label}", file=sys.stderr)
            all_data.append((label, [None] * len(target_cvs)))
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cv_list = data["service_time_variation_coef"]
        calc_list = data["calc"]
        sim_list = data["sim"]
        row_vals = []
        for target_cv in target_cvs:
            idx = min(range(len(cv_list)), key=lambda i: abs(cv_list[i] - target_cv))
            c, s = calc_list[idx], sim_list[idx]
            sw_c = std_from_moments(c["w"])
            sw_s = std_from_moments(s["w"])
            sv_c = std_from_moments(c["v"])
            sv_s = std_from_moments(s["v"])
            sw_err = rel_err(sw_s, sw_c)
            sv_err = rel_err(sv_s, sv_c)
            row_vals.append((sw_c, sw_s, sw_err, sv_c, sv_s, sv_err))
        all_data.append((label, row_vals))

    print("% ----- Table sigma(W) and sigma(V), same structure as Table 1 (2 CV blocks) -----")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{@{}lccccccc@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{$\sigma_W$} & \multicolumn{3}{c}{$\sigma_V$} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"Сценарий & Числ & ИМ & Отн.\% & Числ & ИМ & Отн.\% \\")
    print(r"\midrule")
    cv_labels = [
        r"\multicolumn{7}{@{}l}{\textit{Коэффициент вариации} $CV=0{,}8$} \\",
        r"\multicolumn{7}{@{}l}{\textit{Коэффициент вариации} $CV=1{,}2$ (базовое значение)} \\",
    ]
    for cv_idx, target_cv in enumerate(target_cvs):
        if cv_idx < len(cv_labels):
            print(cv_labels[cv_idx])
        for label, row_vals in all_data:
            if row_vals[cv_idx] is None:
                print(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
            else:
                sw_c, sw_s, sw_err, sv_c, sv_s, sv_err = row_vals[cv_idx]
                print(f"{label} & {sw_c:.4g} & {sw_s:.4g} & {sw_err:.2f} & {sv_c:.4g} & {sv_s:.4g} & {sv_err:.2f} \\\\")
        if cv_idx < len(target_cvs) - 1:
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Среднеквадратичные отклонения $\sigma_W$ и $\sigma_V$ при фиксированных $n=3$, $\lambda_{\text{pos}}=1.0$, $\delta=0.3$, $\rho=0.7$ и различных коэффициентах вариации времени обслуживания.}"
    )
    print(r"\label{tab:sko}")
    print(r"\end{table}")


def export_legacy(base: str, target_rho: float = 0.7, n_prob_rows: int = 10) -> None:
    """Legacy: state probabilities and moments of V for rcs and disaster (single path each)."""
    for discipline, label in [("rcs", "RCS"), ("disaster", "DISASTER")]:
        path = os.path.join(base, "negative_queues_figures", discipline, "utilization", "results.json")
        if not os.path.isfile(path):
            print(f"# {path} not found, skip {label}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        util = data["utilization_factor"]
        idx = min(range(len(util)), key=lambda i: abs(util[i] - target_rho))
        calc_list = data["calc"]
        sim_list = data["sim"]
        c, s = calc_list[idx], sim_list[idx]
        print(f"% ----- Tables for {label} (utilization_factor={util[idx]:.4f}) -----")
        print("% State probabilities: Состояние & Числ & ИМ & Отн. погр., \\%")
        for i in range(n_prob_rows):
            pc = c["p"][i] if i < len(c["p"]) else 0.0
            ps = s["p"][i] if i < len(s["p"]) else 0.0
            err = rel_err(pc, ps)
            print(f"{i} & {pc:.5f} & {ps:.5f} & {err:.2f} \\\\")
        vc, vs = c["v"], s["v"]
        if label == "RCS":
            rows = [
                ("Среднее", vc[0], vs[0]),
                ("Дисперсия", variance_from_moments(vc), variance_from_moments(vs)),
                ("Асимметрия", skewness_from_moments(vc), skewness_from_moments(vs)),
                ("Эксцесс", kurtosis_from_moments(vc), kurtosis_from_moments(vs)),
            ]
        else:
            rows = [(str(k), vc[k], vs[k]) for k in range(min(4, len(vc), len(vs)))]
        print("% Moments of V: Момент & Числ & ИМ & Отн. погр., \\%")
        for name, ac, as_ in rows:
            err = rel_err(ac, as_)
            print(f"{name} & {ac:.4g} & {as_:.4g} & {err:.2f} \\\\")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LaTeX table fragments from results.json")
    parser.add_argument("--legacy", action="store_true", help="Legacy: state probs and moments per discipline")
    parser.add_argument("--wv", action="store_true", help="E[W] and E[V] table (default)")
    parser.add_argument("--sko", action="store_true", help="sigma_W and sigma_V table (Table 2)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(base))
    os.chdir(repo_root)

    if args.sko:
        export_sko_table(base)
    elif args.legacy:
        export_legacy(base)
        print("% Paste the above snippets into the corresponding \\begin{tabular}...\\end{tabular}.")
    else:
        export_wv_table(base)


if __name__ == "__main__":
    main()
