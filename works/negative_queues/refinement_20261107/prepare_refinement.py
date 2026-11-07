"""Recalculate resampling rows and assemble v6 without changing earlier versions."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.refinement_20261107.restart_phase import calculate

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "revision_20261107"


def main():
    data = copy.deepcopy(json.loads((OLD / "experiment-results.json").read_text()))
    full_work = json.loads((ROOT / "full-work-results.json").read_text())
    for row in data["results"]:
        row["method"] = "previous deletion or fixed-duration approximation"
        if row["scenario"] == "fixed" and row["mode"] == "disaster":
            refined = next(r for r in full_work["results"] if r["cv"] == row["cv"])
            row["previous_calculation"] = row["calculation"]
            row["calculation"] = refined["calculation"]
            row["method"] = refined["calculation"]["method"]
            row["full_work_validation"] = refined["comparison"]
            row["queue_truncation_study"] = full_work["queue_truncation_change"]
            row.pop("truncation_change", None)
            continue
        if row["scenario"] != "restart":
            continue
        h = H2Distribution.get_params_clx(gamma_moments_by_mean_and_cv(2.1, row["cv"]))
        options = dict(n=3, lam=1, delta=0.3, y=[h.p1, 1 - h.p1], mu=[h.mu1, h.mu2], mode=row["mode"], order=2)
        small = calculate(**options, levels=160, behind=160)
        large = calculate(**options, levels=240, behind=240)
        more_levels = calculate(**options, levels=240, behind=160)
        more_behind = calculate(**options, levels=160, behind=240)
        row["previous_calculation"] = row["calculation"]
        row["calculation"] = small
        row["truncation_change"] = max(abs(small[k][j] - large[k][j]) for k in ["w", "v"] for j in [0, 1])
        row["truncation_study"] = {
            "levels_only": {k: more_levels[k] for k in ["w", "v"]},
            "behind_only": {k: more_behind[k] for k in ["w", "v"]},
            "both": {k: large[k] for k in ["w", "v"]},
        }
        row["method"] = "direct phase-state equations with tagged-customer moments"
        print(row["cv"], row["mode"], {k: small[k] for k in ["w", "v", "sw", "sv"]}, "cutoff", row["truncation_change"])
    data["simulation_source"] = "../revision_20261107/replications.jsonl; original 96 replications retained"
    data["independent_validation"] = (
        "independent-results.json; 80 additional replications, 50000 warm-up and 500000 observed completions each"
    )
    (ROOT / "experiment-results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("tables", OLD / "prepare_article.py")
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    rows = data["results"]
    texts = [
        tables.table(
            [r for r in rows if r["scenario"] != "fixed"],
            ("w", "v", "w_1", "v_1"),
            r"Средние ожидание и пребывание при $\delta=0{,}3$. «Новая длительность» означает независимый выбор длительности после каждого прерывания; «все каналы» --- одновременный повтор на всех занятых каналах. ДИ --- полуширина 95\%-го доверительного интервала.",
            "tab:results",
        ),
        tables.table(
            [r for r in rows if r["scenario"] == "fixed"],
            ("w", "v", "w_1", "v_1"),
            r"Средние при повторе с сохранённой длительностью, $\delta=0{,}02$: после каждого прерывания заново выполняется первоначально выбранная работа.",
            "tab:results_fixed",
        ),
        tables.table(
            rows,
            ("sw", "sv", r"\sigma_W", r"\sigma_V"),
            r"Стандартные отклонения ожидания и пребывания. Варианты повторного обслуживания и интенсивности соответствуют таблицам 1 и 2.",
            "tab:sko",
        ),
    ]
    replacements = {
        "RCS, новый $B$": "RCS, новая длительность",
        "Общий, новый $B$": "Все каналы, новая длительность",
        "RCS, прежний $B$": "RCS, сохранённая длительность",
        "Общий, прежний $B$": "Все каналы, сохранённая длительность",
        r"\begin{tabular}{@{}lcccccc@{}}": r"\begin{tabular}{@{}p{4.2cm}cccccc@{}}",
    }
    source = "\n\n".join(texts) + "\n"
    for old, new in replacements.items():
        source = source.replace(old, new)
    (ROOT / "generated_tables_fixed.tex").write_text(source)
    preamble = (OLD / "negative_queues_takahasi_takami.tex").read_text().split(r"\begin{abstract}")[0]
    body = (ROOT / "article_body.tex").read_text().replace(r"\end{samepage}", "").replace(r"\).", r".\)")
    (ROOT / "negative_queues_takahasi_takami.tex").write_text(preamble + body)
    shutil.copy2(OLD / "citations.bib", ROOT / "citations.bib")
    shutil.copytree(OLD / "figures", ROOT / "figures", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
