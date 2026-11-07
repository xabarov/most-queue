"""Parameter-study comparisons, including all misses of pointwise intervals."""

import json
from pathlib import Path

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.utils.restarts import effective_service_moments
from works.negative_queues.revision_v7_20261107.study_cases import cases
from works.negative_queues.revision_v7_20261107.summarize_step1 import extrapolate

ROOT = Path(__file__).resolve().parent


def read(case, k, n, b):
    name = f"{case['id']}-k{k}-n{n}-b{b}.json"
    if case["id"] == "case11" and k == 12 and n == 240:
        selected = ROOT / "case11-k12-relaxed.json"
        if selected.exists():
            result = json.loads(selected.read_text())
            assert result["balance_residual"] < 1e-10 and result["service_mean_residual"] < 1e-6
            return result
    fast = ROOT / "step3_fast" / name
    return json.loads((fast if fast.exists() else ROOT / "step3" / name).read_text())


def main():
    simulations = json.loads((ROOT / "step3-independent-results.json").read_text())
    results = []
    for case in cases():
        if case["action"] != "fixed":
            calc = read(case, 0, 240, 80)
            small = read(case, 0, 160, 40)
            change = {key: (np.array(calc[key]) - small[key]).tolist() for key in ["w", "v"]}
            assert (
                max(abs(x) / (1 + abs(calc[key][i])) for key in ["w", "v"] for i, x in enumerate(change[key])) < 1e-6
            ), (case, change)
            sources = [small, calc]
        else:
            n, b = (120, 32) if case["n"] == 2 else (240, 40)
            if case["mode"] == "disaster":
                b = 0
            sources = [read(case, k, n, b) for k in [4, 8, 12]]
            calc, weights = extrapolate(sources, [4, 8, 12])
            calc.update(extrapolation_weights=weights, classes=3, stages=[4, 8, 12], levels=n, behind=b)
            if case["mode"] == "disaster":
                bm = gamma_moments_by_mean_and_cv(case["mean"], case["cv"])
                h = H2Distribution.get_params_clx(bm)
                t = effective_service_moments(
                    bm, [h.p1, 1 - h.p1], [h.mu1, h.mu2], case["delta"], fixed=True, num_of_moments=2
                )
                calc["v"] = [calc["w"][0] + t[0], calc["w"][1] + 2 * calc["w"][0] * t[0] + t[1]]
                calc["sv"] = float(np.sqrt(calc["v"][1] - calc["v"][0] ** 2))
            change = {}
            if case["n"] == 3:
                large = read(case, 8, 400, b)
                change = {key: (np.array(large[key]) - sources[1][key]).tolist() for key in ["w", "v"]}
                assert (
                    max(abs(x) / (1 + abs(large[key][i])) for key in ["w", "v"] for i, x in enumerate(change[key]))
                    < 1e-4
                ), (case, change)
                sources.append(large)
        ref = next(r for r in simulations if r["case"]["id"] == case["id"])
        comparison = {}
        for key in ["w", "v", "sw", "sv", "q"]:
            value = calc[key][0] if key in ["w", "v"] else calc[key]
            estimate = ref["estimates"][key]
            comparison[key] = dict(
                calculation=value,
                simulation=estimate["mean"],
                half_ci95=estimate["half_ci95"],
                difference_percent=100 * (value / estimate["mean"] - 1),
                inside_ci95=bool(abs(value - estimate["mean"]) <= estimate["half_ci95"] + 1e-12),
            )
        results.append(
            dict(
                case=case,
                calculation={k: v for k, v in calc.items() if k != "p"},
                comparison=comparison,
                truncation_changes=change,
                total_calculation_seconds=sum(r["seconds"] for r in sources),
                peak_rss_kib=max(r["peak_rss_kib"] for r in sources),
                stationarity_half_checks={
                    k: ref[k] for k in ["half_difference_mean", "half_difference_ci95", "maximum_observed_queue"]
                },
            )
        )
    (ROOT / "step3-results.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(
            r["case"]["id"],
            {k: round(v["difference_percent"], 3) for k, v in r["comparison"].items()},
            "misses",
            [k for k, v in r["comparison"].items() if not v["inside_ci95"]],
            flush=True,
        )


if __name__ == "__main__":
    main()
