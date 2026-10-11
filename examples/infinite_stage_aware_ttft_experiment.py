"""Frozen infinite-queue grid with all refusals and fresh independent DES.

Run python -m examples.infinite_stage_aware_ttft_experiment. Historical DES
references are reused, never relabelled as fresh measurements. Model bounds
are floating-evaluated analytic bounds, not certified arithmetic intervals.
"""

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from examples.stage_aware_ttft_experiment import COSTS, LAWS
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import InfiniteStageAwareTtftCalc


def width_passes(result):
    """Apply the frozen 0.1% width criterion to all three means."""
    return all(hi - lo <= 1e-8 + 0.001 * ((hi + lo) / 2) for lo, hi in result.mean_bounds.values())


def run():
    """Retain every accuracy/stability refusal without tuning the pilot."""
    historical = json.loads(Path("docs/research/stage-aware-ttft-results-2026.json").read_text(encoding="utf-8"))
    rows = []
    for cap, budget, law_id, cost_id, load in itertools.product((1, 2), (2, 4), range(2), range(2), (0.1, 0.5, 0.8)):
        law, costs = LAWS[law_id], COSTS[cost_id]
        calc = InfiniteStageAwareTtftCalc(cap, budget)
        calc.set_servers(law, *costs)
        probs = np.array([row[2] for row in law])
        rate = float(load / (probs @ calc.isolated_times()[0]))
        calc.set_sources(rate)
        row = {
            "batch_cap": cap,
            "token_budget": budget,
            "law": law,
            "costs": costs,
            "load": load,
            "arrival_rate": rate,
            "refinements": [],
        }
        row["dominating_load"] = float(rate * (probs @ calc.dominating_work()))
        for truncation in (4, 8, 16, 32, 64):
            calc.truncation = truncation
            try:
                result = calc.run()
            except ValueError as exc:
                row.update(status="refused", reason=str(exc))
                break
            row["refinements"].append(asdict(result))
            if width_passes(result):
                row.update(status="bounded", result=asdict(result))
                break
        else:
            row.update(status="accuracy_refused", reason="frozen width target not reached by N64")
        old = next(
            item
            for item in historical["rows"]
            if item["batch_cap"] == cap
            and item["token_budget"] == budget
            and item["law"] == [list(mark) for mark in law]
            and item["costs"] == list(costs)
            and item["load"] == load
        )
        row["historical_reference"] = old["reference"]
        row["historical_finite_k6"] = old["finite_refinements"][-1]
        if row["status"] == "bounded":
            row["historical_ttft_absolute_error"] = abs(row["result"]["ttft_mean"] - old["reference"]["ttft"])
            finite = row["historical_finite_k6"]["ttft_mean"]
            row["finite_k6_minus_infinite_ttft"] = finite - row["result"]["ttft_mean"]
            lo, hi = row["result"]["mean_bounds"]["ttft"]
            row["finite_k6_inside_infinite_ttft_bounds"] = lo <= finite <= hi
        rows.append(row)
        print(f"B{cap} M{budget} law{law_id} costs{cost_id} load{load}: {row['status']}", flush=True)
    fresh = []
    for budget in (2, 4):
        calc = InfiniteStageAwareTtftCalc(2, budget, 16)
        calc.set_servers(LAWS[1], *COSTS[0])
        calc.set_sources(1.3)
        samples = []
        for seed in range(95000, 95003):
            rng = np.random.default_rng(seed)
            arrivals = np.cumsum(rng.exponential(1 / 1.3, 30000))
            marks = np.array([item[:2] for item in LAWS[1]])[rng.choice(2, len(arrivals), p=[0.6, 0.4])]
            sim = PrefillDecodeSim(2, budget)
            sim.set_servers(1, 1, *COSTS[0])
            result = sim.replay(arrivals, *marks.T, warmup_fraction=0.2)
            samples.append({"seed": seed, "w": result.w[0], "v": result.v[0], "ttft": result.ttft_mean})
        estimates = {}
        prediction = calc.run()
        for key, predicted in (("w", prediction.w[0]), ("v", prediction.v[0]), ("ttft", prediction.ttft_mean)):
            values = np.array([item[key] for item in samples])
            mean = float(values.mean())
            half = float(t.ppf(0.975, 2) * values.std(ddof=1) / np.sqrt(3))
            estimates[key] = {"mean": mean, "ci95": [mean - half, mean + half], "absolute_error": abs(mean - predicted)}
        fresh.append(
            {"token_budget": budget, "samples": samples, "reference": estimates, "prediction": asdict(prediction)}
        )
    solved = [row for row in rows if row["status"] == "bounded"]
    batched = [row for row in solved if row["batch_cap"] > 1]
    return {
        "protocol": "EPIC-095 frozen before computation; ordinary floating-point analytic truncation bounds",
        "historical_artifact": "stage-aware-ttft-results-2026.json",
        "rows": rows,
        "fresh_des": fresh,
        "summary": {
            "total": len(rows),
            "bounded": len(solved),
            "refused": len(rows) - len(solved),
            "b_gt_1_bounded": len(batched),
            "max_states": max(row["result"]["states"] for row in solved),
            "max_selected_truncation": max(row["result"]["truncation"] for row in batched),
            "max_waiting_tail_upper": max(row["result"]["waiting_tail_upper"] for row in batched),
            "median_historical_ttft_absolute_error": float(
                np.median([row["historical_ttft_absolute_error"] for row in solved])
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research/infinite-stage-aware-ttft-results-2026.json")
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
