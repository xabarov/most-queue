"""EPIC-096 preregistered grids and fresh infinite token replay.

Run python -m examples.phase_aware_stability_experiment. Float mean bounds
and exact rational drift verification are reported separately, with refusals.
"""

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from examples.infinite_stage_aware_ttft_experiment import width_passes
from examples.stage_aware_ttft_experiment import COSTS, LAWS
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import PhaseAwareStabilityCalc, PhaseAwareTtftCalc

# The physical configuration is intentionally explicit, not a positional bundle.
# pylint: disable=too-many-arguments,too-many-positional-arguments


def configured(cls, cap, budget, law, costs, rate=1.0, **kwargs):
    """Configure without fitting any physical cost to DES."""
    calc = cls(cap, budget, **kwargs)
    calc.set_servers(law, *costs)
    calc.set_sources(rate)
    return calc


def refine(cap, budget, law, costs, rate):
    """Preserve every domain attempt and explicit refusal."""
    calc = configured(PhaseAwareTtftCalc, cap, budget, law, costs, rate)
    attempts = []
    for truncation in (8, 16, 32, 64, 128, 256):
        calc.truncation = truncation
        try:
            result = calc.run()
        except ValueError as exc:
            return {"status": "refused", "reason": str(exc), "refinements": attempts}
        attempts.append(asdict(result))
        if width_passes(result):
            return {"status": "bounded", "result": asdict(result), "refinements": attempts}
    return {"status": "accuracy_refused", "reason": "frozen width target not reached by N256", "refinements": attempts}


def run():
    """Run the original 48 cases, 40 threshold probes and 12 fresh DES seeds."""
    historical = json.loads(
        Path("docs/research/infinite-stage-aware-ttft-results-2026.json").read_text(encoding="utf-8")
    )
    rows, extended, fresh = [], [], []
    for cap, budget, law_id, cost_id, load in itertools.product((1, 2), (2, 4), range(2), range(2), (0.1, 0.5, 0.8)):
        law, costs = LAWS[law_id], COSTS[cost_id]
        calc = configured(PhaseAwareStabilityCalc, cap, budget, law, costs)
        rate = float(load / (np.array([mark[2] for mark in law]) @ calc.isolated_times()[0]))
        calc.set_sources(rate)
        old = next(
            item
            for item in historical["rows"]
            if item["batch_cap"] == cap
            and item["token_budget"] == budget
            and item["law"] == [list(mark) for mark in law]
            and item["costs"] == list(costs)
            and item["load"] == load
        )
        row = {
            "batch_cap": cap,
            "token_budget": budget,
            "law": law,
            "costs": costs,
            "load": load,
            "arrival_rate": rate,
            "phase_certificate": asdict(calc.run()),
            "epic095_status": old["status"],
            **refine(cap, budget, law, costs, rate),
        }
        rows.append(row)
        print(f"B{cap} M{budget} law{law_id} costs{cost_id} load{load}: {row['status']}", flush=True)
    for budget, law_id, cost_id in itertools.product((2, 4), range(2), range(2)):
        law, costs = LAWS[law_id], COSTS[cost_id]
        calc = configured(PhaseAwareStabilityCalc, 2, budget, law, costs)
        capacity = calc.run().capacity
        for load in (0.6, 0.85, 0.95, 1.0, 1.05):
            calc.set_sources(load * capacity)
            extended.append(
                {
                    "token_budget": budget,
                    "law": law,
                    "costs": costs,
                    "saturated_load_design": load,
                    "arrival_rate": load * capacity,
                    "certificate": asdict(calc.run()),
                }
            )
    for budget, load in itertools.product((2, 4), (0.6, 0.85)):
        law, costs = LAWS[1], COSTS[0]
        capacity = configured(PhaseAwareStabilityCalc, 2, budget, law, costs).run().capacity
        rate, samples = load * capacity, []
        prediction = refine(2, budget, law, costs, rate)
        for seed in range(96000, 96003):
            rng = np.random.default_rng(seed)
            arrivals = np.cumsum(rng.exponential(1 / rate, 50000))
            marks = np.array([item[:2] for item in law])[rng.choice(2, len(arrivals), p=[0.6, 0.4])]
            sim = PrefillDecodeSim(2, budget)
            sim.set_servers(1, 1, *costs)
            result = sim.replay(arrivals, *marks.T, warmup_fraction=0.2, record_iterations=False)
            samples.append({"seed": seed, "w": result.w[0], "v": result.v[0], "ttft": result.ttft_mean})
        estimates = {}
        for key in ("w", "v", "ttft"):
            values = np.array([item[key] for item in samples])
            mean = float(values.mean())
            half = float(t.ppf(0.975, 2) * values.std(ddof=1) / np.sqrt(3))
            estimates[key] = {"mean": mean, "ci95": [mean - half, mean + half]}
            if prediction["status"] == "bounded":
                point = sum(prediction["result"]["mean_bounds"][key]) / 2
                estimates[key].update(
                    prediction=point,
                    absolute_error=abs(mean - point),
                    prediction_inside_ci=mean - half <= point <= mean + half,
                )
        fresh.append(
            {
                "token_budget": budget,
                "saturated_load_design": load,
                "arrival_rate": rate,
                "jobs_per_seed": 50000,
                "warmup_fraction": 0.2,
                "samples": samples,
                "reference": estimates,
                "analytic": prediction,
            }
        )
        print(f"Fresh M{budget} saturated load{load}: {prediction['status']}", flush=True)
    bounded = [row for row in rows if row["status"] == "bounded"]
    return {
        "protocol": "EPIC-096 frozen before computation; exact binary row drift, floating mean bounds",
        "historical_artifact": "infinite-stage-aware-ttft-results-2026.json",
        "rows": rows,
        "extended_stability": extended,
        "fresh_des": fresh,
        "summary": {
            "total": len(rows),
            "bounded": len(bounded),
            "refused": len(rows) - len(bounded),
            "phase_stable": sum(row["phase_certificate"]["stable"] for row in rows),
            "old_refusals_recovered": sum(row["epic095_status"] != "bounded" for row in bounded),
            "max_selected_truncation": max(row["result"].get("truncation") or 0 for row in bounded),
            "max_states": max(row["result"]["states"] for row in bounded),
            "extended_classifications": {
                status: sum(row["certificate"]["classification"] == status for row in extended)
                for status in ("positive_recurrence", "positive_drift_escape", "critical_or_numerically_unresolved")
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/research/phase-aware-stability-results-2026.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
