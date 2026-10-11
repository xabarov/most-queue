"""Frozen GL3 grid: first-moment predictions, independent token DES and cost.

python -m examples.stage_aware_ttft_experiment
No GPU measurements or synthetic SLO-tail estimates are implied.
"""

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import PrefillDecodeCalc, StageAwareTtftCalc

LAWS = (((2, 2, 1.0),), ((1, 1, 0.6), (3, 2, 0.4)))
COSTS = ((0.1, 0.02, 0.003), (0.07, 0.03, 0.005))


def lps_baseline(rate, full, first, probabilities, cap):
    """Zhang2009 (6.28--6.30): constant-speed LPS adaptation, NOT our policy."""
    mean = float(probabilities @ full)
    cv2 = float(probabilities @ full**2 / mean**2 - 1)
    rho = rate * mean
    dp = rho**cap  # Poisson c_a^2=1 makes Zhang's exponent K.
    wait = (1 + cv2) / 2 * dp * mean / (1 - rho)
    return wait + (1 - dp) * float(probabilities @ first) / (1 - rho)


def run(count=12000):
    """Prespecified heterogeneous and held-out cost grid without fitted params."""
    rows = []
    for cap, budget, law_id, cost_id, load in itertools.product((1, 2), (2, 4), range(2), range(2), (0.1, 0.5, 0.8)):
        law, costs = LAWS[law_id], COSTS[cost_id]
        probs = np.array([row[2] for row in law])
        calc = StageAwareTtftCalc(cap, budget)
        calc.set_servers(law, *costs)
        full, first = calc.isolated_times()
        rate = load / (probs @ full)
        calc.set_sources(rate)
        predictions = []
        for queue in (6,) if cap == 1 else (2, 4, 6):
            calc.queue_capacity = queue
            predictions.append({"queue_capacity": queue, **asdict(calc.run())})
        means = probs @ np.array([row[:2] for row in law])
        old = PrefillDecodeCalc(cap, budget)
        old.set_sources(float(rate))
        old.set_servers(*means, *costs)
        old_result = old.run()
        profile = old.stage_profile(max(1.0, old_result.batch_mean))
        # Explicit engineering correction, not the new predictor.
        clamped = old_result.w[0] + profile["prefill"] + (profile["service"] - profile["prefill"]) / means[1]
        refs, durations, queues = [], [], []
        for seed in range(93000, 93003):
            rng = np.random.default_rng(seed)
            arrivals = np.cumsum(rng.exponential(1 / rate, count))
            marks = np.array([row[:2] for row in law], dtype=int)[rng.choice(len(law), count, p=probs)]
            sim = PrefillDecodeSim(cap, budget)
            sim.set_servers(1, 1, *costs)
            result = sim.replay(arrivals, *marks.T, warmup_fraction=0.2)
            refs.append(result.ttft_mean)
            durations.append(result.duration)
            queues.append(int(np.sum(np.array(result.completion_times) > arrivals[-1])))
        reference = float(np.mean(refs))
        half = float(t.ppf(0.975, 2) * np.std(refs, ddof=1) / np.sqrt(3))
        candidates = {
            "literal_l3": old_result.ttft_mean,
            "engineering_clamp": clamped,
            "constant_speed_lps": lps_baseline(rate, full, first, probs, cap),
            "stage_aware": predictions[-1]["ttft_mean"],
        }
        rows.append(
            {
                "batch_cap": cap,
                "token_budget": budget,
                "law": law,
                "costs": costs,
                "cost_set": "development" if cost_id == 0 else "heldout",
                "load": load,
                "arrival_rate": float(rate),
                "reference": {
                    "ttft": reference,
                    "ci95": [reference - half, reference + half],
                    "seed_means": refs,
                    "mean_cpu_seconds": float(np.mean(durations)),
                    "backlog_at_last_arrival": queues,
                },
                "finite_refinements": predictions,
                "predictions": {key: float(value) for key, value in candidates.items()},
                "absolute_errors": {key: float(abs(value - reference)) for key, value in candidates.items()},
            }
        )
        print(f"GL3 B={cap} M={budget} law={law_id} costs={cost_id} load={load} complete", flush=True)
    summary = {}
    for label, selected in (
        ("all", rows),
        ("B_gt_1", [r for r in rows if r["batch_cap"] > 1]),
        ("heldout_costs", [r for r in rows if r["cost_set"] == "heldout"]),
        ("B1", [r for r in rows if r["batch_cap"] == 1]),
    ):
        errors = {name: float(np.median([row["absolute_errors"][name] for row in selected])) for name in candidates}
        reduction = 1 - errors["stage_aware"] / errors["literal_l3"]
        summary[label] = {
            "median_absolute_errors": errors,
            "reduction_vs_literal": reduction,
            "target_25pct_passed": reduction >= 0.25,
        }
    # Choosing lowest mean TTFT is a diagnostic distinct from SLO ranking.
    choices = []
    for law, costs, load in itertools.product(LAWS, COSTS, (0.1, 0.5, 0.8)):
        group = [r for r in rows if r["law"] == law and r["costs"] == costs and r["load"] == load]
        # Rate depends on M through isolated chunks: not equal offered rate!
        # Compare B at fixed M only; do not invent cross-budget configuration regret.
        for budget in (2, 4):
            pair = [r for r in group if r["token_budget"] == budget]
            truth = min(pair, key=lambda r: r["reference"]["ttft"])
            choices.append(
                {
                    "law": law,
                    "costs": costs,
                    "load": load,
                    "budget": budget,
                    "true_best_cap": truth["batch_cap"],
                    "chosen_caps": {
                        name: min(pair, key=lambda r, metric=name: r["predictions"][metric])["batch_cap"]
                        for name in candidates
                    },
                }
            )
    return {
        "schema_version": 1,
        "jobs_per_seed": count,
        "seeds": [93000, 93001, 93002],
        "rows": rows,
        "summary": summary,
        "choices": choices,
        "scope": "Mean TTFT only; finite-loss approximation for B>1; known methods; no GPU validation",
    }


def main():
    """Save the complete grid and failures, without selecting a winning subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/research/stage-aware-ttft-results-2026.json"))
    parser.add_argument("--jobs", type=int, default=12000)
    args = parser.parse_args()
    if args.jobs < 1000:
        raise ValueError("at least1000 jobs required")
    result = run(args.jobs)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
