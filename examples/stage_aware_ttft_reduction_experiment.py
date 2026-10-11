"""Frozen exact-reduction comparison; run timing separately on an idle host.

python -m examples.stage_aware_ttft_reduction_experiment
python -m examples.stage_aware_ttft_reduction_experiment --timing
Outputs are new artifacts; previous GL3 measurements are never overwritten.
"""

# Independent physical and experimental parameters must remain explicit.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import argparse
import itertools
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from examples.stage_aware_ttft_experiment import COSTS, LAWS
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import ReducedStageAwareTtftCalc, StageAwareTtftCalc

FIELDS = ("ttft_mean", "w", "v", "accepted_rate", "blocking_probability", "utilization")


def configure(cls, cap, budget, queue, law, costs, load, **kwargs):
    """Use offered serialized load, not an assumed shared-queue threshold."""
    calc = cls(cap, budget, queue, **kwargs)
    calc.set_servers(law, *costs)
    calc.set_sources(float(load / (np.array([row[2] for row in law]) @ calc.isolated_times()[0])))
    return calc


def compare(cap, budget, queue, law, costs, load):
    """Record all exact first moments and explicit budget refusals."""
    parameters = {"cap": cap, "budget": budget, "queue": queue, "law": law, "costs": costs, "load": load}
    predictions = {}
    for name, cls, options in (
        ("full", StageAwareTtftCalc, {}),
        ("queue_only", ReducedStageAwareTtftCalc, {"canonical_decoders": False}),
        ("reduced", ReducedStageAwareTtftCalc, {}),
    ):
        try:
            predictions[name] = asdict(configure(cls, **parameters, **options).run())
        except ValueError as exc:
            predictions[name] = {"refusal": str(exc)}
    differences = {}
    if "refusal" not in predictions["full"]:
        for name in ("queue_only", "reduced"):
            if "refusal" not in predictions[name]:
                differences[name] = {
                    field: float(np.max(np.abs(np.asarray(predictions[name][field]) - predictions["full"][field])))
                    for field in FIELDS
                }
                for field in FIELDS:
                    if not np.allclose(predictions[name][field], predictions["full"][field], rtol=1e-10, atol=1e-10):
                        raise AssertionError(f"{name} differs in {field}: {parameters}")
    return {**parameters, "predictions": predictions, "absolute_differences": differences}


def accuracy():
    """All old48 cases, larger heldout support, and capacity refinement."""
    frozen = [
        compare(cap, budget, 6, LAWS[law_id], COSTS[cost_id], load)
        for cap, budget, law_id, cost_id, load in itertools.product((1, 2), (2, 4), range(2), range(2), (0.1, 0.5, 0.8))
    ]
    expanded = []
    law = ((1, 1, 0.3), (2, 2, 0.4), (4, 3, 0.3))
    for cap, multiplier, queue in itertools.product((2, 3), (1, 2), (2, 4, 6, 8)):
        row = compare(cap, multiplier * cap, queue, law, COSTS[0], 0.5)
        expanded.append(row)
        print(json.dumps({k: row[k] for k in ("cap", "budget", "queue")}), flush=True)
    refinement = []
    for law_id, budget in itertools.product(range(2), (2, 4)):
        predictions = []
        for queue in (2, 4, 6, 8, 12):
            result = configure(ReducedStageAwareTtftCalc, 2, budget, queue, LAWS[law_id], COSTS[0], 0.8).run()
            predictions.append({"queue_capacity": queue, **asdict(result)})
        refinement.append({"law": LAWS[law_id], "budget": budget, "load": 0.8, "predictions": predictions})
    differences = [
        value
        for row in frozen + expanded
        for comparison in row["absolute_differences"].values()
        for value in comparison.values()
    ]
    summary = {
        "frozen_cases": len(frozen),
        "expanded_cases": len(expanded),
        "expanded_full_refusals": sum("refusal" in row["predictions"]["full"] for row in expanded),
        "expanded_reduced_refusals": sum("refusal" in row["predictions"]["reduced"] for row in expanded),
        "max_absolute_reward_difference": max(differences),
    }
    return {"frozen": frozen, "expanded": expanded, "refinement": refinement, "summary": summary}


def timing():
    """Paired wall/CPU measurement including kernel build and sparse solve."""
    rows = []
    for budget, law_id in itertools.product((2, 4), range(2)):
        law = LAWS[law_id]
        rate = configure(StageAwareTtftCalc, 2, budget, 6, law, COSTS[0], 0.5).l
        probs = np.array([row[2] for row in law])
        repetitions = []
        for seed in range(93000, 93003):
            rng = np.random.default_rng(seed)
            arrivals = np.cumsum(rng.exponential(1 / rate, 12000))
            marks = np.array([item[:2] for item in law])[rng.choice(len(law), 12000, p=probs)]
            measured = {"seed": seed}
            methods = [
                ("full", StageAwareTtftCalc, {}),
                ("queue_only", ReducedStageAwareTtftCalc, {"canonical_decoders": False}),
                ("reduced", ReducedStageAwareTtftCalc, {}),
            ]
            # Rotate execution order; do not give one solver a fixed warm-cache position.
            offset = seed - 93000
            for name, cls, options in methods[offset:] + methods[:offset]:
                calc = configure(cls, 2, budget, 6, law, COSTS[0], 0.5, **options)
                wall, cpu = time.perf_counter(), time.process_time()
                result = calc.run()
                measured[name] = {
                    "wall": time.perf_counter() - wall,
                    "cpu": time.process_time() - cpu,
                    "states": result.states,
                    "ttft": result.ttft_mean,
                }
            sim = PrefillDecodeSim(2, budget)
            sim.set_servers(1, 1, *COSTS[0])
            wall, cpu = time.perf_counter(), time.process_time()
            result = sim.replay(arrivals, *marks.T, warmup_fraction=0.2)
            measured["des"] = {
                "wall": time.perf_counter() - wall,
                "cpu": time.process_time() - cpu,
                "ttft": result.ttft_mean,
            }
            repetitions.append(measured)
        summary = {
            "state_factor": repetitions[0]["full"]["states"] / repetitions[0]["reduced"]["states"],
            **{
                f"median_{clock}_{numerator}_over_{denominator}": float(
                    np.median([r[numerator][clock] / r[denominator][clock] for r in repetitions])
                )
                for clock, numerator, denominator in itertools.product(
                    ("wall", "cpu"), ("full", "reduced"), ("reduced", "des")
                )
                if numerator != denominator
            },
        }
        rows.append({"budget": budget, "law": law, "repetitions": repetitions, "summary": summary})
    return {"jobs": 12000, "queue_capacity": 6, "load": 0.5, "costs": COSTS[0], "rows": rows}


def main():
    """Write independent artifacts and print compact, auditable summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    result = timing() if args.timing else accuracy()
    result.update(schema_version=1, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    suffix = "timing" if args.timing else "results"
    output = Path(f"docs/research/stage-aware-ttft-reduction-{suffix}-2026.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([row["summary"] for row in result["rows"]] if args.timing else result["summary"], indent=2))


if __name__ == "__main__":
    main()
