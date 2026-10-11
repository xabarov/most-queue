"""EPIC-097 frozen original/near-critical ablations and independent token DES.

Run OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m
examples.core_aware_continuation_experiment. Never overwrite historical results.
Single-run elapsed time and CSR storage are descriptive, not peak RSS/speedup CI.
"""

# Physical model and ablation are explicit experiment inputs.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import argparse
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from examples.infinite_stage_aware_ttft_experiment import width_passes
from examples.phase_aware_stability_experiment import configured
from examples.stage_aware_ttft_experiment import COSTS, LAWS
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import CoreAwareTtftCalc, PhaseAwareStabilityCalc, PhaseAwareTtftCalc

DOMAINS = (8, 16, 32, 64, 128, 256, 512, 1024)
MODES = ("global", "phase", "core")


def refine(mode, cap, budget, law, costs, rate):
    """Record every attempt, including explicit budget/accuracy refusals."""
    kwargs = {} if mode == "global" else {"mode": mode, "core_truncation": 8}
    cls = PhaseAwareTtftCalc if mode == "global" else CoreAwareTtftCalc
    calc = configured(cls, cap, budget, law, costs, rate, **kwargs)
    attempts = []
    for truncation in DOMAINS:
        calc.truncation = truncation
        started = time.perf_counter()
        try:
            result = calc.run()
        except ValueError as exc:
            attempts.append(
                {
                    "requested_truncation": truncation,
                    "status": "refused",
                    "reason": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            return {"status": "refused", "reason": str(exc), "attempts": attempts}
        attempts.append({"requested_truncation": truncation, "status": "computed", "result": asdict(result)})
        if width_passes(result):
            return {"status": "bounded", "result": asdict(result), "attempts": attempts}
    return {"status": "accuracy_refused", "reason": "width target not reached by N1024", "attempts": attempts}


def selected_cost(result, key):
    """Count auxiliary solves instead of pretending they were free."""
    d = result.get("continuation_diagnostics", {})
    return d.get(
        {
            "states": "total_count_states",
            "words": "total_count_transition_words",
            "bytes": "total_kernel_storage_bytes",
        }[key],
        result[{"states": "states", "words": "transition_words", "bytes": "kernel_storage_bytes"}[key]],
    )


def nested(outer, inner):
    """Floating implementation check; no certified arithmetic implication."""
    for key, (lo, hi) in outer["mean_bounds"].items():
        ilo, ihi = inner["mean_bounds"][key]
        tolerance = 1e-12 + 1e-10 * max(abs(lo), abs(hi), abs(ilo), abs(ihi))
        if ilo < lo - tolerance or ihi > hi + tolerance:
            return False
    return True


def comparison(methods):
    """Save common-domain nesting and selected budget ratios separately."""
    by_domain = {
        mode: {
            attempt["requested_truncation"]: attempt["result"]
            for attempt in methods[mode]["attempts"]
            if attempt["status"] == "computed"
        }
        for mode in MODES
    }
    common = sorted(set.intersection(*(set(rows) for rows in by_domain.values())))
    out = {
        "common_domain_checks": [
            {
                "truncation": n,
                "phase_in_global": nested(by_domain["global"][n], by_domain["phase"][n]),
                "core_in_phase": nested(by_domain["phase"][n], by_domain["core"][n]),
            }
            for n in common
        ]
    }
    if all(methods[mode]["status"] == "bounded" for mode in ("global", "core")):
        old, new = methods["global"]["result"], methods["core"]["result"]
        out.update(
            {
                f"selected_{key}_ratio_global_over_core": selected_cost(old, key) / selected_cost(new, key)
                for key in ("states", "words", "bytes")
                if selected_cost(new, key)
            }
        )
        out["selected_duration_ratio_global_over_core"] = old["duration"] / new["duration"]
    out["baseline_refusal_recovered"] = (
        methods["global"]["status"] != "bounded" and methods["core"]["status"] == "bounded"
    )
    return out


def summarize(rows):
    """All refusals remain in the denominator of the engineering target."""
    ratios = [
        row["comparison"]["selected_states_ratio_global_over_core"]
        for row in rows
        if "selected_states_ratio_global_over_core" in row["comparison"]
    ]
    return {
        "total": len(rows),
        "bounded_by_mode": {mode: sum(row["methods"][mode]["status"] == "bounded" for row in rows) for mode in MODES},
        "baseline_refusals_recovered": sum(row["comparison"]["baseline_refusal_recovered"] for row in rows),
        "twofold_selected_state_reductions": sum(ratio >= 2 for ratio in ratios),
        "paired_selected_state_ratios": ratios,
        "all_common_domain_checks_pass": all(
            check["phase_in_global"] and check["core_in_phase"]
            for row in rows
            for check in row["comparison"]["common_domain_checks"]
        ),
    }


def run():
    """Frozen diagnostic grid; no physical costs fitted to fresh DES."""
    groups = {"original": [], "near_critical": []}
    for group, combinations in (
        ("original", itertools.product((1, 2), (2, 4), range(2), range(2), (0.1, 0.5, 0.8))),
        ("near_critical", itertools.product((2,), (2, 4), range(2), range(2), (0.85, 0.90, 0.95, 0.98))),
    ):
        for cap, budget, law_id, cost_id, load in combinations:
            law, costs = LAWS[law_id], COSTS[cost_id]
            phase = configured(PhaseAwareStabilityCalc, cap, budget, law, costs)
            capacity = phase.run().capacity
            rate = (
                float(load / (np.array([item[2] for item in law]) @ phase.isolated_times()[0]))
                if group == "original"
                else load * capacity
            )
            methods = {mode: refine(mode, cap, budget, law, costs, rate) for mode in MODES}
            row = {
                "batch_cap": cap,
                "token_budget": budget,
                "law": law,
                "costs": costs,
                "load_design": load,
                "capacity": capacity,
                "arrival_rate": rate,
                "methods": methods,
                "comparison": comparison(methods),
            }
            groups[group].append(row)
            print(
                f"{group} B{cap} M{budget} law{law_id} costs{cost_id} load{load}: "
                + ", ".join(f"{mode}={methods[mode]['status']}" for mode in MODES),
                flush=True,
            )
    fresh = []
    for budget, load in itertools.product((2, 4), (0.85, 0.95)):
        law, costs = LAWS[1], COSTS[0]
        capacity = configured(PhaseAwareStabilityCalc, 2, budget, law, costs).run().capacity
        rate = load * capacity
        prediction = next(
            row["methods"]["core"]
            for row in groups["near_critical"]
            if row["token_budget"] == budget
            and row["law"] == law
            and row["costs"] == costs
            and row["load_design"] == load
        )
        samples = []
        for seed in range(97000, 97005):
            rng = np.random.default_rng(seed)
            arrivals = np.cumsum(rng.exponential(1 / rate, 100000))
            marks = np.array([item[:2] for item in law])[rng.choice(2, len(arrivals), p=[0.6, 0.4])]
            sim = PrefillDecodeSim(2, budget)
            sim.set_servers(1, 1, *costs)
            ref = sim.replay(arrivals, *marks.T, warmup_fraction=0.2, record_iterations=False)
            samples.append({"seed": seed, "w": ref.w[0], "v": ref.v[0], "ttft": ref.ttft_mean})
        estimates = {}
        for key in ("w", "v", "ttft"):
            values = np.array([sample[key] for sample in samples])
            mean = float(values.mean())
            half = float(t.ppf(0.975, 4) * values.std(ddof=1) / np.sqrt(5))
            estimates[key] = {"mean": mean, "ci95": [mean - half, mean + half]}
            if prediction["status"] == "bounded":
                point = sum(prediction["result"]["mean_bounds"][key]) / 2
                estimates[key].update(
                    prediction=point,
                    absolute_error=abs(mean - point),
                    relative_error=abs(mean - point) / mean,
                    prediction_inside_ci=mean - half <= point <= mean + half,
                )
        fresh.append(
            {
                "token_budget": budget,
                "saturated_load_design": load,
                "arrival_rate": rate,
                "jobs_per_seed": 100000,
                "warmup_fraction": 0.2,
                "samples": samples,
                "reference": estimates,
                "analytic_status": prediction["status"],
            }
        )
        print(f"Fresh M{budget} load{load}: {prediction['status']}", flush=True)
    summaries = {group: summarize(rows) for group, rows in groups.items()}
    return {
        "protocol": "EPIC-097 frozen before computation; K8; floating means, rational phase drift",
        "domains": DOMAINS,
        "core_truncation": 8,
        "budgets": {"states": 20000, "transition_words": 2000000},
        "memory_metric": "sum of CSR kernel data/index/indptr bytes; NOT peak RSS",
        **groups,
        "fresh_des": fresh,
        "summary": summaries,
        "engineering_target_pass": summaries["near_critical"]["twofold_selected_state_reductions"] >= 16,
    }


def pack_results(payload):
    """Losslessly deduplicate repeated certificates; retain per-call timing."""
    registry, identifiers = {}, {}

    def visit(value):
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        out = {key: visit(item) for key, item in value.items() if key != "phase_certificate"}
        if "phase_certificate" in value:
            certificate = dict(value["phase_certificate"])
            duration = certificate.pop("duration")
            signature = json.dumps(certificate, sort_keys=True, separators=(",", ":"))
            if signature not in identifiers:
                identifier = f"certificate_{len(registry):03d}"
                identifiers[signature] = identifier
                registry[identifier] = certificate
            out["phase_certificate_id"] = identifiers[signature]
            out["phase_certificate_duration"] = duration
        return out

    packed = visit(payload)
    packed["certificate_encoding"] = "shared certificate without duration; each result retains its call duration"
    packed["phase_certificates"] = registry
    return packed


def unpack_results(packed):
    """Restore all original result dictionaries for auditing and reuse."""
    if "phase_certificates" not in packed:
        return packed
    registry = packed["phase_certificates"]

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        out = {
            key: visit(item)
            for key, item in value.items()
            if key not in ("phase_certificate_id", "phase_certificate_duration")
        }
        if "phase_certificate_id" in value:
            out["phase_certificate"] = {
                **registry[value["phase_certificate_id"]],
                "duration": value["phase_certificate_duration"],
            }
        return out

    return visit(
        {key: value for key, value in packed.items() if key not in ("phase_certificates", "certificate_encoding")}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/research/core-aware-continuation-results-2026.json"))
    parser.add_argument(
        "--pack-existing", action="store_true", help="Lossless storage-only rewrite of an existing run; no experiments"
    )
    args = parser.parse_args()
    experiment_payload = unpack_results(json.loads(args.output.read_text())) if args.pack_existing else run()
    packed_payload = pack_results(experiment_payload)
    assert unpack_results(packed_payload) == json.loads(json.dumps(experiment_payload))
    args.output.write_text(json.dumps(packed_payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
