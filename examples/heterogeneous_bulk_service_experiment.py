"""Reproduce Л8 tables, diagnose their priority labels, refine physical CTMC and DES.

Run from repository root:
  OPENBLAS_NUM_THREADS=1 python -m examples.heterogeneous_bulk_service_experiment
JSON excludes timing and is deterministic for fixed source/platform and seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from examples.heterogeneous_bulk_service_oracle import physical_oracle
from most_queue.sim.heterogeneous_bulk_service import BulkServiceHeterogeneousSim
from most_queue.theory.batch import BulkServiceHeterogeneousCalc

# Published Table 2, lower-left diagonal entries as labelled A-priority.
TABLE_QUEUE = [
    [4.074, 3.604, 3.066, 2.657, 2.506],
    [3.953, 3.454, 2.928, 2.548, 2.426],
    [3.791, 3.258, 2.785, 2.445, 2.360],
    [3.598, 3.066, 2.526, 2.389, 2.337],
    [3.391, 2.875, 2.378, 2.148, 2.382],
    [3.193, 2.707, 2.266, 2.094, 2.230],
    [3.074, 2.625, 2.234, 2.099, 2.248],
    [3.031, 2.622, 2.266, 2.147, 2.291],
    [3.062, 2.682, 2.348, 2.225, 2.350],
]
TABLE_QUEUE_B_DIAGONAL = [4.072, 3.430, 2.692, 2.236, 2.247]
TABLE_ZERO = [
    [0.077, 0.124, 0.155, 0.165, 0.157],
    [0.086, 0.146, 0.171, 0.178, 0.168],
    [0.094, 0.153, 0.194, 0.188, 0.176],
    [0.100, 0.157, 0.194, 0.205, 0.180],
    [0.103, 0.158, 0.193, 0.201, 0.188],
    [0.103, 0.157, 0.190, 0.196, 0.184],
    [0.100, 0.153, 0.185, 0.191, 0.180],
    [0.095, 0.146, 0.178, 0.186, 0.176],
    [0.089, 0.139, 0.171, 0.180, 0.173],
]
TABLE_ZERO_B_DIAGONAL = [0.076, 0.143, 0.186, 0.195, 0.182]


def calculator(counts, rates, thresholds, capacities, l):
    """Configure the public calculator API."""
    m = BulkServiceHeterogeneousCalc(counts)
    m.set_sources(l)
    m.set_servers(rates, thresholds, capacities)
    return m


def experiment():
    """Return source reproduction and independent validation, with provenance."""
    tables = []
    for tb in range(1, 10):
        for ta in range(1, 6):
            for priority in ["A", "B"] if ta == tb else ["threshold"]:
                if priority == "B":
                    m = calculator([2, 4], [0.2, 0.4], [tb, ta], [9, 5], 6)
                    printed_q = TABLE_QUEUE_B_DIAGONAL[ta - 1]
                    corrected_q = TABLE_QUEUE[tb - 1][ta - 1]
                    printed_zero = TABLE_ZERO_B_DIAGONAL[ta - 1]
                else:
                    m = calculator([4, 2], [0.4, 0.2], [ta, tb], [5, 9], 6)
                    printed_q = TABLE_QUEUE[tb - 1][ta - 1]
                    corrected_q = TABLE_QUEUE_B_DIAGONAL[ta - 1] if ta == tb else printed_q
                    printed_zero = TABLE_ZERO[tb - 1][ta - 1]
                r = m.run()
                assert abs(r.mean_queue - corrected_q) <= 0.00051
                assert abs(r.zero_wait_probability - printed_zero) <= 0.00051
                tables.append(
                    {
                        "TA": ta,
                        "TB": tb,
                        "priority": priority,
                        "mean_queue": r.mean_queue,
                        "zero_wait_probability": r.zero_wait_probability,
                        "printed_mean_queue": printed_q,
                        "printed_zero_wait": printed_zero,
                        "queue_delta_literal": r.mean_queue - printed_q,
                        "queue_delta_swapped_diagonal": r.mean_queue - corrected_q,
                        "w": r.w,
                        "v": r.v,
                    }
                )
    oracle = []
    for label, counts, rates, thresholds, capacities, l in [
        ("source_tie_A", [4, 2], [0.4, 0.2], [4, 4], [5, 9], 6),
        ("source_tie_B", [2, 4], [0.2, 0.4], [4, 4], [9, 5], 6),
        ("three_types", [1, 1, 1], [1.3, 0.6, 0.9], [2, 2, 4], [2, 3, 5], 2.2),
    ]:
        m = calculator(counts, rates, thresholds, capacities, l)
        exact = m.run()
        for cap in (30, 70, 110):
            o = physical_oracle(counts, rates, thresholds, capacities, l, cap=cap)
            oracle.append(
                {
                    "label": label,
                    "cap": cap,
                    **o,
                    "queue_error": o["mean_queue"] - exact.mean_queue,
                    "waiting_moment_errors": (np.array(o["w"]) - exact.w).tolist(),
                    "response_moment_errors": (np.array(o["v"]) - exact.v).tolist(),
                }
            )
    settings = [
        ("source_1_1", [4, 2], [0.4, 0.2], [1, 1], [5, 9], 6),
        ("source_4_4_A", [4, 2], [0.4, 0.2], [4, 4], [5, 9], 6),
        ("source_4_4_B", [2, 4], [0.2, 0.4], [4, 4], [9, 5], 6),
        ("source_4_6", [4, 2], [0.4, 0.2], [4, 6], [5, 9], 6),
    ]
    for load in (0.2, 0.6, 0.85):
        for thresholds in ([1, 1, 1], [2, 2, 4]):
            settings.append(
                (
                    f"three_types_{load}_{thresholds}",
                    [1, 1, 1],
                    [1.3, 0.6, 0.9],
                    thresholds,
                    [2, 3, 5],
                    load * (1.3 * 2 + 0.6 * 3 + 0.9 * 5),
                )
            )
    simulations = []
    times = [0.0, 1.0, 3.0]
    for label, counts, rates, thresholds, capacities, l in settings:
        m = calculator(counts, rates, thresholds, capacities, l)
        r = m.run()
        expected = np.array(
            [
                r.w[0],
                r.v[0],
                *r.customer_throughputs,
                *(m.tail(t) for t in times),
                *(m.tail(t, response=True) for t in times),
            ]
        )
        observations = []
        counters = []
        for seed in range(4):
            sim = BulkServiceHeterogeneousSim(counts, seed=81000 + seed)
            sim.set_sources(l)
            sim.set_servers(rates, thresholds, capacities)
            s = sim.run(12000, warmup=1000, tail_times=times)
            assert s.arrived == s.completed + s.unfinished
            wait_error = s.total_queue_area - s.started_wait_exposure - s.unfinished_wait_exposure
            residence_error = s.total_population_area - s.completed_residence - s.unfinished_residence
            assert max(abs(wait_error), abs(residence_error)) < 1e-6
            observations.append(
                [s.waiting_proxy, s.response_proxy, *s.customer_throughputs, *s.waiting_tails, *s.response_tails]
            )
            counters.append(
                {
                    "seed": 81000 + seed,
                    "arrived": s.arrived,
                    "completed": s.completed,
                    "unfinished": s.unfinished,
                    "cohort_size": s.cohort_size,
                    "cohort_unstarted": s.cohort_unstarted,
                    "cohort_unfinished": s.cohort_unfinished,
                    "queue_area_residual": wait_error,
                    "population_area_residual": residence_error,
                }
            )
        samples = np.array(observations)
        se = samples.std(axis=0, ddof=1) / 2
        difference = np.abs(samples.mean(axis=0) - expected)
        allowed = 4.5 * se + 0.02
        assert np.all(difference <= allowed), (label, difference, allowed)
        simulations.append(
            {
                "label": label,
                "counts": counts,
                "rates": rates,
                "thresholds": thresholds,
                "capacities": capacities,
                "arrival_rate": l,
                "tail_times": times,
                "expected": expected.tolist(),
                "samples": observations,
                "standard_errors": se.tolist(),
                "absolute_errors": difference.tolist(),
                "allowances": allowed.tolist(),
                "counters": counters,
            }
        )
    return {
        "source": {
            "doi": "10.1007/s11134-020-09654-y",
            "read": "complete §§1–6 + references",
            "pdf_sha256": "00694b64aeb3f770d98bcdb4327c6fe20542d758fb91b6659410a636d3e68e0b",
            "date": "2026-10-11",
            "not_gpu_calibration": True,
        },
        "tables": tables,
        "physical_oracles": oracle,
        "simulations": simulations,
        "des_contract": {
            "horizon": 12000,
            "warmup": 1000,
            "replications": 4,
            "cohort_end": 6500,
            "no_drain": True,
            "completed_only_moments": False,
            "metric_order": "E[W],E[V],customer flow by input type,W tails,V tails",
        },
        "minimum_queue": min(tables, key=lambda x: x["mean_queue"]),
        "maximum_zero_wait": max(tables, key=lambda x: x["zero_wait_probability"]),
    }


def main():
    """Write the deterministic JSON report requested by this experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research/heterogeneous-bulk-service-results-2026.json")
    )
    args = parser.parse_args()
    result = experiment()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"{len(result['tables'])} source rows, {len(result['physical_oracles'])} CTMC refinements, "
        f"{len(result['simulations'])} DES settings x 4 seeds -> {args.output}"
    )


if __name__ == "__main__":
    main()
