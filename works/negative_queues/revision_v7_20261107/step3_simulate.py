"""Run all prespecified simulations, retaining every replication."""

import concurrent.futures
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from works.negative_queues.revision_v7_20261107.simulation import run
from works.negative_queues.revision_v7_20261107.study_cases import cases

ROOT = Path(__file__).resolve().parent


def summarize(rows):
    results = []
    for case in cases():
        group = [r for r in rows if r["case"]["id"] == case["id"]]
        assert len(group) == 8
        estimates = {}
        for key in ["w", "v", "sw", "sv", "q", "little_w", "little_v"]:
            values = np.array([r[key] for r in group])
            estimates[key] = dict(
                mean=float(values.mean()), half_ci95=float(t.ppf(0.975, 7) * values.std(ddof=1) / np.sqrt(8))
            )
        differences = np.array([np.array(r["half_means"][1]) - r["half_means"][0] for r in group])
        results.append(
            dict(
                case=case,
                estimates=estimates,
                half_difference_mean=differences.mean(axis=0).tolist(),
                half_difference_ci95=(t.ppf(0.975, 7) * differences.std(axis=0, ddof=1) / np.sqrt(8)).tolist(),
                maximum_observed_queue=max(r["max_queue"] for r in group),
            )
        )
    (ROOT / "step3-independent-results.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    path = ROOT / "step3-replications.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["seed"] for r in rows}
    specifications = cases()
    (ROOT / "step3-cases.json").write_text(json.dumps(specifications, indent=2) + "\n")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run, case, 202611073000 + 100 * i + j)
            for i, case in enumerate(specifications)
            for j in range(8)
            if 202611073000 + 100 * i + j not in done
        ]
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            rows.append(r)
            with path.open("a") as out:
                out.write(json.dumps(r) + "\n")
            print(f"{len(rows)}/96 {r['case']['id']} seed={r['seed']}", flush=True)
    summarize(rows)


if __name__ == "__main__":
    main()
