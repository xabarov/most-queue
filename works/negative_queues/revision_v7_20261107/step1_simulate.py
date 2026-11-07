"""Additional independent RCS retained-work runs, specified before comparison."""

import concurrent.futures
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from works.negative_queues.refinement_20261107.diagnose import independent_simulation

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "step1-replications.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["seed"] for r in rows}
    tasks = [
        (cv, "rcs", "fixed", False, 202611072000 + 100 * i + j, 500000)
        for i, cv in enumerate([0.8, 1.2])
        for j in range(8)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(independent_simulation, task) for task in tasks if task[-2] not in done]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(f"{len(rows)}/16 cv={row['cv']} seed={row['seed']}", flush=True)
    results = []
    for cv in [0.8, 1.2]:
        group = [r for r in rows if r["cv"] == cv]
        estimates = {}
        for key in ["w", "v", "sw", "sv", "little_w", "little_v", "covariance_ws"]:
            values = np.array([r[key] for r in group])
            estimates[key] = dict(
                mean=float(values.mean()), half_ci95=float(t.ppf(0.975, 7) * values.std(ddof=1) / np.sqrt(8))
            )
        results.append(dict(cv=cv, mode="rcs", scenario="fixed", estimates=estimates))
    (ROOT / "step1-independent-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
