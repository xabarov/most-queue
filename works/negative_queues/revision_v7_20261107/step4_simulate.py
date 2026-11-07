"""Independent validation where the positive PH representation is exact."""

import concurrent.futures
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from works.negative_queues.revision_v7_20261107.simulation import run
from works.negative_queues.revision_v7_20261107.step4_calculate import erlang_cases

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "step4-replications.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["seed"] for r in rows}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(run, c, 202611075000 + 100 * i + j)
            for i, c in enumerate(erlang_cases())
            for j in range(8)
            if 202611075000 + 100 * i + j not in done
        ]
        for f in concurrent.futures.as_completed(futures):
            row = f.result()
            rows.append(row)
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(f"{len(rows)}/16 {row['case']['id']} seed={row['seed']}", flush=True)
    results = []
    for case in erlang_cases():
        group = [r for r in rows if r["case"]["id"] == case["id"]]
        assert len(group) == 8
        estimates = {}
        for key in ["w", "v", "sw", "sv", "q"]:
            a = np.array([r[key] for r in group])
            estimates[key] = dict(mean=float(a.mean()), half_ci95=float(t.ppf(0.975, 7) * a.std(ddof=1) / np.sqrt(8)))
        results.append(dict(case=case, estimates=estimates))
    (ROOT / "step4-independent-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
