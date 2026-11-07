"""Refine work quadrature, progress phases and queue truncation independently."""

import concurrent.futures
import json
from pathlib import Path

from works.negative_queues.refinement_20261107.fixed_work_phase import calculate

ROOT = Path(__file__).resolve().parent / "full_work"


def run_case(case):
    cv, classes, stages, levels = case
    path = ROOT / f"cv{cv}-j{classes}-k{stages}-n{levels}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = calculate(
        cv=cv, classes=classes, stages=stages, levels=levels, tolerance=1e-8, max_iter=100000, method="krylov"
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    cases = [(1.2, j, k, 100) for j in [3, 4, 5] for k in [2, 4, 8]]
    cases += [(1.2, 3, 12, 100), (1.2, 4, 12, 100), (1.2, 4, 8, 160)]
    cases += [(0.8, j, k, 100) for j in [3, 4] for k in [2, 4, 8, 12]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_case, case) for case in cases]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(
                {k: result[k] for k in ["cv", "classes", "stages", "levels", "w", "v", "sw", "sv", "tail", "seconds"]},
                flush=True,
            )
