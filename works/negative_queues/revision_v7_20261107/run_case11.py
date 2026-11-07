"""Reproduce the selected large case at the original Krylov tolerance."""

import json
from pathlib import Path
import works.negative_queues.revision_v7_20261107.ph_queue as engine
from works.negative_queues.revision_v7_20261107.preconditioned_stationary_batch import solve
from works.negative_queues.revision_v7_20261107.stationary_check import calculate
from works.negative_queues.revision_v7_20261107.study_cases import cases


def main():
    path = Path(__file__).resolve().parent / "case11-k12-relaxed.json"
    if path.exists():
        print("Using saved result:", path)
        return
    case = cases()[11]
    engine.solve_jump_chain = solve
    result = calculate(
        case["cv"], 3, 12, 240, n=case["n"], lam=case["lam"], delta=case["delta"], tolerance=1e-10, progress=True
    )
    result["case"] = case
    result["stages"] = 12
    result["solver"] = "independent assembly; batched block ILU; Krylov rtol=1e-12"
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
