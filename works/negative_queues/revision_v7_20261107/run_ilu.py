"""Record an independent solution of a large shared-restart grid."""

import argparse
import json
from pathlib import Path
import works.negative_queues.revision_v7_20261107.ph_queue as engine
from works.negative_queues.revision_v7_20261107.preconditioned_stationary_ilu import solve
from works.negative_queues.revision_v7_20261107.stationary_check import calculate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--cv", type=float, default=1.2)
    args = parser.parse_args()
    engine.solve_jump_chain = solve
    result = calculate(args.cv, 4, 16, levels=args.levels, tolerance=1e-12, progress=True)
    result["solver"] = "incomplete block-Jacobi preconditioned embedded-chain equations; independent assembly"
    path = Path(__file__).resolve().parent / f"ilu-cv{args.cv}-j4-k16-n{args.levels}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ["w", "v", "seconds", "balance_residual", "stationary_products"]}, flush=True)
