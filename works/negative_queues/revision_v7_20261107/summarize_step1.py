"""Save pre-refinement RCS estimates; do not choose grids using simulation."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def extrapolate(rows, ks, power=1):
    x = np.asarray(ks, dtype=float) ** (-power)
    weights = np.array([np.prod([-x[j] / (x[i] - x[j]) for j in range(len(x)) if j != i]) for i in range(len(x))])
    result = {key: (weights @ np.array([r[key] for r in rows])).tolist() for key in ["w", "v"]}
    result.update(
        sw=float(np.sqrt(result["w"][1] - result["w"][0] ** 2)),
        sv=float(np.sqrt(result["v"][1] - result["v"][0] ** 2)),
        q=1.0,
    )
    return result, weights.tolist()


def main():
    independent = json.loads((ROOT / "step1-independent-results.json").read_text())
    old = json.loads((ROOT.parent / "refinement_20261107/experiment-results.json").read_text())["results"]
    results = []
    for cv in [0.8, 1.2]:
        ks = [2, 4, 8]
        rows = [json.loads((ROOT / "step1" / f"fixed-cv{cv}-j4-k{k}-n100-b24.json").read_text()) for k in ks]
        estimate, weights = extrapolate(rows, ks)
        reference = next(r for r in independent if r["cv"] == cv)["estimates"]
        comparison = {}
        for key in ["w", "v", "sw", "sv"]:
            value = estimate[key][0] if key in ["w", "v"] else estimate[key]
            comparison[key] = dict(
                mean=reference[key]["mean"],
                half_ci95=reference[key]["half_ci95"],
                difference_percent=100 * (value / reference[key]["mean"] - 1),
                inside_ci95=bool(abs(value - reference[key]["mean"]) <= reference[key]["half_ci95"]),
            )
        results.append(
            dict(
                cv=cv,
                scenario="fixed",
                mode="rcs",
                calculation=estimate,
                classes=4,
                stages=ks,
                weights=weights,
                comparison=comparison,
            )
        )
        r = json.loads((ROOT / "step1" / f"remove-cv{cv}-j0-k0-n160-b80.json").read_text())
        results.append(dict(cv=cv, scenario="remove", mode="rcs", calculation=r))
    (ROOT / "step1-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([r for r in results if r["scenario"] == "fixed"], indent=2))


if __name__ == "__main__":
    main()
