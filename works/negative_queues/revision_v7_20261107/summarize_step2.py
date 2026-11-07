"""Compare predetermined refinement sequences and retain numerical sensitivity."""

import json
from pathlib import Path
import numpy as np

from works.negative_queues.revision_v7_20261107.summarize_step1 import extrapolate

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "refinement_20261107"


def point(mode, cv, j, k, levels=100, behind=24):
    independent = ROOT / f"ilu-cv{cv}-j{j}-k{k}-n{levels}.json"
    if mode == "disaster" and independent.exists():
        return json.loads(independent.read_text())
    path = ROOT / "step2" / f'{mode}-cv{cv}-j{j}-k{k}-n{levels}-b{behind if mode=="rcs" else 0}.json'
    if path.exists():
        return json.loads(path.read_text())
    path = (
        (ROOT / "step1" / f"fixed-cv{cv}-j{j}-k{k}-n{levels}-b{behind}.json")
        if mode == "rcs"
        else (OLD / "full_work" / f"cv{cv}-j{j}-k{k}-n{levels}.json")
    )
    return json.loads(path.read_text())


def estimate(mode, cv, j, ks, power=1):
    rows = [point(mode, cv, j, k) for k in ks]
    result, weights = extrapolate(rows, ks, power)
    if mode == "disaster":
        source = json.loads((OLD / "full-work-results.json").read_text())["results"]
        t = next(r for r in source if r["cv"] == cv)["calculation"]["service_moments"]
        result["v"] = [result["w"][0] + t[0], result["w"][1] + 2 * result["w"][0] * t[0] + t[1]]
        result["sv"] = float(np.sqrt(result["v"][1] - result["v"][0] ** 2))
    return result, weights


def main(allow_pending=False):
    results = []
    for mode in ["rcs", "disaster"]:
        independent = json.loads(
            (
                (ROOT / "step1-independent-results.json") if mode == "rcs" else (OLD / "independent-results.json")
            ).read_text()
        )
        for cv in [0.8, 1.2]:
            selected, weights = estimate(mode, cv, 4, [8, 12, 16])
            coarser, _ = estimate(mode, cv, 4, [4, 8, 12])
            fewer, _ = estimate(mode, cv, 3, [8, 12, 16])
            alternative, _ = estimate(mode, cv, 4, [8, 12, 16], 0.5)
            ref = next(
                r
                for r in independent
                if r["cv"] == cv and r["mode"] == mode and r["scenario"] == "fixed" and not r.get("independent", False)
            )["estimates"]
            comparison = {}
            sensitivities = {}
            for key in ["w", "v", "sw", "sv"]:
                value = selected[key][0] if key in ["w", "v"] else selected[key]
                comparison[key] = dict(
                    mean=ref[key]["mean"],
                    half_ci95=ref[key]["half_ci95"],
                    difference_percent=100 * (value / ref[key]["mean"] - 1),
                    inside_ci95=bool(abs(value - ref[key]["mean"]) <= ref[key]["half_ci95"]),
                )
                sensitivities[key] = {
                    name: 100 * ((a[key][0] if key in ["w", "v"] else a[key]) / value - 1)
                    for name, a in [("coarser", coarser), ("fewer_classes", fewer), ("sqrt_extrapolation", alternative)]
                }
            results.append(
                dict(
                    cv=cv,
                    scenario="fixed",
                    mode=mode,
                    calculation=selected,
                    classes=4,
                    stages=[8, 12, 16],
                    weights=weights,
                    coarser=coarser,
                    fewer_classes=fewer,
                    alternative=alternative,
                    comparison=comparison,
                    sensitivity_percent=sensitivities,
                )
            )
    cutoffs = {}
    base = point("rcs", 1.2, 4, 8)
    for name, r in [("levels", point("rcs", 1.2, 4, 8, levels=160)), ("behind", point("rcs", 1.2, 4, 8, behind=48))]:
        cutoffs[name] = {key: (np.array(r[key]) - base[key]).tolist() for key in ["w", "v"]}
    base = point("disaster", 1.2, 4, 16)
    pending = []
    try:
        extra = point("disaster", 1.2, 4, 16, levels=160)
        cutoffs["shared_levels"] = {key: (np.array(extra[key]) - base[key]).tolist() for key in ["w", "v"]}
    except FileNotFoundError:
        if not allow_pending:
            raise
        pending.append("shared_levels_J4_K16_N160")
    quadrature = []
    for mode, k in [("rcs", 4), ("rcs", 8), ("disaster", 8)]:
        a, b, c = [point(mode, 0.8, j, k) for j in [3, 4, 5]]
        quadrature.append(
            dict(
                mode=mode,
                cv=0.8,
                stages=k,
                w_by_classes={str(j): row["w"] for j, row in zip([3, 4, 5], [a, b, c])},
                j3_to_j4_percent=100 * (b["w"][0] / a["w"][0] - 1),
                j4_to_j5_percent=100 * (c["w"][0] / b["w"][0] - 1),
            )
        )
    payload = dict(results=results, cutoff_changes=cutoffs, extra_quadrature_checks=quadrature, pending_checks=pending)
    (ROOT / "step2-results.json").write_text(json.dumps(payload, indent=2) + "\n")
    for r in results:
        print(r["mode"], r["cv"], r["calculation"], r["sensitivity_percent"], r["comparison"])
    print("Cutoff changes:", cutoffs)
    print("Pending checks:", pending)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending", action="store_true")
    main(parser.parse_args().allow_pending)
