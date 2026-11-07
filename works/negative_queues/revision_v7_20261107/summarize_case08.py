"""Retain the primary study result and report subsequent grid sensitivity."""

import json
from pathlib import Path
from works.negative_queues.revision_v7_20261107.summarize_step1 import extrapolate

ROOT = Path(__file__).resolve().parent


def main():
    original = next(r for r in json.loads((ROOT / "step3-results.json").read_text()) if r["case"]["id"] == "case08")
    rows = []
    for j in [3, 4, 5]:
        for ks in [[4, 8, 12], [8, 12, 16]]:
            raw = []
            for k in ks:
                path = (
                    ROOT / "step3" / f"case08-k{k}-n120-b32.json"
                    if j == 3 and k != 16
                    else ROOT / "case08_refinement" / f"j{j}-k{k}.json"
                )
                raw.append(json.loads(path.read_text()))
            calc, weights = extrapolate(raw, ks)
            comparison = {}
            for key in ["w", "v", "sw", "sv"]:
                ref = original["comparison"][key]
                x = calc[key][0] if key in ["w", "v"] else calc[key]
                comparison[key] = dict(
                    calculation=x,
                    simulation=ref["simulation"],
                    half_ci95=ref["half_ci95"],
                    difference_percent=100 * (x / ref["simulation"] - 1),
                    inside_ci95=abs(x - ref["simulation"]) <= ref["half_ci95"],
                    change_from_primary_percent=100 * (x / ref["calculation"] - 1),
                )
            rows.append(dict(classes=j, stages=ks, weights=weights, calculation=calc, comparison=comparison))
    (ROOT / "case08-refinement-results.json").write_text(
        json.dumps(dict(primary_unchanged=True, original_case="case08", results=rows), indent=2) + "\n"
    )
    for row in rows:
        print(row["classes"], row["stages"], {k: v for k, v in row["comparison"].items() if k in ["w", "v"]})


if __name__ == "__main__":
    main()
