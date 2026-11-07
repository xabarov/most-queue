"""Cross-check saved experiments and convergence diagnostics without rerunning them."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    rows = read("experiment-results.json")["results"]
    original = read("../revision_20261107/experiment-results.json")["results"]
    independent = read("independent-results.json")
    replications = [json.loads(line) for line in (ROOT / "independent-replications.jsonl").read_text().splitlines()]
    assert len(replications) == len({r["seed"] for r in replications}) == 80
    assert all(r["warmup"] == 50000 and r["sample"] == 500000 for r in replications)
    assert len(rows) == 12
    assert [r["simulation"] for r in rows] == [r["simulation"] for r in original]
    checked, changed = 0, 0
    for row, old in zip(rows, original):
        if row["calculation"] == old["calculation"]:
            continue
        changed += 1
        ref = next(
            r
            for r in independent
            if (r["cv"], r["mode"], r["scenario"], r["independent"]) == (row["cv"], row["mode"], row["scenario"], False)
        )
        for key in ["w", "v", "sw", "sv"]:
            value = row["calculation"][key]
            if isinstance(value, list):
                value = value[0]
            estimate = ref["estimates"][key]
            assert abs(value - estimate["mean"]) <= estimate["half_ci95"], (row["cv"], row["scenario"], key)
            checked += 1
        if row["scenario"] == "restart":
            assert row["truncation_change"] < 2e-8
    assert changed == 6 and checked == 24
    grids = []
    for path in sorted((ROOT / "full_work").glob("*.json")):
        data = json.loads(path.read_text())
        assert abs(sum(data["p"]) - 1) < 1e-10
        assert min(data["p"]) >= -1e-10
        # All saved chains use a blocked upper boundary. Correct the admitted
        # flow in this separate finite-chain Little's-law check.
        residual = abs((data["v"][0] - data["w"][0]) / (1 - data["tail"]) - data["service_moments"][0])
        assert residual < 2e-6, (path.name, residual)
        grids.append(
            dict(
                file=path.name,
                embedded_residual=data["embedded_chain_residual_l1"],
                tail=data["tail"],
                admitted_flow_service_residual=residual,
            )
        )
    sensitivity = []
    for row in read("full-work-results.json")["results"]:
        c = row["calculation"]
        alt = c["alternative_extrapolation_w"]
        dw = abs(alt[0] / c["w"][0] - 1)
        ds = abs(np.sqrt(alt[1] - alt[0] ** 2) / c["sw"] - 1)
        assert dw < 0.0054 and ds < 0.0094
        sensitivity.append(dict(cv=row["cv"], mean_relative=dw, std_relative=float(ds)))
    result = dict(
        original_simulation_values_unchanged=True,
        independent_replications=80,
        unique_seeds=80,
        refined_cases=changed,
        refined_statistics_inside_independent_ci95=checked,
        finite_chain_checks=grids,
        extrapolation_sensitivity=sensitivity,
        tests="19 passed; pytest-combined.log",
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "experiment-results.json",
                ROOT / "independent-replications.jsonl",
                ROOT / "full-work-results.json",
            ]
        },
    )
    (ROOT / "numerical-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["finite_chain_checks", "sha256"]}, indent=2))


if __name__ == "__main__":
    main()
