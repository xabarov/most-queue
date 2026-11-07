"""Compare PH approximations without using simulation to choose a fit."""

import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.stats import gamma
from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph, positive_h2_equivalent, transform
from works.negative_queues.revision_v7_20261107.step4_calculate import erlang_cases

ROOT = Path(__file__).resolve().parent


def read(name, j, n=240, b=80):
    return json.loads((ROOT / "step4" / f"{name}-j{j}-n{n}-b{b}.json").read_text())


def values(row):
    return {k: row[k][0] if k in ["w", "v"] else row[k] for k in ["w", "v", "sw", "sv", "q"]}


def main():
    h = H2Distribution.get_params_clx(gamma_moments_by_mean_and_cv(2.1, 0.8))
    cox = positive_h2_equivalent([h.p1, 1 - h.p1], [h.mu1, h.mu2])
    grids = []
    shape = 1 / 0.8**2
    theta = 2.1 / shape
    times = np.r_[0, np.geomspace(1e-4, 40, 250)]
    s_values = np.r_[0, np.logspace(-4, 3, 250)]
    for j in [2, 4, 8]:
        p = gamma_ph(2.1, 0.8, j)
        cdf = np.array([1 - p.alpha @ expm(p.generator * x) @ np.ones(p.size) for x in times])
        grids.append(
            dict(
                quadrature=j,
                phases=p.size,
                max_cdf_error_on_grid=float(max(abs(cdf - gamma.cdf(times, shape, scale=theta)))),
                max_transform_error_on_grid=float(
                    max(abs(transform(p, s) - (1 + theta * s) ** -shape) for s in s_values)
                ),
                moments=p.moments(5),
            )
        )
    approximations = []
    old = json.loads((ROOT.parent / "refinement_20261107/experiment-results.json").read_text())["results"]
    for action in ["remove", "restart"]:
        for mode in ["rcs", "disaster"]:
            name = f"base-{action}-{mode}-cv0.8"
            rows = [read(name, j, 160, 40) for j in [2, 4, 8]]
            selected = read(name, 8)
            original = next(r for r in old if r["cv"] == 0.8 and r["mode"] == mode and r["scenario"] == action)
            approximations.append(
                dict(
                    action=action,
                    mode=mode,
                    estimates={str(j): values(r) for j, r in zip([2, 4, 8], rows)},
                    selected=values(selected),
                    old_h2=values(original["calculation"]),
                    j4_to_j8_percent={
                        k: 100 * (values(rows[2])[k] / values(rows[1])[k] - 1) for k in ["w", "v", "sw", "sv"]
                    },
                    cutoff_changes={k: (np.array(selected[k]) - rows[2][k]).tolist() for k in ["w", "v"]},
                    simulation=original["simulation"],
                )
            )
    extended = []
    # The non-retained-work cases are complete independently of the expensive
    # retained-work grids in step 3; use their saved raw results directly.
    study = []
    for ref in json.loads((ROOT / "step3-independent-results.json").read_text()):
        case = ref["case"]
        if case["action"] == "fixed":
            continue
        original = json.loads((ROOT / "step3" / f"{case['id']}-k0-n240-b80.json").read_text())
        comparison = {}
        for k, val in values(original).items():
            est = ref["estimates"][k]
            comparison[k] = dict(
                calculation=val,
                simulation=est["mean"],
                half_ci95=est["half_ci95"],
                difference_percent=100 * (val / est["mean"] - 1),
                inside_ci95=bool(abs(val - est["mean"]) <= est["half_ci95"] + 1e-12),
            )
        study.append(dict(case=case, comparison=comparison))
    for row in study:
        case = row["case"]
        if case["cv"] != 0.8 or case["action"] == "fixed":
            continue
        selected = read(case["id"], 8)
        comparison = {}
        for k, val in values(selected).items():
            ref = row["comparison"][k]
            comparison[k] = dict(
                calculation=val,
                simulation=ref["simulation"],
                half_ci95=ref["half_ci95"],
                difference_percent=100 * (val / ref["simulation"] - 1),
                inside_ci95=bool(abs(val - ref["simulation"]) <= ref["half_ci95"] + 1e-12),
            )
        extended.append(
            dict(case=case, comparison=comparison, calculation=selected, old_h2_comparison=row["comparison"])
        )
    exact = []
    simulations = json.loads((ROOT / "step4-independent-results.json").read_text())
    for case in erlang_cases():
        selected = read(case["id"], 4)
        small = read(case["id"], 4, 160, 40)
        ref = next(r for r in simulations if r["case"]["id"] == case["id"])["estimates"]
        comparison = {}
        for k, val in values(selected).items():
            comparison[k] = dict(
                calculation=val,
                simulation=ref[k]["mean"],
                half_ci95=ref[k]["half_ci95"],
                difference_percent=100 * (val / ref[k]["mean"] - 1),
                inside_ci95=bool(abs(val - ref[k]["mean"]) <= ref[k]["half_ci95"] + 1e-12),
            )
        exact.append(
            dict(
                case=case,
                calculation=selected,
                comparison=comparison,
                cutoff_changes={k: (np.array(selected[k]) - small[k]).tolist() for k in ["w", "v"]},
            )
        )
    payload = dict(
        coxian=dict(alpha=cox.alpha.tolist(), S=cox.generator.tolist()),
        input_law_checks=grids,
        baseline=approximations,
        extended=extended,
        exact_erlang=exact,
    )
    (ROOT / "step4-results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Input laws:", grids)
    for r in approximations:
        print(r["action"], r["mode"], r["j4_to_j8_percent"], r["cutoff_changes"])
    for r in extended + exact:
        print(
            r["case"]["id"],
            {k: round(v["difference_percent"], 3) for k, v in r["comparison"].items()},
            "misses",
            [k for k, v in r["comparison"].items() if not v["inside_ci95"]],
        )


if __name__ == "__main__":
    main()
