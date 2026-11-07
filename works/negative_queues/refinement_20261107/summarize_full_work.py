"""Extrapolate raw moments using grid coordinates only; compare independent data."""

import json
from pathlib import Path

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.utils.restarts import effective_service_moments

ROOT = Path(__file__).resolve().parent


def point(cv, j, k, n=100):
    return json.loads((ROOT / "full_work" / f"cv{cv}-j{j}-k{k}-n{n}.json").read_text())


def extrapolate(cv, j, ks, power=1):
    x = np.array(ks, float) ** (-power)
    weights = np.array([np.prod([-x[l] / (x[i] - x[l]) for l in range(len(x)) if l != i]) for i in range(len(x))])
    rows = [point(cv, j, k) for k in ks]
    moments = weights @ np.array([r["w"] for r in rows])
    assert moments[0] > 0 and moments[1] > moments[0] ** 2
    return moments, weights


def main():
    original = json.loads((ROOT.parent / "revision_20261107/experiment-results.json").read_text())["results"]
    independent = json.loads((ROOT / "independent-results.json").read_text())
    results = []
    for cv in [0.8, 1.2]:
        ks = [4, 8, 12]
        w, weights = extrapolate(cv, 4, ks)
        coarser, _ = extrapolate(cv, 4, [2, 4, 8])
        fewer_classes, _ = extrapolate(cv, 3, ks)
        alternative, _ = extrapolate(cv, 4, ks, power=0.5)
        b = gamma_moments_by_mean_and_cv(2.1, cv)
        h = H2Distribution.get_params_clx(b)
        service = effective_service_moments(b, [h.p1, 1 - h.p1], [h.mu1, h.mu2], 0.02, fixed=True, num_of_moments=2)
        v = [w[0] + service[0], w[1] + 2 * w[0] * service[0] + service[1]]
        calc = dict(
            w=w.tolist(),
            v=v,
            sw=float(np.sqrt(w[1] - w[0] ** 2)),
            sv=float(np.sqrt(v[1] - v[0] ** 2)),
            q=1.0,
            method="joint retained-work process, positive Erlang approximation, raw-moment extrapolation",
            classes=4,
            stages=ks,
            levels=100,
            extrapolation_weights=weights.tolist(),
            service_moments=service,
            stage_check_abs=(w - coarser).tolist(),
            quadrature_check_abs=(w - fewer_classes).tolist(),
            alternative_extrapolation_w=alternative.tolist(),
            extrapolation_sensitivity_abs=(w - alternative).tolist(),
        )
        old = next(r for r in original if r["cv"] == cv and r["mode"] == "disaster" and r["scenario"] == "fixed")
        sim = next(
            r
            for r in independent
            if r["cv"] == cv and r["mode"] == "disaster" and r["scenario"] == "fixed" and not r["independent"]
        )
        comparisons = {}
        for key in ["w", "v", "sw", "sv"]:
            actual = calc[key][0] if key in ["w", "v"] else calc[key]
            previous = old["calculation"][key][0] if key in ["w", "v"] else old["calculation"][key]
            estimate = sim["estimates"][key]
            comparisons[key] = dict(
                previous_error_percent=100 * (previous / estimate["mean"] - 1),
                refined_error_percent=100 * (actual / estimate["mean"] - 1),
                independent_mean=estimate["mean"],
                half_ci95=estimate["half_ci95"],
                inside_ci95=bool(abs(actual - estimate["mean"]) <= estimate["half_ci95"]),
            )
        results.append(dict(cv=cv, mode="disaster", scenario="fixed", calculation=calc, comparison=comparisons))
        print(cv, calc, comparisons, flush=True)
    a, b = point(1.2, 4, 8), point(1.2, 4, 8, 160)
    queue_change = {k: (np.array(b[k]) - np.array(a[k])).tolist() for k in ["w", "v"]}
    (ROOT / "full-work-results.json").write_text(
        json.dumps(dict(results=results, queue_truncation_change=queue_change), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
