"""v9: removal and clearing by Takahashi-Takami + tagged equations.

1) Main table rows (n=3, lambda=1, delta=0.3) and extended cases 00-03.
2) Law substitution: H2 fitted to three Gamma moments (TT) vs the positive
   phase-type representation of the Gamma law itself (16 quadrature nodes,
   v7 positive_ph), CV=0.8, 0.6, 0.5.
Output: tt-results.json. Run from the repository root with PYTHONPATH=.
"""

import json
from pathlib import Path

from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107 import ph_queue
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v9_20261108 import tt_tagged

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "revision_v7_20261107"
M = 2.1
KEYS = ("w", "v", "sw", "sv", "q", "balance_residual", "little_residuals", "tail", "tt_library", "h2")


def row(b, n, lam, delta, mode, levels=240, behind=80):
    r = tt_tagged.calculate(b, n, lam, delta, mode, levels=levels, behind=behind)
    return {k: r[k] for k in KEYS}


def main():
    v7 = json.loads((V7 / "experiment-results.json").read_text())
    out = {"main": [], "extended": [], "law": []}
    for item in v7["results"]:
        if item["scenario"] != "remove":
            continue
        cv, mode = item["cv"], item["mode"]
        r = row(gamma_moments_by_mean_and_cv(M, cv), 3, 1.0, 0.3, mode)
        out["main"].append({"label": item["label"], "cv": cv, "mode": mode, "tt": r,
                            "v8_calculation": {k: item["calculation"][k] for k in ("w", "v", "sw", "sv", "q")},
                            "simulation": item["simulation"]})
    for item in v7["extended"]:
        case = item["case"]
        if case["action"] != "remove":
            continue
        r = row(gamma_moments_by_mean_and_cv(M, case["cv"]), case["n"], case["lam"], case["delta"], case["mode"])
        out["extended"].append({"case": case, "tt": r,
                                "v8_calculation": {k: item["comparison"][k]["calculation"] for k in ("w", "v", "sw", "sv")},
                                "simulation": {k: {"mean": item["comparison"][k]["simulation"],
                                                   "half_ci95": item["comparison"][k]["half_ci95"]} for k in ("w", "v", "sw", "sv")}})
    for cv in (0.8, 0.6, 0.5):
        for n, rho, delta in [(3, 0.7, 0.3), (3, 0.84, 0.3), (2, 0.63, 0.15)]:
            lam = rho * n / M
            for mode in ("rcs", "disaster"):
                tt = row(gamma_moments_by_mean_and_cv(M, cv), n, lam, delta, mode)
                ref = ph_queue.calculate(gamma_ph(M, cv, 16), n=n, lam=lam, delta=delta, mode=mode, action="remove",
                                         levels=240, behind=80)
                out["law"].append({"cv": cv, "n": n, "rho": rho, "delta": delta, "mode": mode, "tt": tt,
                                   "gamma_ph16": {k: ref[k] for k in ("w", "v", "sw", "sv", "q", "model")}})
    (ROOT / "tt-results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in out["main"]:
        t, s = r["tt"], r["simulation"]
        print(r["label"], r["cv"], "w %.5f (ИМ %.5f±%.5f) v %.5f (ИМ %.5f±%.5f) sw %.4f (%.4f±%.4f) sv %.4f (%.4f±%.4f)" % (
            t["w"][0], s["w"]["mean"], s["w"]["half_ci95"], t["v"][0], s["v"]["mean"], s["v"]["half_ci95"],
            t["sw"], s["sw"]["mean"], s["sw"]["half_ci95"], t["sv"], s["sv"]["mean"], s["sv"]["half_ci95"]))
    for r in out["extended"]:
        t, s = r["tt"], r["simulation"]
        print(r["case"]["id"], "w %.5f (ИМ %.5f±%.5f) v %.5f (%.5f±%.5f) sw %.4f (%.4f±%.4f) sv %.4f (%.4f±%.4f)" % (
            t["w"][0], s["w"]["mean"], s["w"]["half_ci95"], t["v"][0], s["v"]["mean"], s["v"]["half_ci95"],
            t["sw"], s["sw"]["mean"], s["sw"]["half_ci95"], t["sv"], s["sv"]["mean"], s["sv"]["half_ci95"]))


if __name__ == "__main__":
    main()
