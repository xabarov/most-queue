"""v9: quadrature correction of the retained-work results and comparison with simulation.

m_corr(delta) = E_K[m_{J,K}(delta)] + ( m_ref(0) - E_K[m_{J,K}(0)] ),
where E_K is the predetermined extrapolation in 1/K (main: K=8,12,16, weights
2,-9,8; extended: K=4,8,12, weights 1/2,-4,9/2), m are raw moments w1,w2,v1,v2,
and m_ref(0) is the M/G/n model without negative events (positive Gamma PH with
16 nodes for CV=0.8, the H2 itself for CV=1.2). Standard deviations are
computed after the correction. The correction does not use simulation values.
Output: corrected-results.json.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "revision_v7_20261107"
COR = ROOT / "correction"
SIM = ROOT / "long_sim"
WEIGHTS = {(8, 12, 16): (2, -9, 8), (4, 8, 12): (0.5, -4, 4.5)}


def load(name):
    return json.loads((COR / name).read_text())


def ext(label, j, ks, levels):
    grids = [load(f"{label}-fixed-j{j}-k{k}-n{levels}.json") for k in ks]
    a = WEIGHTS[ks]
    return {key: [sum(c * g[key][i] for c, g in zip(a, grids)) for i in range(2)] for key in ("w", "v")}


def sd(m):
    return math.sqrt(m[1] - m[0] ** 2)


def compare(calc, sim):
    out = {}
    for key, value in (("w", calc["w"][0]), ("v", calc["v"][0]), ("sw", sd(calc["w"])), ("sv", sd(calc["v"]))):
        s = sim[key]
        out[key] = {"calculation": value, "simulation": s["mean"], "half_ci95": s["half_ci95"],
                    "difference_percent": 100 * (value / s["mean"] - 1),
                    "inside_ci95": abs(value - s["mean"]) <= s["half_ci95"]}
    return out


def corrected(raw, label, j, ks, levels):
    zero = ext(label, j, ks, levels)
    ref = load(f"{label}-ref-j0-k0-n240.json")
    shift = {key: [ref[key][i] - zero[key][i] for i in range(2)] for key in ("w", "v")}
    calc = {key: [raw[key][i] + shift[key][i] for i in range(2)] for key in ("w", "v")}
    return calc, {"delta0_extrapolated": zero, "delta0_reference": {k: ref[k] for k in ("w", "v", "model")},
                  "shift": shift, "relative_shift_w1_percent": 100 * shift["w"][0] / raw["w"][0]}


def main():
    v7 = json.loads((V7 / "experiment-results.json").read_text())
    out = {"main": [], "extended": [], "validation_n2": {}}
    names = {("rcs", 0.8): "main-fixed-rcs-cv0.8", ("disaster", 0.8): "main-fixed-disaster-cv0.8",
             ("rcs", 1.2): "main-fixed-rcs-cv1.2", ("disaster", 1.2): "main-fixed-disaster-cv1.2"}
    for item in v7["results"]:
        if item["scenario"] != "fixed":
            continue
        raw = {k: item["calculation"][k] for k in ("w", "v")}
        calc, info = corrected(raw, f"main-cv{item['cv']}", 4, (8, 12, 16), 100)
        sim = json.loads((SIM / (names[(item["mode"], item["cv"])] + ".json")).read_text())["summary"]
        out["main"].append({"label": item["label"], "cv": item["cv"], "mode": item["mode"], "raw_v8": raw,
                            "corrected": calc, "correction": info,
                            "long_simulation": compare(calc, sim), "long_simulation_raw": compare(raw, sim),
                            "original_simulation": compare(calc, item["simulation"])})
    labels = {"case08": "case08-10", "case10": "case08-10", "case09": "case09", "case11": "case11"}
    for item in v7["extended"]:
        cid = item["case"]["id"]
        if cid not in labels:
            continue
        c = item["calculation"]
        raw = {k: c[k] for k in ("w", "v")}
        calc, info = corrected(raw, labels[cid], 3, (4, 8, 12), c["levels"])
        sim = json.loads((SIM / f"{cid}.json").read_text())["summary"]
        orig = {k: {"mean": item["comparison"][k]["simulation"], "half_ci95": item["comparison"][k]["half_ci95"]}
                for k in ("w", "v", "sw", "sv")}
        out["extended"].append({"case": item["case"], "raw_v8": raw, "corrected": calc, "correction": info,
                                "long_simulation": compare(calc, sim), "long_simulation_raw": compare(raw, sim),
                                "original_simulation": compare(calc, orig)})
    # Validation on n=2 (case08): full J sequence at delta=0.04 (K=8,12,16).
    seq = json.loads((ROOT / "diagnostics/case08_jseq.json").read_text())
    j = {int(k[1:]): v["extrapolated"]["w"][0] for k, v in seq.items()}
    d = [j[4] - j[3], j[5] - j[4], j[6] - j[5]]
    ratio = d[2] / d[1]
    limit = j[6] + d[2] * ratio / (1 - ratio)
    zero = json.loads((ROOT / "diagnostics/d0_n2_cv08.json").read_text())
    z = {J: 2 * zero["fixed"][f"J{J}K8"]["w"][0] - 9 * zero["fixed"][f"J{J}K12"]["w"][0] + 8 * zero["fixed"][f"J{J}K16"]["w"][0]
         for J in (3, 4, 5, 6)}
    ref = zero["ref"]["gammaPH16"]["w"][0]
    out["validation_n2"] = {"J_sequence_delta0.04": j, "geometric_limit": limit, "ratio": ratio,
                            "corrected_from_J": {J: j[J] + ref - z[J] for J in (3, 4, 5, 6)}}
    (ROOT / "corrected-results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in out["main"] + out["extended"]:
        name = r.get("label") or r["case"]["id"]
        cv = r.get("cv") or r["case"]["cv"]
        ls, lr = r["long_simulation"], r["long_simulation_raw"]
        print(f"{name} CV={cv}: shift w1 {r['correction']['relative_shift_w1_percent']:+.3f}%")
        for k in ("w", "v", "sw", "sv"):
            print(f"   {k}: raw {lr[k]['calculation']:.5f} ({lr[k]['difference_percent']:+.2f}%, {'in' if lr[k]['inside_ci95'] else 'OUT'})"
                  f"  corr {ls[k]['calculation']:.5f} ({ls[k]['difference_percent']:+.2f}%, {'in' if ls[k]['inside_ci95'] else 'OUT'})"
                  f"  ИМ {ls[k]['simulation']:.5f}±{ls[k]['half_ci95']:.5f}")
    print("n=2 validation:", json.dumps(out["validation_n2"], indent=1))


if __name__ == "__main__":
    main()
