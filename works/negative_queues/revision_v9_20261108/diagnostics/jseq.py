"""Retained-work J-sequence at given delta, K=8,12,16 extrapolated (weights 2,-9,8)."""
import json, sys, time
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work
n, lam, cv, delta, mode = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
Js = [int(x) for x in sys.argv[6].split(",")]; levels, behind = int(sys.argv[7]), int(sys.argv[8])
out = {}
for J in Js:
    g = {}
    for K in (8, 12, 16):
        t = time.perf_counter()
        r = calculate(fixed_work(cv, J, K), n=n, lam=lam, delta=delta, mode=mode, action="fixed", levels=levels, behind=behind)
        g[K] = {"w": list(r["w"]), "v": list(r["v"]), "sec": time.perf_counter() - t}
    ext = {k: [2*g[8][k][i] - 9*g[12][k][i] + 8*g[16][k][i] for i in range(2)] for k in ("w", "v")}
    out[f"J{J}"] = {"grids": g, "extrapolated": ext}
    print(J, ext, flush=True)
    json.dump(out, open(sys.argv[9], "w"), indent=1)
