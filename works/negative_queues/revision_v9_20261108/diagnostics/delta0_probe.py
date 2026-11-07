"""delta=0 diagnostic: quadrature-class (retained work) model vs accurate M/G/n reference."""
import json, sys, time
import numpy as np
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work, hyperexponential
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv

n, lam, cv = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
Js = [int(x) for x in sys.argv[4].split(",")]
Ks = [int(x) for x in sys.argv[5].split(",")]
levels = int(sys.argv[6]) if len(sys.argv) > 6 else 120
out = {"n": n, "lam": lam, "cv": cv, "ref": {}, "fixed": {}}
def run(service, action):
    t = time.perf_counter()
    r = calculate(service, n=n, lam=lam, delta=0.0, mode="rcs", action=action, levels=levels, behind=0)
    return {"w": list(r["w"]), "v": list(r["v"]), "sec": time.perf_counter() - t}
if cv < 1:
    for q in (8, 16):
        out["ref"][f"gammaPH{q}"] = run(gamma_ph(2.1, cv, q), "restart"); print("ref", q, out["ref"][f"gammaPH{q}"], flush=True)
else:
    h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(2.1, cv))
    out["ref"]["H2"] = run(hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2]), "restart"); print("ref", out["ref"]["H2"], flush=True)
for J in Js:
    for K in Ks:
        out["fixed"][f"J{J}K{K}"] = run(fixed_work(cv, J, K), "fixed"); print(J, K, out["fixed"][f"J{J}K{K}"], flush=True)
json.dump(out, open(sys.argv[7], "w"), indent=1)
