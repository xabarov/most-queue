import math
import json
import mpmath as mp
import sympy as sp
import most_queue.theory.negative.mgn_rcs as rcs
import most_queue.theory.negative.mgn_disaster as disaster
from most_queue.theory.utils.restarts import effective_service_moments, is_real_h2
from works.negative_queues.revision_20261107.run_experiments import calculate

x, z = sp.symbols("x z")
c = [sp.Integer(1)]
polys = []
for k in range(1, 5):
    c.append(
        sp.expand(
            (1 if k == 1 else 0) - z * sum(x ** (j - 1) / sp.factorial(j - 1) * c[k - j] for j in range(1, k + 1))
        )
    )
    polys.append(sp.Poly((-1) ** k * sp.factorial(k) * c[k], z, x))


def exact(b, y, mu, r, *, fixed=False, num_of_moments=4):
    if not fixed or r == 0:
        return effective_service_moments(b, y, mu, r, fixed=fixed, num_of_moments=num_of_moments)
    with mp.workdps(65):
        rr = mp.mpf(float(r))
        if is_real_h2(y, mu):
            pairs = [(mp.mpf(float(yi.real)), mp.mpf(float(mui.real))) for yi, mui in zip(y, mu) if yi.real > 0]
            rate = min(mui for yi, mui in pairs)

            def dM(t, q):
                return sum(yi * mui * mp.factorial(q) / (mui - t) ** (q + 1) for yi, mui in pairs)

        else:
            mean = mp.mpf(b[0])
            var = mp.mpf(b[1]) - mean**2
            shape, scale = mean**2 / var, var / mean
            rate = 1 / scale

            def dM(t, q):
                return mp.rf(shape, q) * scale**q * (1 - scale * t) ** (-shape - q)

        return [
            (
                float(sum(mp.mpf(str(coef)) * rr**q * dM(j * rr, q) for (j, q), coef in p.terms()) / rr**k)
                if k * rr < rate
                else float("inf")
            )
            for k, p in enumerate(polys[:num_of_moments], 1)
        ]


if __name__ == "__main__":
    rcs.effective_service_moments = exact
    disaster.effective_service_moments = exact
    rows = json.load(open("works/negative_queues/revision_20261107/experiment-results.json"))["results"]
    out = []
    for row in rows:
        if row["scenario"] != "fixed":
            continue
        try:
            a = calculate(row["cv"], row["mode"], "fixed", 160)
        except (ValueError, FloatingPointError) as e:
            out.append({"cv": row["cv"], "mode": row["mode"], "error": str(e)})
            print(row["cv"], row["mode"], str(e), flush=True)
            continue
        record = {
            "cv": row["cv"],
            "mode": row["mode"],
            "old": row["calculation"],
            "exact_moments": a,
            "simulation": row["simulation"],
        }
        out.append(record)
        print(
            row["cv"],
            row["mode"],
            [
                (k, round(100 * ((a[k][0] if k in ["w", "v"] else a[k]) / row["simulation"][k]["mean"] - 1), 3))
                for k in ["w", "v", "sw", "sv"]
            ],
            flush=True,
        )
    json.dump(out, open("works/negative_queues/refinement_20261107/exact-moment-candidates.json", "w"), indent=2)
