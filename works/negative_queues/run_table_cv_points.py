"""
Run numerical and simulation for CV=0.8, 1.2, 1.6 at n=3, rho=0.7, lambda_pos=1.0, delta=0.3.
Output: table rows with Числ, ИМ, Отн.% for w1 and v1 (for filling negative_queues_takahasi_takami.tex).
"""

import os
import sys

import numpy as np
import yaml

# run from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import DisasterScenario, NegativeServiceType, QsSimNegatives, RcsScenario
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc


def load_qp():
    path = os.path.join(os.path.dirname(__file__), "base_parameters.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_calc(qp, n, b, discipline, requeue):
    if discipline == NegativeServiceType.RCS:
        calc = MGnNegativeRCSCalc(
            n=n,
            calc_params=TakahashiTakamiParams(tolerance=float(qp["accuracy"])),
            requeue_on_disaster=requeue,
        )
    else:
        calc = MGnNegativeDisasterCalc(
            n=n,
            calc_params=TakahashiTakamiParams(tolerance=float(qp["accuracy"])),
            requeue_on_disaster=requeue,
        )
    calc.set_sources(
        l_pos=float(qp["arrival_rate"]["positive"]),
        l_neg=float(qp["arrival_rate"]["negative"]),
    )
    calc.set_servers(b=b)
    res = calc.run()
    w = calc.get_w(num_of_moments=2)
    v = calc.get_v(num_of_moments=2)
    return w[0], v[0]


def collect_sim(qp, n, b, discipline, rcs_scenario, disaster_scenario):
    sim = QsSimNegatives(
        n,
        discipline,
        verbose=False,
        rcs_scenario=rcs_scenario,
        disaster_scenario=disaster_scenario,
    )
    sim.set_negative_sources(float(qp["arrival_rate"]["negative"]), "M")
    sim.set_positive_sources(float(qp["arrival_rate"]["positive"]), "M")
    mean_b = float(b[0])
    var_b = float(b[1] - b[0] ** 2)
    cv_b = float(np.sqrt(max(0.0, var_b)) / mean_b) if mean_b > 0 else 0.0
    if cv_b > 1.0 + 1e-12:
        h2_params = H2Distribution.get_params(b)
        sim.set_servers(h2_params, "H")
    else:
        gamma_params = GammaDistribution.get_params([b[0], b[1]])
        sim.set_servers(gamma_params, "Gamma")
    res = sim.run(int(qp["num_of_jobs"]))
    w1 = res.w[0] if hasattr(res, "w") and res.w else (res.v[0] - 1.0 / (1.0 / b[0]) if b[0] else 0)
    # sojourn v from sim
    v1 = res.v[0]
    # waiting from results if available
    if hasattr(res, "w") and len(res.w):
        w1 = res.w[0]
    else:
        # approximate w = v - service_mean for single server moment
        w1 = v1 - b[0]
    return w1, v1


def rel_err(calc_val, sim_val):
    c = float(calc_val.real) if hasattr(calc_val, "real") else float(calc_val)
    s = float(sim_val)
    if s == 0:
        return 0.0
    return 100.0 * (c - s) / s


SCENARIOS = [
    ("RCS (удаление)", NegativeServiceType.RCS, False, RcsScenario.REMOVE, DisasterScenario.CLEAR_SYSTEM),
    ("RCS (повторное обслуж.)", NegativeServiceType.RCS, True, RcsScenario.REQUEUE, DisasterScenario.CLEAR_SYSTEM),
    ("Катастрофы (очищение)", NegativeServiceType.DISASTER, False, RcsScenario.REMOVE, DisasterScenario.CLEAR_SYSTEM),
    (
        "Катастрофы (повторное обслуж.)",
        NegativeServiceType.DISASTER,
        True,
        RcsScenario.REMOVE,
        DisasterScenario.REQUEUE_ALL,
    ),
]


def main():
    qp = load_qp()
    n = qp["channels"]["base"]
    lam = float(qp["arrival_rate"]["positive"])
    rho = qp["utilization"]["base"]
    service_mean = n * rho / lam

    cvs = [0.8, 1.2, 1.6]
    all_rows = []

    for cv in cvs:
        b_full = gamma_moments_by_mean_and_cv(service_mean, cv)
        b_calc = b_full  # theory uses get_params_clx(b) and needs at least 3 moments
        b_sim = [b_full[0], b_full[1]] if cv <= 1.0 + 1e-12 else b_full[:4]  # Gamma: 2 moments; H2: need 3+
        row_block = []
        for name, discipline, requeue, rcs_sc, dis_sc in SCENARIOS:
            w_calc, v_calc = collect_calc(qp, n, b_calc, discipline, requeue)
            w_sim, v_sim = collect_sim(qp, n, b_sim, discipline, rcs_sc, dis_sc)
            w_err = rel_err(w_calc, w_sim)
            v_err = rel_err(v_calc, v_sim)
            w_c = float(w_calc.real) if hasattr(w_calc, "real") else float(w_calc)
            v_c = float(v_calc.real) if hasattr(v_calc, "real") else float(v_calc)
            row_block.append((name, w_c, w_sim, w_err, v_c, v_sim, v_err))
            print(
                f"CV={cv} {name}: w1 calc={w_c:.4f} sim={w_sim:.4f} err={w_err:.2f}%  v1 calc={v_c:.4f} sim={v_sim:.4f} err={v_err:.2f}%"
            )
        all_rows.append((cv, row_block))

    # Print LaTeX table rows
    print("\n--- LaTeX table body (replace --- with values) ---\n")
    for cv, rows in all_rows:
        cv_label = "0{,}8" if cv == 0.8 else "1{,}2" if cv == 1.2 else "1{,}6"
        base_label = " (базовое значение)" if cv == 1.2 else ""
        print(f"% CV={cv_label}{base_label}")
        for name, w_c, w_s, w_e, v_c, v_s, v_e in rows:
            print(f"{name} & {w_c:.4f} & {w_s:.4f} & {w_e:.2f} & {v_c:.4f} & {v_s:.4f} & {v_e:.2f} \\\\")
        print()


if __name__ == "__main__":
    main()
