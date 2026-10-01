"""All-resource conservative jobs reduce to M/E2/1, regardless of upper bounds."""

from pathlib import Path

import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.theory.fifo.mg1 import MG1Calc


def test_conservative_all_resource_jobs_vs_mg1():
    with (Path(__file__).parent / "default_params.yaml").open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    service = ErlangParams(r=2, mu=2)
    calc = MG1Calc()
    calc.set_sources(0.4)
    calc.set_servers(ErlangDistribution.calc_theory_moments(service, 4))
    theory = calc.run(2)
    sim = MsjGeneralSim(4, "conservative", seed=8048)
    sim.set_sources([0.4])
    sim.set_servers([4], [(service, "E")])
    trace = sim.make_trace(int(params["num_of_jobs"]) + 1000, estimates="oracle")
    measured = sim.run_trace(trace, warmup_jobs=1000)
    print_waiting_moments(measured.w[:2], theory.w)
    print_sojourn_moments(measured.v[:2], theory.v)
    probs_print(measured.p, theory.p, min(8, len(measured.p), len(theory.p)))
    assert np.allclose(measured.w[:2], theory.w, rtol=float(params["moments_rtol"]), atol=float(params["moments_atol"]))
    assert np.allclose(measured.v[:2], theory.v, rtol=float(params["moments_rtol"]), atol=float(params["moments_atol"]))
    n = min(8, len(measured.p), len(theory.p))
    assert np.allclose(measured.p[:n], theory.p[:n], rtol=float(params["probs_rtol"]), atol=float(params["probs_atol"]))
    assert abs(measured.utilization - theory.utilization) < float(params["probs_atol"])
    assert measured.backfilled == measured.reservation_violations == 0
