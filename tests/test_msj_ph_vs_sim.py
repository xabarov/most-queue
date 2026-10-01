"""Independent PH-MSJ CTMC / DES validation, using project-wide tolerances."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.map_ph import PHDistribution
from most_queue.random.utils.params import Cox2Params, ErlangParams, H2Params
from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.theory.msj import MsjPHCalc

with (Path(__file__).parent / "default_params.yaml").open(encoding="utf-8") as stream:
    PARAMS = yaml.safe_load(stream)


@pytest.mark.parametrize(
    "service",
    [
        PHDistribution.from_erlang(ErlangParams(r=2, mu=2)),
        PHDistribution.from_cox(Cox2Params(mu1=2, mu2=1.5, p1=0.75)),
        PHDistribution.from_h2(H2Params(p1=0.25, mu1=0.5, mu2=1.5)),
    ],
)
def test_ph_msj_vs_independent_sim(service):
    rates, needs = [0.25, 0.15], [1, 2]
    services = [service, PHDistribution.from_exp(1)]
    calc = MsjPHCalc(2, truncation=11)
    calc.set_sources(rates)
    calc.set_servers(needs, services)
    theory = calc.run()
    assert theory.boundary_mass < 1e-3
    sim = MsjGeneralSim(2, seed=8047)
    sim.set_sources(rates)
    sim.set_servers(needs, [(params, "PH") for params in services])
    measured = sim.run(int(PARAMS["num_of_jobs"]))
    print_waiting_moments(measured.w[:1], theory.w)
    print_sojourn_moments(measured.v[:1], theory.v)
    probs_print(measured.p, theory.p, min(8, len(measured.p), len(theory.p)))
    print("Per-class sojourn, CTMC / DES:", theory.v_per_class, measured.v_per_class)
    for expected, observed in ((theory.w_per_class, measured.w_per_class), (theory.v_per_class, measured.v_per_class)):
        assert np.allclose(expected, observed, rtol=float(PARAMS["moments_rtol"]), atol=float(PARAMS["moments_atol"]))
    assert abs(theory.utilization - measured.utilization) < float(PARAMS["probs_atol"])
    n = min(8, len(theory.p), len(measured.p))
    assert np.allclose(theory.p[:n], measured.p[:n], rtol=float(PARAMS["probs_rtol"]), atol=float(PARAMS["probs_atol"]))
