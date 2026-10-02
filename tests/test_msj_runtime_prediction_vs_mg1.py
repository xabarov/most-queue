"""Trained, non-oracle predictions must preserve the all-resource M/E2/1 limit."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor
from most_queue.theory.fifo.mg1 import MG1Calc


@pytest.mark.parametrize("policy", ["easy", "conservative"])
@pytest.mark.parametrize("calibration_mode", ["pooled", "grouped"])
def test_trained_runtime_bounds_preserve_mg1_limit(policy, calibration_mode):
    with (Path(__file__).parent / "default_params.yaml").open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    rng = np.random.default_rng(1549)
    model = LogLinearRuntimePredictor().fit(np.empty((1500, 0)), rng.gamma(2, 0.5, 1500))
    calibration_services = rng.gamma(2, 0.5, 1000)
    if calibration_mode == "pooled":
        model.calibrate(np.empty((1000, 0)), calibration_services)
    else:
        model.calibrate_by_group(np.empty((1000, 0)), calibration_services, np.full(1000, 4))
    service = ErlangParams(r=2, mu=2)
    theory = MG1Calc()
    theory.set_sources(0.4)
    theory.set_servers(ErlangDistribution.calc_theory_moments(service, 4))
    expected = theory.run(2)
    sim = MsjGeneralSim(4, policy, seed=8049)
    sim.set_sources([0.4])
    sim.set_servers([4], [(service, "E")])
    trace = sim.make_trace(int(params["num_of_jobs"]) + 1000)
    groups = np.full(len(trace), 4) if calibration_mode == "grouped" else None
    estimates = model.predict(np.empty((len(trace), 0)), upper=True, groups=groups)
    predicted_trace = tuple(replace(job, estimate=float(estimate)) for job, estimate in zip(trace, estimates))
    measured = sim.run_trace(predicted_trace, warmup_jobs=1000)
    print_waiting_moments(measured.w[:2], expected.w)
    print_sojourn_moments(measured.v[:2], expected.v)
    probs_print(measured.p, expected.p, min(8, len(measured.p), len(expected.p)))
    for actual, reference in ((measured.w[:2], expected.w), (measured.v[:2], expected.v)):
        assert np.allclose(actual, reference, rtol=float(params["moments_rtol"]), atol=float(params["moments_atol"]))
    n = min(8, len(measured.p), len(expected.p))
    assert np.allclose(
        measured.p[:n], expected.p[:n], rtol=float(params["probs_rtol"]), atol=float(params["probs_atol"])
    )
    assert abs(measured.utilization - expected.utilization) < float(params["probs_atol"])
    assert measured.backfilled == 0
    assert measured.reservation_violations > 0  # statistical coverage is not an all-jobs upper bound
