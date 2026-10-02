"""Censored age forecasts preserve the matched all-resource M/E2/1 limit."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.sim.utils.residual_runtime import KaplanMeierRuntimeEstimator
from most_queue.theory.fifo.mg1 import MG1Calc


@pytest.mark.parametrize("policy", ["easy", "conservative"])
def test_censored_residual_forecasts_preserve_mg1_limit(policy):
    """All-resource jobs reduce exactly to FCFS despite forecast revisions."""
    with (Path(__file__).parent / "default_params.yaml").open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    rng = np.random.default_rng(1551)
    history, exposure = rng.gamma(2, 0.5, 4000), rng.exponential(5, 4000)
    model = KaplanMeierRuntimeEstimator().fit(np.minimum(history, exposure), history <= exposure)
    service = ErlangParams(r=2, mu=2)
    theory = MG1Calc()
    theory.set_sources(0.4)
    theory.set_servers(ErlangDistribution.calc_theory_moments(service, 4))
    expected = theory.run(2)
    sim = MsjGeneralSim(4, policy, seed=8051)
    sim.set_sources([0.4])
    sim.set_servers([4], [(service, "E")])
    trace = sim.make_trace(int(params["num_of_jobs"]) + 1000, estimates=[model.remaining_quantile(0)])
    measured = sim.run_trace(
        trace, warmup_jobs=1000, remaining_predictor=lambda cls, age: model.remaining_quantile(age)
    )
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
    assert measured.runtime_updates > 0
    fcfs = MsjGeneralSim(4, "fcfs")
    fcfs.set_servers([4])
    assert measured.start_times == fcfs.run_trace(trace, warmup_jobs=1000).start_times
