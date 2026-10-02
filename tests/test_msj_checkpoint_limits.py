"""Nonzero-cost replay still has matched M/E2/1 and M/M/k no-preemption limits."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.theory.fifo.erlang import ErlangCCalc
from most_queue.theory.fifo.mg1 import MG1Calc

with (Path(__file__).parent / "default_params.yaml").open(encoding="utf-8") as stream:
    PARAMS = yaml.safe_load(stream)


@pytest.mark.parametrize("need", [1, 4])
@pytest.mark.parametrize("protection", [0, 5])
def test_checkpoint_no_preemption_matches_theory(need, protection):
    """With identical needs no preemption occurs, so configured costs vanish."""
    sim = MsjCheckpointSim(4, checkpoint_time=3, resume_time=2, seed=8053, min_service_time=protection)
    if need == 4:
        params = ErlangParams(r=2, mu=2)
        calc = MG1Calc()
        calc.set_sources(0.4)
        calc.set_servers(ErlangDistribution.calc_theory_moments(params, 4))
        sim.set_sources([0.4])
        sim.set_servers([4], [(params, "E")])
    else:
        calc = ErlangCCalc(4)
        calc.set_sources(2.2)
        calc.set_servers(1.0)
        sim.set_sources([2.2])
        sim.set_servers([1], [(1.0, "M")])
    expected = calc.run(2)
    actual = sim.run(int(PARAMS["num_of_jobs"]))
    print_waiting_moments(actual.w[:2], expected.w)
    print_sojourn_moments(actual.v[:2], expected.v)
    n = min(8, len(actual.p), len(expected.p))
    probs_print(actual.p, expected.p, n)
    for observed, reference in ((actual.w[:2], expected.w), (actual.v[:2], expected.v)):
        assert np.allclose(observed, reference, rtol=float(PARAMS["moments_rtol"]), atol=float(PARAMS["moments_atol"]))
    assert np.allclose(actual.p[:n], expected.p[:n], rtol=float(PARAMS["probs_rtol"]), atol=float(PARAMS["probs_atol"]))
    assert abs(actual.utilization - expected.utilization) < float(PARAMS["probs_atol"])
    assert actual.preemptions == 0
    assert actual.checkpoint_utilization == actual.resume_utilization == 0
    assert actual.productive_utilization == actual.utilization
    assert actual.protected_preemptions == actual.protection_expirations == 0
