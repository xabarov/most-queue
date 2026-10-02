"""Matched analytical limits; no transfer of MSFQ exponential theorems to G."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.sim.utils.msj_packing import PACKING_POLICIES
from most_queue.structs import QueueResults
from most_queue.theory.fifo.mg1 import MG1Calc

with (Path(__file__).parent / "default_params.yaml").open(encoding="utf-8") as stream:
    PARAMS = yaml.safe_load(stream)


@pytest.mark.parametrize("policy", PACKING_POLICIES)
@pytest.mark.parametrize("limit", ["all_resource_erlang2", "unit_resource_exp"])
def test_packing_known_limits(policy, limit):
    """All-wide M/E2/1; all-unit M/M/4 (MSFQ ell=0 only for this limit)."""
    sim = MsjGeneralSim(4, policy, seed=8052)
    if limit == "all_resource_erlang2":
        service = ErlangParams(r=2, mu=2)
        calc = MG1Calc()
        calc.set_sources(0.4)
        calc.set_servers(ErlangDistribution.calc_theory_moments(service, 4))
        theory = calc.run(2)
        sim.set_sources([0.4])
        sim.set_servers([4], [(service, "E")])
    else:
        rate, k = 2.2, 4
        rho = rate / k
        p0 = 1 / (sum(rate**i / math.factorial(i) for i in range(k)) + rate**k / math.factorial(k) / (1 - rho))
        p_wait = p0 * rate**k / math.factorial(k) / (1 - rho)
        w = [p_wait / (k - rate), 2 * p_wait / (k - rate) ** 2]
        p = [
            p0 * rate**i / math.factorial(i) if i <= k else p0 * rate**k / math.factorial(k) * rho ** (i - k)
            for i in range(20)
        ]
        theory = QueueResults(w=w, v=[w[0] + 1, w[1] + 2 * w[0] + 2], p=p, utilization=rho)
        sim.set_sources([rate])
        sim.set_servers([1], [(1.0, "M")])
    measured = sim.run(int(PARAMS["num_of_jobs"]))
    print_waiting_moments(measured.w[:2], theory.w)
    print_sojourn_moments(measured.v[:2], theory.v)
    n = min(8, len(measured.p), len(theory.p))
    probs_print(measured.p, theory.p, n)
    for actual, expected in ((measured.w[:2], theory.w), (measured.v[:2], theory.v)):
        assert np.allclose(actual, expected, rtol=float(PARAMS["moments_rtol"]), atol=float(PARAMS["moments_atol"]))
    assert np.allclose(measured.p[:n], theory.p[:n], rtol=float(PARAMS["probs_rtol"]), atol=float(PARAMS["probs_atol"]))
    assert abs(measured.utilization - theory.utilization) < float(PARAMS["probs_atol"])
    assert measured.preemptions == measured.reservations == 0
