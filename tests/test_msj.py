"""
Tests for the multiserver-job (MSJ) model: exact FCFS CTMC (mean response time),
saturated-system throughput / stability threshold, and the discrete-event
simulator.
"""

import numpy as np
import pytest

from most_queue.sim.msj import MsjClass, MsjSim
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.msj import MsjExactCalc, MsjSaturatedCalc


def test_exact_single_class_need1_is_mmk():
    """A single class needing 1 server each reduces MSJ to M/M/k."""
    calc = MsjExactCalc(k=2, classes=[MsjClass(1.2, 1, 1.0)], truncation=40)
    v = calc.run().v[0]

    mm = MMnrCalc(n=2, r=200)
    mm.set_sources(l=1.2)
    mm.set_servers(mu=1.0)
    assert np.isclose(v, mm.run().v[0], rtol=2e-3)


@pytest.mark.parametrize("l0, l1", [(0.4, 0.2), (0.5, 0.25)])
def test_exact_vs_sim(l0, l1):
    """FCFS MSJ mean response (2 classes, needs 1 and 2) — exact CTMC vs simulation."""
    classes = [MsjClass(l0, 1, 1.0), MsjClass(l1, 2, 1.0)]
    calc = MsjExactCalc(k=2, classes=classes, truncation=14)  # 2 classes -> ~2^15 states
    r = calc.run()
    assert calc.boundary_mass < 1e-2

    sim = MsjSim(k=2, classes=classes, seed=1234)
    rs = sim.run(120_000)
    assert np.isclose(r.v[0], rs.v[0], rtol=0.05)


def test_saturated_need1_is_kmu():
    """Saturated throughput of a need-1 class is k*mu (as in M/M/k)."""
    sat = MsjSaturatedCalc(k=3, classes=[MsjClass(1.0, 1, 2.0)])
    assert np.isclose(sat.run(), 3 * 2.0, rtol=1e-9)


def test_saturated_is_stability_threshold():
    """
    The saturated throughput X_sat is the max sustainable total arrival rate:
    below it the system is stable (finite mean response), and X_sat does not
    exceed the naive capacity bound.
    """
    classes_mix = [MsjClass(1.0, 1, 1.0), MsjClass(1.0, 2, 1.0)]  # 1:1 mix, needs 1 and 2
    x_sat = MsjSaturatedCalc(k=2, classes=classes_mix).run()
    assert 0.0 < x_sat <= 2.0  # cannot exceed k*mu

    # simulate at 80% of the threshold -> finite (stable)
    lam = 0.8 * x_sat
    sim = MsjSim(k=2, classes=[MsjClass(lam * 0.5, 1, 1.0), MsjClass(lam * 0.5, 2, 1.0)], seed=1234)
    v = sim.run(120_000).v[0]
    assert np.isfinite(v) and v < 50.0


if __name__ == "__main__":
    test_exact_single_class_need1_is_mmk()
    test_saturated_need1_is_kmu()
    test_saturated_is_stability_threshold()
