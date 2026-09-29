"""
Unit tests for the EDF (Earliest Deadline First) discipline simulator
(most_queue.sim.edf, EPIC-029).

No exact CTMC/closed-form calculator exists for this model (see
docs/research/edf-scheduling-2026.md for why -- three hypotheses tested and
rejected) -- correctness rests on: (1) the classical work-conservation law,
which is only exact when reneging is negligible; (2) direct EDF-vs-FCFS
comparison, using the same simulator with only the selection rule changed,
under the exact same reneging mechanics.
"""

import numpy as np
import pytest

from most_queue.sim.edf import EDFQueueSim
from most_queue.theory.utils.sla import deadline_violation_prob


def _run(discipline, deadlines, seed=1, n_events=300_000, lam=0.4, mu=1.0):
    sim = EDFQueueSim(n_classes=2, discipline=discipline, seed=seed)
    sim.set_sources([{"type": "M", "params": lam}, {"type": "M", "params": lam}])
    sim.set_servers([{"type": "M", "params": mu}, {"type": "M", "params": mu}])
    sim.set_deadlines(deadlines)
    return sim.run(n_events)


def test_conservation_law_holds_when_reneging_is_negligible():
    """Sum_k rho_k*E[W_k] = rho*E0/(1-rho) is only exact in the no-reneging limit
    (see research doc: reneging breaks the invariance argument at its root)."""
    lam, mu = 0.4, 1.0
    deadlines = [{"type": "M", "params": 0.001}, {"type": "M", "params": 0.002}]
    res = _run("edf", deadlines, seed=7, n_events=400_000, lam=lam, mu=mu)

    assert max(res.miss_prob) < 0.01  # reneging genuinely negligible

    rho_k = [lam / mu, lam / mu]
    rho = sum(rho_k)
    e0 = sum(lam * (2 / mu**2) / 2 for _ in range(2))
    lhs = sum(rho_k[c] * res.w[c][0] for c in range(2))
    rhs = rho * e0 / (1 - rho)
    assert np.isclose(lhs, rhs, rtol=0.1)


def test_edf_misses_fewer_deadlines_than_fcfs():
    """Same arrivals/service/reneging mechanics, only the selection rule differs."""
    deadlines = [{"type": "M", "params": 0.3}, {"type": "M", "params": 0.6}]
    edf = _run("edf", deadlines, seed=3)
    fcfs = _run("fcfs", deadlines, seed=3)

    avg_miss_edf = sum(edf.miss_prob) / 2
    avg_miss_fcfs = sum(fcfs.miss_prob) / 2
    assert avg_miss_edf < avg_miss_fcfs


def test_more_generous_deadline_class_misses_less():
    """Class 0 has a longer mean deadline (1/0.3) than class 1 (1/0.6): fewer misses."""
    deadlines = [{"type": "M", "params": 0.3}, {"type": "M", "params": 0.6}]
    res = _run("edf", deadlines, seed=5)
    assert res.miss_prob[0] < res.miss_prob[1]


def test_sojourn_equals_wait_plus_service():
    deadlines = [{"type": "M", "params": 0.3}, {"type": "M", "params": 0.6}]
    sim = EDFQueueSim(n_classes=2, seed=1)
    sim.set_sources([{"type": "M", "params": 0.4}, {"type": "M", "params": 0.4}])
    sim.set_servers([{"type": "M", "params": 1.0}, {"type": "M", "params": 2.0}])
    sim.set_deadlines(deadlines)
    res = sim.run(300_000)

    assert np.isclose(res.v[0][0] - res.w[0][0], 1.0, rtol=0.05)
    assert np.isclose(res.v[1][0] - res.w[1][0], 0.5, rtol=0.05)


def test_sla_layer_composition_deterministic_deadline():
    """
    Composition with EPIC-021's SLA layer: a served job's realized wait is
    bounded by its (deterministic) deadline by construction of EDF, so the
    fitted P(W>D) over the served-jobs' wait moments must be small (a
    genuine consistency check between the discipline and the fit-based
    tail estimate -- not exactly zero, since the smooth H2/Gamma fit does
    not know about the hard cutoff at D).
    """
    deadlines = [{"type": "D", "params": [3.0]}, {"type": "D", "params": [1.5]}]
    res = _run("edf", deadlines, seed=11)
    for cls, d in enumerate((3.0, 1.5)):
        p = deadline_violation_prob(res.w[cls], d)
        assert 0.0 <= p < 0.1


def test_deterministic_deadline_supported():
    """Deadlines need not be random -- the "D" (deterministic) Kendall distribution works too."""
    deadlines = [{"type": "D", "params": [2.0]}, {"type": "D", "params": [1.0]}]
    res = _run("edf", deadlines, seed=9, n_events=150_000)
    assert all(0.0 <= p <= 1.0 for p in res.miss_prob)


def test_invalid_n_classes_rejected():
    with pytest.raises(ValueError):
        EDFQueueSim(n_classes=0)


def test_invalid_discipline_rejected():
    with pytest.raises(ValueError):
        EDFQueueSim(n_classes=2, discipline="bogus")


def test_mismatched_source_count_rejected():
    sim = EDFQueueSim(n_classes=2)
    with pytest.raises(ValueError):
        sim.set_sources([{"type": "M", "params": 0.4}])


if __name__ == "__main__":
    test_conservation_law_holds_when_reneging_is_negligible()
    test_edf_misses_fewer_deadlines_than_fcfs()
    test_more_generous_deadline_class_misses_less()
    test_sojourn_equals_wait_plus_service()
    test_sla_layer_composition_deterministic_deadline()
    test_deterministic_deadline_supported()
    test_invalid_n_classes_rejected()
    test_invalid_discipline_rejected()
    test_mismatched_source_count_rejected()
    print("all EDF tests passed")
