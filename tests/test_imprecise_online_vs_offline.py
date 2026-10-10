"""
Score online imprecise-computation policies against the EXACT offline optimum
(EPIC-080, roadmap item R6).

This is the comparison the pair of modules exists for. The offline problem is
solved exactly, so an online rule can be measured against the best any
scheduler could have achieved on the very same jobs -- not against another
heuristic and not against an asymptotic bound.

The invariant that matters, and the reason the comparison needs a guard: the
offline optimum completes every mandatory part, so any online policy that also
completes them all can only do worse. A policy that *skips* mandatory work is
not solving the same problem and will happily report a lower optional error
while being strictly worse. ``full_edf`` does exactly that, and is kept here as
the counterexample that makes the guard necessary.
"""

import numpy as np
import pytest

from most_queue.sim.imprecise import (
    ImpreciseComputationSim,
    compare_with_offline,
    to_offline_tasks,
)
from most_queue.theory.imprecise import ImpreciseComputationScheduler

MANDATORY_RATE = 5.0  # mean 0.2
OPTIONAL_RATE = 5.0  # mean 0.2


def _sim(policy, lam, deadline, seed, kendall="D"):
    sim = ImpreciseComputationSim(policy=policy, seed=seed)
    sim.set_sources(lam, "M")
    sim.set_servers(MANDATORY_RATE, OPTIONAL_RATE, "M")
    sim.set_deadlines(deadline, kendall)
    return sim


@pytest.mark.parametrize("lam", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mandatory_first_never_beats_the_offline_optimum(lam, seed):
    """
    The invariant. When the online run completes every mandatory part, its
    optional error cannot be below the exact optimum -- if it ever were, either
    the optimum or the simulator would be wrong.
    """
    sim = _sim("mandatory_first", lam, 2.0, seed)
    comparison = compare_with_offline(sim, sim.generate_jobs(120))
    if not comparison["comparable"]:
        pytest.skip("instance not comparable (mandatory set infeasible or missed online)")
    assert comparison["online"] >= comparison["offline"] - 1e-7
    assert comparison["ratio"] is None or comparison["ratio"] >= 1.0 - 1e-9


def test_full_edf_is_flagged_rather_than_credited():
    """
    ``full_edf`` misses mandatory deadlines and so reports a *lower* optional
    error than the offline optimum. The comparison must refuse to turn that into
    a flattering ratio.
    """
    sim = _sim("full_edf", 2.0, 2.0, seed=3)
    comparison = compare_with_offline(sim, sim.generate_jobs(150))
    assert comparison["online_mandatory_misses"] > 0
    assert not comparison["comparable"]
    assert comparison["ratio"] is None
    assert comparison["online"] < comparison["offline"]  # the very reason it is not comparable


def test_mandatory_first_keeps_every_mandatory_part_when_the_instance_allows_it():
    """
    When an optimal scheduler can fit all the mandatory work, the mandatory-first
    rule does too -- that is the one guarantee it is supposed to give.
    """
    for seed in (1, 2, 3, 4):
        sim = _sim("mandatory_first", 1.5, 2.0, seed)
        jobs = sim.generate_jobs(120)
        offline = ImpreciseComputationScheduler().set_tasks(to_offline_tasks(jobs)).solve()
        if not offline.feasible:
            continue
        assert sim.run_on(jobs).mandatory_miss_fraction == 0.0


def test_error_grows_with_load_and_vanishes_when_idle():
    previous = None
    for lam in (0.3, 1.0, 2.0, 3.0):
        result = _sim("mandatory_first", lam, 2.0, seed=5).run(3000)
        if previous is not None:
            assert result.mean_error >= previous - 1e-9
        previous = result.mean_error
    light = _sim("mandatory_first", 0.05, 50.0, seed=5).run(500)
    assert light.mean_error < 1e-6
    assert light.mandatory_miss_fraction == 0.0


def test_a_generous_deadline_removes_all_error():
    """With far more time than work, nothing has to be truncated."""
    result = _sim("mandatory_first", 0.5, 100.0, seed=9).run(800)
    assert np.isclose(result.mean_error, 0.0, atol=1e-9)
    assert np.isclose(result.error_fraction, 0.0, atol=1e-12)


def test_generated_batch_round_trips_into_the_offline_model():
    sim = _sim("mandatory_first", 1.0, 2.0, seed=11)
    jobs = sim.generate_jobs(40)
    tasks = to_offline_tasks(jobs)
    assert len(tasks) == len(jobs)
    for job, task in zip(jobs, tasks):
        assert task.release == job.arrival
        assert task.deadline == job.deadline
        assert task.mandatory == job.mandatory
        assert task.optional == job.optional


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="policy must be"):
        ImpreciseComputationSim(policy="nonsense")
    with pytest.raises(ValueError, match="must be called first"):
        ImpreciseComputationSim().generate_jobs(5)
