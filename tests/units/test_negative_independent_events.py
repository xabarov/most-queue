"""Generalizing the independent simulator preserves established seeded paths."""

import pytest

from works.negative_queues.refinement_20261107.diagnose import independent_simulation
from works.negative_queues.revision_v7_20261107.simulation import run


@pytest.mark.parametrize("cv", [0.8, 1.2])
@pytest.mark.parametrize("action", ["fixed", "restart"])
def test_rcs_trajectories_match_previous_independent_simulator(cv, action):
    seed = 738294
    case = dict(
        id="validation",
        n=3,
        lam=1,
        delta=0.02 if action == "fixed" else 0.3,
        cv=cv,
        mean=2.1,
        action=action,
        mode="rcs",
    )
    actual = run(case, seed, sample=2000, warm=50000)
    expected = independent_simulation((cv, "rcs", action, False, seed, 2000))
    for key in ["w", "v", "sw", "sv", "little_w", "little_v"]:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
