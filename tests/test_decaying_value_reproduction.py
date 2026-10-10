"""
Reproduction of Master & Bambos (ACC 2015), and the simulation check on the
dynamic program.

The paper's figure 1 is four policy maps, each chosen to show a different
structural behaviour. Reproducing them means reproducing those behaviours --
which rates get used where, and how the optimal rate moves with the backlog and
with the residual value -- so a regression shows up as a contradiction of the
published figure rather than as a changed number.

The second half checks the solver against a simulation of the same system. A
recursion that returns a cost is easy to get subtly wrong in a way no internal
consistency check would catch, so the realised cost is the reference that
matters.
"""

import numpy as np
import pytest

from most_queue.sim.decaying_value import (
    DecayingValueSim,
    compare_with_optimal,
    constant_rate_policy,
)
from most_queue.theory.decaying_value import DecayingValueRateControl

# The paper's common figure-1 parameters: h(b) = b, c(s) = 5 ln(1/(1-s)),
# V = 10, B = 20. Only the reward and the rate set change between panels.
BASE = {
    "num_jobs": 20,
    "initial_value": 10,
    "service_cost": lambda s: 5 * np.log(1 / (1 - s)),
    "holding_cost": lambda b: b,
}

PANELS = {
    "1a": {"reward": lambda v: v, "rates": [0.1, 0.5, 0.9]},
    "1b": {"reward": lambda v: v / 10 + 25, "rates": [0.6, 0.7, 0.8]},
    "1c": {"reward": lambda v: v / 10 + 20, "rates": [0.6, 0.7, 0.9]},
    "1d": {"reward": lambda v: 5 * np.log(1 + v), "rates": [0.700, 0.705, 0.710]},
}


def _panel(name):
    return DecayingValueRateControl(**BASE, **PANELS[name])


# --------------------------------------------------------------------------
# figure 1: the four policy maps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PANELS))
def test_theorem1_holds_in_every_panel(name):
    """
    "For each v, b -> mu(b,v) is non-decreasing." The paper states this holds
    in all four panels, and Theorem 1 says it must.
    """
    assert _panel(name).policy_monotonicity()["backlog_non_decreasing"]


def test_panel_1a_rate_rises_with_both_backlog_and_value():
    """
    Figure 1a, r(v) = v: "b -> mu(b,v) is non-decreasing for all v and
    v -> mu(b,v) is non-decreasing for all b." The server both tries harder
    when more work is waiting and gives up on a job as its value decays.
    """
    observed = _panel("1a").policy_monotonicity()
    assert observed["backlog_non_decreasing"]
    assert np.all(observed["value_non_decreasing"][1:])


def test_panel_1b_rate_falls_as_value_grows():
    """
    Figure 1b, r(v) = v/10 + 25: "v -> mu(b,v) is non-increasing for all b."
    The server still tries harder with more jobs waiting, but here it also
    tries harder as the head-of-line job DECAYS -- the opposite reflex to 1a,
    and the paper's point that neither direction is automatic.
    """
    observed = _panel("1b").policy_monotonicity()
    assert observed["backlog_non_decreasing"]
    assert np.all(observed["value_non_increasing"][1:])
    assert not np.all(observed["value_non_decreasing"][1:])


def test_panel_1c_direction_depends_on_the_backlog():
    """
    Figure 1c, r(v) = v/10 + 20: "the monotonicity of v -> mu(b,v) varies with
    b." Whether it is worth finishing a nearly-dead job depends on how many
    others are waiting.
    """
    observed = _panel("1c").policy_monotonicity()
    assert observed["backlog_non_decreasing"]
    rising, falling = observed["value_non_decreasing"][1:], observed["value_non_increasing"][1:]
    assert not np.all(rising) and not np.all(falling)
    assert rising.any() or falling.any()


def test_panel_1d_is_not_monotone_in_value_at_all():
    """
    Figure 1d, r(v) = 5 ln(1+v) on a narrow rate set: "v -> mu(b,v) is not
    necessarily monotone in anyway; note that v -> mu(5,v) is neither
    non-decreasing nor non-increasing." That specific state is named in the
    caption, so it is asserted specifically.
    """
    control = _panel("1d")
    result = control.solve()
    assert control.policy_monotonicity(result)["backlog_non_decreasing"]
    row = result.policy[5, 1 : control.initial_value + 1]
    assert not np.all(np.diff(row) >= 0)
    assert not np.all(np.diff(row) <= 0)


@pytest.mark.parametrize("name", list(PANELS))
def test_every_panel_actually_uses_its_whole_rate_set(name):
    """Otherwise the panel would not be showing what it claims to show."""
    control = _panel(name)
    used = set(np.unique(control.solve().policy[1:, 1:]))
    assert used == set(control.rates.tolist()), (name, sorted(used))


@pytest.mark.parametrize("name", list(PANELS))
def test_the_two_solution_routes_agree_in_every_panel(name):
    control = _panel(name)
    direct, bellman = control.solve("direct"), control.solve("bellman")
    assert np.allclose(direct.cost_to_go, bellman.cost_to_go)
    assert np.allclose(direct.policy, bellman.policy)


@pytest.mark.parametrize("name", list(PANELS))
def test_the_algebraic_conditions_never_overclaim(name):
    """
    Theorems 2 and 3 are sufficient, not necessary, so the only possible error
    is a condition firing where the policy is not in fact monotone.
    """
    control = _panel(name)
    result = control.solve()
    predicted = control.monotonicity_conditions(result)
    observed = control.policy_monotonicity(result)
    for b in range(1, control.num_jobs + 1):
        if predicted["value_non_decreasing"][b]:
            assert observed["value_non_decreasing"][b], (name, b)
        if predicted["value_non_increasing"][b]:
            assert observed["value_non_increasing"][b], (name, b)


# --------------------------------------------------------------------------
# the dynamic program against a simulation of the same system
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PANELS))
def test_simulated_cost_matches_the_dynamic_program(name):
    """The check that matters: does the predicted cost actually get incurred?"""
    control = _panel(name)
    predicted = control.solve().total_cost
    simulated = DecayingValueSim(control, seed=11).run(replications=20_000)
    sigmas = abs(simulated.mean_cost - predicted) / simulated.cost_error
    assert sigmas < 4.0, (name, predicted, simulated.mean_cost, simulated.cost_error)


@pytest.mark.parametrize("rate", [0.1, 0.5, 0.9])
def test_simulated_cost_matches_the_evaluation_of_a_fixed_policy(rate):
    """
    ``evaluate`` is the same sweep as the solver without the minimisation, so
    it needs its own check -- an error there would quietly flatter or damn
    every baseline.
    """
    control = _panel("1a")
    report = compare_with_optimal(control, policy=constant_rate_policy(control, rate), replications=20_000, seed=5)
    assert report["sigmas"] < 4.0, report


def test_an_ejection_free_regime_is_reported_as_such():
    """At the fastest rates almost nothing decays all the way to zero."""
    control = _panel("1a")
    simulated = DecayingValueSim(control, policy=constant_rate_policy(control, 0.9), seed=3).run(5_000)
    assert simulated.ejected_fraction < 0.01
    assert simulated.completed_fraction > 0.99
    assert simulated.completed_fraction + simulated.ejected_fraction == pytest.approx(1.0)


def test_a_crawling_server_loses_most_jobs_to_decay():
    """The opposite regime, as a check that ejections are counted at all."""
    control = _panel("1a")
    simulated = DecayingValueSim(control, policy=constant_rate_policy(control, 0.1), seed=3).run(5_000)
    assert simulated.ejected_fraction > 0.3
    assert simulated.completed_fraction + simulated.ejected_fraction == pytest.approx(1.0)


def test_simulator_rejects_a_malformed_policy():
    control = _panel("1a")
    with pytest.raises(ValueError, match="shape"):
        DecayingValueSim(control, policy=np.zeros((2, 2)))
    bad = control.solve().policy.copy()
    bad[1, 1] = 2.0
    with pytest.raises(ValueError, match="probability"):
        DecayingValueSim(control, policy=bad)
    with pytest.raises(ValueError, match="replications"):
        DecayingValueSim(control, seed=0).run(replications=0)


# --------------------------------------------------------------------------
# what the optimal control is worth -- our own question, not the paper's
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PANELS))
def test_nothing_beats_the_optimum_and_the_optimum_is_attained(name):
    control = _panel(name)
    optimal = control.solve()
    assert control.evaluate(optimal.policy)[20, 10] == pytest.approx(optimal.total_cost)
    for rate in control.rates:
        assert control.evaluate(constant_rate_policy(control, rate))[20, 10] >= optimal.total_cost - 1e-9
    assert control.evaluate(control.myopic_policy())[20, 10] >= optimal.total_cost - 1e-9


def test_a_myopic_rule_can_be_worse_than_not_adapting_at_all():
    """
    On figure 1a the myopic rule costs more than simply running at the best
    fixed rate -- its adaptivity points the wrong way.

    The reason is visible in the recursion. The optimal rule ranks rates by
    ``r(v) + sigma(b,v-1)``, and at a backlog of 15 the look-ahead term
    ``sigma`` is worth 15 to 26 while the reward ``r(v)`` is only 1 to 10. A
    rule that drops ``sigma`` is therefore reading a signal several times
    smaller than the real one, and slows down on low-value jobs exactly when
    the cost of holding everything behind them argues for clearing them.
    """
    control = _panel("1a")
    optimal = control.solve()
    myopic = control.evaluate(control.myopic_policy())[20, 10]
    best_constant = min(control.evaluate(constant_rate_policy(control, rate))[20, 10] for rate in control.rates)
    assert myopic > best_constant
    assert myopic > 1.5 * optimal.total_cost

    sigma = optimal.sigma[15, 0 : control.initial_value]
    rewards = np.array([float(control.reward(v)) for v in range(1, control.initial_value + 1)])
    assert sigma.max() > 2 * rewards.max() / 3  # the dropped term dominates


def test_a_near_constant_reward_makes_the_myopic_rule_nearly_optimal():
    """
    The flip side, and the condition to remember: when the reward barely
    varies over its range -- figure 1b has r between 25.1 and 26.0 -- dropping
    the look-ahead term hardly changes which rate wins.
    """
    control = _panel("1b")
    optimal = control.solve().total_cost
    myopic = control.evaluate(control.myopic_policy())[20, 10]
    assert abs(myopic - optimal) < 0.15 * abs(optimal)
