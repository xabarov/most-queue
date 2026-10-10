"""
Unit tests for service rate control of jobs with decaying value.

Three independent routes to the same answer are kept in play: the paper's
``delta``/``sigma`` recursion (Proposition 1), the Bellman equation written out
directly, and -- here only -- a genuine value iteration that sweeps the state
space in a deliberately unhelpful order until it converges. The third exists
because the first two both exploit the fact that the state space is acyclic;
value iteration does not, so agreeing with it confirms that reading of the
dynamics rather than assuming it.
"""

import numpy as np
import pytest

from most_queue.sim.decaying_value import constant_rate_policy
from most_queue.theory.decaying_value import DecayingValueRateControl


def _standard(**overrides):
    """The paper's figure-1 base instance, with optional overrides."""
    params = {
        "num_jobs": 20,
        "initial_value": 10,
        "rates": [0.1, 0.5, 0.9],
        "service_cost": lambda s: 5 * np.log(1 / (1 - s)),
        "holding_cost": lambda b: b,
        "reward": lambda v: v,
    }
    params.update(overrides)
    return DecayingValueRateControl(**params)


def _random_instance(rng):
    num_jobs = int(rng.integers(1, 12))
    initial_value = int(rng.integers(1, 9))
    rates = sorted({round(float(x), 4) for x in rng.uniform(0.02, 0.95, int(rng.integers(1, 5)))})
    scale = float(rng.uniform(0.5, 8.0))
    holding = float(rng.uniform(0.0, 3.0))
    slope = float(rng.uniform(0.0, 2.0))
    base = float(rng.uniform(0.5, 30.0))
    return DecayingValueRateControl(
        num_jobs=num_jobs,
        initial_value=initial_value,
        rates=rates,
        service_cost=lambda s, k=scale: k * np.log(1 / (1 - s)),
        holding_cost=lambda b, h=holding: h * b,
        reward=lambda v, a=slope, c=base: a * v + c,
    )


def _value_iteration(control, tol=1e-12, max_sweeps=100_000):
    """
    A deliberately naive reference: start from zero and sweep the whole state
    space in reverse order until nothing moves.

    The order is chosen to be the opposite of the one the solver relies on, so
    convergence here is evidence about the fixed point rather than about the
    sweep.
    """
    rows = control.num_jobs + 1
    cols = control.initial_value + 1
    top = control.initial_value
    costs = control._service_costs  # pylint: disable=protected-access
    holding = control._holding_costs  # pylint: disable=protected-access
    rewards = control._rewards  # pylint: disable=protected-access
    rates = control.rates

    cost_to_go = np.zeros((rows, cols))
    for _ in range(max_sweeps):
        shift = 0.0
        for b in range(rows - 1, 0, -1):
            for v in range(cols - 1, 0, -1):
                ejected = cost_to_go[b - 1, top]
                continuation = cost_to_go[b, v - 1] if v > 1 else ejected
                objective = costs + holding[b] + rates * (-rewards[v] + ejected) + (1.0 - rates) * continuation
                new = float(np.min(objective))
                shift = max(shift, abs(new - cost_to_go[b, v]))
                cost_to_go[b, v] = new
        if shift < tol:
            return cost_to_go
    raise AssertionError("value iteration did not converge")


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -3])
def test_batch_size_must_be_positive(bad):
    with pytest.raises(ValueError, match="num_jobs"):
        _standard(num_jobs=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_initial_value_must_be_positive(bad):
    with pytest.raises(ValueError, match="initial_value"):
        _standard(initial_value=bad)


def test_rate_set_must_be_non_empty_distinct_and_in_the_unit_interval():
    with pytest.raises(ValueError, match="at least one"):
        _standard(rates=[])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _standard(rates=[0.5, 1.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _standard(rates=[-0.1, 0.5])
    with pytest.raises(ValueError, match="distinct"):
        _standard(rates=[0.5, 0.5])


def test_service_cost_must_be_finite_on_the_rate_grid():
    """c(s) = 5 ln(1/(1-s)) blows up at s = 1, which the paper excludes."""
    with np.errstate(divide="ignore"):
        with pytest.raises(ValueError, match="finite"):
            _standard(rates=[0.5, 1.0])


def test_costs_must_be_non_decreasing_and_rewards_positive():
    with pytest.raises(ValueError, match="service_cost must be non-decreasing"):
        _standard(service_cost=lambda s: 1.0 - s)
    with pytest.raises(ValueError, match="holding_cost must be non-decreasing"):
        _standard(holding_cost=lambda b: 100.0 - b)
    with pytest.raises(ValueError, match="reward must be strictly positive"):
        _standard(reward=lambda v: 0.0)
    with pytest.raises(ValueError, match="reward must be non-decreasing"):
        _standard(reward=lambda v: 20.0 - v)


def test_validation_can_be_switched_off_for_exploration():
    """Outside the model's assumptions the solver still runs; Theorem 1 may not hold."""
    control = _standard(holding_cost=lambda b: 100.0 - b, validate=False)
    result = control.solve()
    assert np.all(np.isfinite(result.cost_to_go))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        _standard().solve(method="policy_iteration")


# --------------------------------------------------------------------------
# the three solution routes agree
# --------------------------------------------------------------------------


def test_direct_and_bellman_agree_on_the_paper_instance():
    control = _standard()
    direct = control.solve("direct")
    bellman = control.solve("bellman")
    assert np.allclose(direct.cost_to_go, bellman.cost_to_go)
    assert np.allclose(direct.policy, bellman.policy)
    assert np.allclose(direct.sigma, bellman.sigma)
    assert np.allclose(direct.delta, bellman.delta)


@pytest.mark.parametrize("seed", range(25))
def test_direct_and_bellman_agree_on_random_instances(seed):
    control = _random_instance(np.random.default_rng(seed))
    direct = control.solve("direct")
    bellman = control.solve("bellman")
    assert np.allclose(direct.cost_to_go, bellman.cost_to_go, atol=1e-9)
    assert np.allclose(direct.policy, bellman.policy)


@pytest.mark.parametrize("seed", range(8))
def test_value_iteration_confirms_the_acyclic_sweep(seed):
    """
    The solver exploits the state space being acyclic. Value iteration does
    not, so this checks that reading of the dynamics.
    """
    control = _random_instance(np.random.default_rng(100 + seed))
    assert np.allclose(control.solve("direct").cost_to_go, _value_iteration(control), atol=1e-8)


def test_value_iteration_confirms_the_paper_instance():
    control = _standard()
    assert np.allclose(control.solve("direct").cost_to_go, _value_iteration(control), atol=1e-8)


# --------------------------------------------------------------------------
# closed forms in the small corners
# --------------------------------------------------------------------------


def test_single_job_single_slot_matches_the_hand_formula():
    """
    With B = V = 1 there is exactly one decision: J(1,1) = h(1) + min_s{c(s) - s*r(1)}.
    """
    control = DecayingValueRateControl(
        num_jobs=1,
        initial_value=1,
        rates=[0.2, 0.6],
        service_cost=lambda s: 3.0 * s,
        holding_cost=lambda b: 2.0 * b,
        reward=lambda v: 7.0,
    )
    expected = 2.0 + min(3.0 * s - s * 7.0 for s in (0.2, 0.6))
    result = control.solve()
    assert result.total_cost == pytest.approx(expected)
    assert result.rate(1, 1) == pytest.approx(0.6)  # 3s - 7s is decreasing, so the fastest


def test_two_slot_recursion_matches_the_hand_formula():
    """B = 1, V = 2: check delta, sigma and J term by term."""
    rates = (0.2, 0.6)
    holding, reward1, reward2 = 2.0, 7.0, 9.0

    def cost(s):
        return 3.0 * s

    control = DecayingValueRateControl(
        num_jobs=1,
        initial_value=2,
        rates=list(rates),
        service_cost=cost,
        holding_cost=lambda b: holding * b,
        reward=lambda v: reward1 if v == 1 else reward2,
    )
    delta1 = holding + min(cost(s) - s * reward1 for s in rates)
    delta2 = holding + min(cost(s) - s * (reward2 + delta1) for s in rates)
    result = control.solve()
    assert result.delta[1, 1] == pytest.approx(delta1)
    assert result.delta[1, 2] == pytest.approx(delta2)
    assert result.sigma[1, 2] == pytest.approx(delta1 + delta2)
    assert result.cost_to_go[1, 1] == pytest.approx(delta1)
    assert result.total_cost == pytest.approx(delta1 + delta2)


def test_a_single_admissible_rate_leaves_no_choice():
    control = _standard(rates=[0.4])
    result = control.solve()
    assert np.all(result.policy[1:, 1:] == pytest.approx(0.4))
    assert np.allclose(result.cost_to_go, control.evaluate(constant_rate_policy(control, 0.4)))


def test_free_service_runs_flat_out():
    """Speed costs nothing and reward is positive: always take the fastest rate."""
    control = _standard(service_cost=lambda s: 0.0)
    assert np.all(control.solve().policy[1:, 1:] == pytest.approx(0.9))


def test_nothing_to_gain_and_no_hurry_means_the_slowest_rate():
    """
    Speed costs money, the reward is negligible AND holding is free, so there
    is no reason to rush. Note that the holding cost has to go too: with
    h(b) = b the server still hurries to clear the backlog even when finishing
    a job earns nothing, which is why this corner is less obvious than it looks.
    """
    control = _standard(
        reward=lambda v: 1e-9,
        service_cost=lambda s: 10.0 * s,
        holding_cost=lambda b: 0.0,
    )
    assert np.all(control.solve().policy[1:, 1:] == pytest.approx(0.1))


def test_holding_cost_alone_is_enough_to_make_the_server_hurry():
    """The flip side: no reward at all, but waiting is expensive."""
    control = _standard(
        reward=lambda v: 1e-9,
        service_cost=lambda s: 0.5 * s,
        holding_cost=lambda b: 10.0 * b,
    )
    policy = control.solve().policy[1:, 1:]
    assert np.any(policy > 0.1), "a positive holding cost should buy some speed"


def test_with_no_decay_horizon_the_optimal_rule_is_exactly_myopic():
    """
    At V = 1 the look-ahead term sigma(b, 0) is zero by definition, so the
    optimal rule sees only r(1) -- which is what the myopic rule sees.
    """
    control = _standard(initial_value=1)
    assert np.allclose(control.solve().policy, control.myopic_policy())


# --------------------------------------------------------------------------
# optimality and policy evaluation
# --------------------------------------------------------------------------


def test_evaluating_the_optimal_policy_reproduces_the_cost_to_go():
    control = _standard()
    result = control.solve()
    assert np.allclose(control.evaluate(result.policy), result.cost_to_go)


@pytest.mark.parametrize("seed", range(12))
def test_no_policy_beats_the_optimum_anywhere(seed):
    """The cost-to-go is a lower bound over all policies, state by state."""
    rng = np.random.default_rng(500 + seed)
    control = _random_instance(rng)
    optimal = control.solve()
    shape = (control.num_jobs + 1, control.initial_value + 1)
    arbitrary = np.zeros(shape)
    arbitrary[1:, 1:] = rng.choice(control.rates, size=(shape[0] - 1, shape[1] - 1))
    assert np.all(control.evaluate(arbitrary)[1:, 1:] >= optimal.cost_to_go[1:, 1:] - 1e-9)
    assert np.all(control.evaluate(control.myopic_policy())[1:, 1:] >= optimal.cost_to_go[1:, 1:] - 1e-9)
    for rate in control.rates:
        constant = control.evaluate(constant_rate_policy(control, rate))
        assert np.all(constant[1:, 1:] >= optimal.cost_to_go[1:, 1:] - 1e-9)


def test_evaluate_rejects_a_malformed_policy():
    control = _standard()
    with pytest.raises(ValueError, match="shape"):
        control.evaluate(np.zeros((3, 3)))
    bad = control.solve().policy.copy()
    bad[1, 1] = 1.5
    with pytest.raises(ValueError, match="probability"):
        control.evaluate(bad)


def test_constant_rate_policy_rejects_an_inadmissible_rate():
    with pytest.raises(ValueError, match="not among the admissible"):
        constant_rate_policy(_standard(), 0.42)


# --------------------------------------------------------------------------
# the monotonicity theorems
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_theorem1_holds_on_every_admissible_instance(seed):
    """
    "For each v, b -> mu(b,v) is non-decreasing." This needs only that the
    holding cost is non-decreasing, so it must hold everywhere in the model.
    """
    control = _random_instance(np.random.default_rng(seed))
    assert control.policy_monotonicity()["backlog_non_decreasing"]


@pytest.mark.parametrize("seed", range(40))
def test_theorems2_and_3_never_predict_a_monotonicity_the_policy_lacks(seed):
    """
    The algebraic conditions are sufficient, not necessary, so the only thing
    that can be wrong is a condition firing where the policy is not monotone.
    """
    control = _random_instance(np.random.default_rng(seed))
    result = control.solve()
    predicted = control.monotonicity_conditions(result)
    observed = control.policy_monotonicity(result)
    for b in range(1, control.num_jobs + 1):
        if predicted["value_non_decreasing"][b]:
            assert observed["value_non_decreasing"][b], (seed, b)
        if predicted["value_non_increasing"][b]:
            assert observed["value_non_increasing"][b], (seed, b)


def test_constant_reward_selects_theorem3():
    constant = _standard(reward=lambda v: 12.0)
    assert constant.monotonicity_conditions()["constant_reward"]
    varying = _standard(reward=lambda v: v)
    assert not varying.monotonicity_conditions()["constant_reward"]


def test_theorem3_sign_test_matches_its_statement():
    """
    With a constant reward the test is the sign of h(b) + min_s{c(s) - s*r};
    a large holding cost makes it positive, a large reward makes it negative.
    """
    expensive = _standard(reward=lambda v: 1.0, holding_cost=lambda b: 50.0 * b)
    conditions = expensive.monotonicity_conditions()
    assert np.all(conditions["value_non_decreasing"][1:])

    rewarding = _standard(reward=lambda v: 500.0, holding_cost=lambda b: 0.0)
    conditions = rewarding.monotonicity_conditions()
    assert np.all(conditions["value_non_increasing"][1:])


def test_monotonicity_helpers_solve_on_demand():
    control = _standard()
    assert control.monotonicity_conditions()["backlog_non_decreasing"]
    assert control.policy_monotonicity()["backlog_non_decreasing"]


# --------------------------------------------------------------------------
# result plumbing
# --------------------------------------------------------------------------


def test_result_reports_the_method_and_the_whole_batch_cost():
    control = _standard()
    result = control.solve("bellman")
    assert result.method == "bellman"
    assert result.total_cost == pytest.approx(result.cost_to_go[20, 10])
    assert result.duration >= 0.0
    assert result.rate(15, 4) == pytest.approx(result.policy[15, 4])


def test_repr_mentions_the_instance_size():
    assert "num_jobs=20" in repr(_standard())
