"""Independent occupation LP, exhaustive policies, quadrature and timestamps."""

import inspect
import math
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import linprog

from most_queue.sim.smdp_batching import BatchSmdpSim
from most_queue.theory.batch import BatchSmdpCalc


def model(n=12, rate=1, weight=1, penalty=100):
    calc = BatchSmdpCalc(n, energy_weight=weight, overflow_cost=penalty)
    calc.set_sources(rate)
    calc.set_servers([0.4, 0.6], [1, 1.3])
    return calc


def independent_epoch(state, action, n, rate, times, energies, weight, penalty):
    """Direct probabilities and rewards, not production helpers."""
    effective = min(state, n)
    row = np.zeros(n + 2)
    if action:
        period = times[action - 1]
        mean = rate * period
        base = effective - action
        probabilities = [math.exp(-mean) * mean**k / math.factorial(k) for k in range(n - base + 1)]
        row[base : n + 1] = probabilities
        row[-1] = max(0, 1 - math.fsum(probabilities))
        area = effective * period + rate * period**2 / 2
        energy = energies[action - 1]
    else:
        period = 1 / rate
        row[min(state + 1, n + 1)] = 1
        area, energy = effective * period, 0
    cost = area / rate + weight * energy + (penalty * period if state == n + 1 else 0)
    return row, period, cost


def occupation_oracle(n, rate, weight, penalty):
    """SMDP occupation rates: flow=0, sum(x*time)=1."""
    columns, rewards, periods = [], [], []
    for s in range(n + 2):
        for a in range(min(s, 2) + 1):
            row, period, cost = independent_epoch(s, a, n, rate, [0.4, 0.6], [1, 1.3], weight, penalty)
            row[s] -= 1
            columns.append(row)
            rewards.append(cost)
            periods.append(period)
    matrix = np.vstack([np.array(columns).T, periods])
    result = linprog(rewards, A_eq=matrix, b_eq=np.r_[np.zeros(n + 2), 1], bounds=(0, None), method="highs")
    assert result.success, result.message
    return result.fun


@pytest.mark.parametrize("n,rate,weight,penalty", [(2, 0.2, 0, 1), (3, 1, 1, 20), (4, 2, 10, 100), (6, 1, 50, 0)])
def test_independent_occupation_lp(n, rate, weight, penalty):
    result = model(n, rate, weight, penalty).run(tolerance=1e-10, max_iterations=30000)
    oracle = occupation_oracle(n, rate, weight, penalty)
    assert result.converged
    assert result.average_cost == pytest.approx(oracle, abs=2e-8)
    assert result.gain_lower - 2e-8 <= oracle <= result.gain_upper + 2e-8
    assert result.bellman_residual < 1e-8 and result.balance_residual < 1e-12


@pytest.mark.parametrize("rate,weight,penalty", [(0.2, 0, 0), (1, 1, 5), (2, 20, 30)])
def test_exhaustive_small_policies(rate, weight, penalty):
    calc = model(2, rate, weight, penalty)
    costs = []
    for policy in product([0], [0, 1], [0, 1, 2], [0, 1, 2]):
        rows, periods, rewards = zip(
            *(independent_epoch(s, a, 2, rate, [0.4, 0.6], [1, 1.3], weight, penalty) for s, a in enumerate(policy))
        )
        matrix = np.vstack([np.array(rows).T - np.eye(4), np.ones(4)])
        stationary = np.linalg.lstsq(matrix, np.r_[np.zeros(4), 1], rcond=None)[0]
        cost = stationary @ rewards / (stationary @ periods)
        assert calc.evaluate_policy(policy).average_cost == pytest.approx(cost, abs=1e-9)
        costs.append(cost)
    assert calc.run().average_cost == pytest.approx(min(costs), abs=1e-7)


@pytest.mark.parametrize("state,action", [(0, 0), (2, 0), (4, 1), (4, 2), (5, 0), (5, 2)])
def test_population_integral_quadrature(state, action):
    row, period, cost, area, energy = model(4).epoch(state, action)
    s = min(state, 4)
    if action:
        integral = quad(
            lambda t: sum((s + k) * math.exp(-t) * t**k / math.factorial(k) for k in range(40)), 0, period
        )[0]
    else:
        integral = quad(lambda t: s * t * math.exp(-t), 0, np.inf)[0]
    assert area == pytest.approx(integral)
    assert cost == pytest.approx(integral + energy + (100 * period if state == 5 else 0))
    assert sum(row) == pytest.approx(1)


def test_overflow_effective_population_is_n():
    calc = model(3)
    regular, t, c, area, _ = calc.epoch(3, 2)
    overflow, to, co, ao, _ = calc.epoch(4, 2)
    assert overflow == pytest.approx(regular)
    assert ao == area and to == t
    assert co - c == pytest.approx(100 * t)
    assert calc.epoch(4, 0)[0][-1] == 1


@pytest.mark.parametrize("eta", [0.04, 0.2, 0.36])
def test_uniformization_bias_scaling(eta):
    calc = model(6, weight=2)
    result = calc.run(eta=eta, tolerance=1e-10, max_iterations=30000)
    assert result.converged
    assert result.average_cost == pytest.approx(calc.run().average_cost, abs=1e-7)
    for s in range(8):
        choices = []
        for a in range(min(s, 2) + 1):
            row, period, cost, _, _ = calc.epoch(s, a)
            choices.append(cost - result.average_cost * period + row @ result.bias - result.bias[s])
        assert abs(min(choices)) < 1e-8


def test_compressed_operator_against_dense_transformation():
    calc = model(7)
    h = np.arange(9, dtype=float) ** 2 / 3
    eta = 0.36
    raw, _, _ = calc._operator(h, calc._prepare(), eta)
    for s in range(9):
        choices = []
        for a in range(min(s, 2) + 1):
            row, period, cost, _, _ = calc.epoch(s, a)
            transformed = eta * row / period
            transformed[s] += 1 - eta / period
            assert min(transformed) >= 0 and sum(transformed) == pytest.approx(1)
            choices.append(cost / period + transformed @ h)
        assert raw[s] == pytest.approx(min(choices))


def test_time_weighting_and_cost_decomposition():
    result = model(4, weight=2, penalty=7).run()
    assert result.decision_pmf != pytest.approx(result.interval_time_weights)
    assert sum(result.interval_time_weights) == pytest.approx(1)
    assert result.average_cost == pytest.approx(
        result.response_proxy + 2 * result.power + 7 * result.overflow_time_probability
    )
    assert result.w is None and result.v is None and result.p is None


def test_small_tail_contribution_does_not_certify_real_stability():
    calc = BatchSmdpCalc(2, delay_weight=1e-8, energy_weight=100, overflow_cost=0)
    calc.set_sources(0.1)
    calc.set_servers([1, 1], [1, 1])
    result = calc.run(tolerance=1e-12)
    assert result.converged and result.policy[-1] == 0
    assert result.overflow_cost_rate < 0.001 and result.average_cost == pytest.approx(2e-7)
    assert not result.tail_extension_stable and result.tail_drift == 1


def test_selected_tail_action_matters():
    calc = model(2, rate=3)
    assert not calc.evaluate_policy([0, 1, 2, 1]).tail_extension_stable
    assert calc.evaluate_policy([0, 1, 2, 2]).tail_extension_stable


def test_iteration_limit_and_reconfiguration():
    calc = model()
    assert not calc.run(max_iterations=1).converged
    before = calc.run().average_cost
    calc.set_sources(0.2)
    calc.set_servers([0.8, 1.2], [2, 2.6])
    assert calc.run().average_cost != pytest.approx(before)


@pytest.mark.parametrize("n", [True, 0, -1, 1.5, 2001])
def test_invalid_state_bound(n):
    with pytest.raises(ValueError):
        BatchSmdpCalc(n)


@pytest.mark.parametrize("rate", [True, 0, -1, float("inf"), float("nan")])
def test_invalid_rate(rate):
    with pytest.raises(ValueError):
        BatchSmdpCalc().set_sources(rate)


@pytest.mark.parametrize(
    "times,energies",
    [
        ([], []),
        ([0], [1]),
        ([-1], [1]),
        ([1], [-1]),
        ([1], []),
        ([True], [1]),
        ([float("nan")], [1]),
        ([1] * 129, [1] * 129),
    ],
)
def test_invalid_profiles(times, energies):
    with pytest.raises(ValueError):
        BatchSmdpCalc(200).set_servers(times, energies)


@pytest.mark.parametrize(
    "kwargs", [{"tolerance": 0}, {"max_iterations": 0}, {"eta": 0}, {"eta": 1}, {"eta": float("nan")}]
)
def test_invalid_solver_options(kwargs):
    with pytest.raises(ValueError):
        model().run(**kwargs)


def test_overload_and_missing_inputs():
    with pytest.raises(ValueError):
        BatchSmdpCalc().run()
    with pytest.raises(ValueError):
        model(rate=4).run()


@pytest.mark.parametrize("policy", [[], [0] * 13, [0, 3] + [2] * 12, [0, True] + [2] * 12, [0, -1] + [2] * 12])
def test_invalid_policy(policy):
    with pytest.raises(ValueError):
        model().evaluate_policy(policy)


def simulator():
    sim = BatchSmdpSim()
    sim.set_sources(1)
    sim.set_servers([1, 1.5], [1, 3])
    return sim


def test_hand_replay_and_warmup():
    sim = simulator()
    result = sim.run(3, lambda n: min(n, 2), arrivals=[0, 0.1, 0.2, 1.2], record_trace=True)
    assert result.iteration_trace == [
        {"time": 0, "population": 1, "action": 1},
        {"time": 1, "population": 2, "action": 2},
        {"time": 2.5, "population": 1, "action": 1},
    ]
    assert result.arrived == 4 and result.completed == 3 and result.unfinished == 1
    assert result.population_area == pytest.approx(7.5)
    assert result.completed_residence == pytest.approx(5.7)
    assert result.unfinished_residence == pytest.approx(1.8)
    assert result.energy == 5 and result.power == pytest.approx(5 / 3)
    assert result.busy_fraction == 1
    warmed = sim.run(3, lambda n: min(n, 2), arrivals=[0, 0.1, 0.2, 1.2], warmup=0.5)
    assert warmed.population_area == pytest.approx(6.3) and warmed.power == pytest.approx(4 / 2.5)
    assert warmed.total_population_area == pytest.approx(7.5)


def test_unbounded_tail_population():
    result = simulator().run(0.5, [0, 0, 2, 2], initial_jobs=10, arrivals=[], record_trace=True)
    assert result.iteration_trace[0] == {"time": 0, "population": 10, "action": 2}
    assert result.total_population_area == 5
    assert result.initial_jobs == result.unfinished == 10


def test_seed_and_no_theory_import():
    import most_queue.sim.smdp_batching as module

    sim = simulator()
    first = sim.run(100, lambda n: min(n, 2), record_trace=True)
    second = sim.run(100, lambda n: min(n, 2), record_trace=True)
    assert first.iteration_trace == second.iteration_trace
    assert "most_queue.theory" not in inspect.getsource(module)
    assert first.arrived == first.completed + first.unfinished
    assert first.total_population_area == pytest.approx(first.completed_residence + first.unfinished_residence)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 1, "warmup": 1},
        {"horizon": 1, "arrivals": [1]},
        {"horizon": 1, "arrivals": [0.2, 0.1]},
        {"horizon": 1, "arrivals": [-1]},
        {"horizon": 1, "initial_jobs": -1},
        {"horizon": 1, "max_events": 0},
    ],
)
def test_invalid_simulation(kwargs):
    with pytest.raises(ValueError):
        simulator().run(policy=lambda n: min(n, 2), **kwargs)


def test_event_limit_and_infeasible_action():
    with pytest.raises(ValueError):
        simulator().run(100, lambda n: min(n, 2), max_events=1)
    with pytest.raises(ValueError):
        simulator().run(100, lambda n: 2)
