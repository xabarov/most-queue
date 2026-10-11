"""Independent rational map, stage Jacobian, and exact Schur arbiters."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from most_queue.sim.kv_cache_congestion import KvCacheCongestionSim
from most_queue.theory.continuous_batching import KvCacheCongestionCalc


def calc(memory=24, inputs=(2,), decodes=(3,), proportions=(1,)):
    """Configured calculator fixture."""
    model = KvCacheCongestionCalc(memory)
    model.set_sources(proportions)
    model.set_servers(inputs, decodes)
    return model


def rational_step(state, inputs, decodes, proportions, capacity, cap=None):
    """Exact rational class-stage transition; no numerical/theory helpers."""
    shifted = [[Fraction(0)] + list(row[:-1]) for row in state]
    used = sum((inputs[k] + j + 1) * mass for k, row in enumerate(shifted) for j, mass in enumerate(row))
    for stage in range(1, max(decodes)):
        overflow = used - capacity
        if overflow <= 0:
            break
        present = [k for k in range(len(inputs)) if stage < decodes[k]]
        released = sum((inputs[k] + stage + 1) * shifted[k][stage] for k in present)
        if not released:
            continue
        removal_fraction = min(Fraction(1), overflow / released)
        for k in present:
            shifted[k][stage] *= 1 - removal_fraction
        used -= released * removal_fraction
    admissions = (capacity - used) / sum(p * (s + 1) for p, s in zip(proportions, inputs))
    if cap is not None:
        admissions = min(admissions, cap)
    for k, probability in enumerate(proportions):
        shifted[k][0] = probability * admissions
    return shifted


def schur(coefficients):
    """Real Schur recursion with Fraction arithmetic (no root solver)."""
    values = list(map(Fraction, coefficients))
    while len(values) > 1:
        first, last = values[0], values[-1]
        if abs(last) >= abs(first):
            return False
        reduced = [first * values[j] - last * values[-1 - j] for j in range(len(values) - 1)]
        values = [value / reduced[0] for value in reduced]
    return True


@pytest.mark.parametrize("s,d", product([0, 2, 10], [1, 2, 3, 5, 8]))
def test_homogeneous_equilibrium_and_cycles(s, d):
    memory = 20 * (s + d)
    model = calc(memory, [s], [d])
    result = model.run()
    area = sum(s + 1 + j for j in range(d))
    assert result.nominal_throughput == pytest.approx(memory / area)
    assert result.worst_cycle_throughput == pytest.approx(memory / ((s + d) * d))
    cycle = [[0.0] * d]
    cycle[0][0] = memory / (s + 1)
    trajectory = model.evolve(d * 4, initial_state=cycle, warmup=d, record_trace=True)
    assert trajectory.throughput == pytest.approx(result.worst_cycle_throughput)
    for step, entry in enumerate(trajectory.iteration_trace):
        stage = (step + 1) % d
        expected = [0.0] * d
        expected[stage] = memory / (s + 1 + stage)
        assert entry["state"][0] == pytest.approx(expected, abs=1e-10)
        assert entry["memory"] == pytest.approx(memory)
    if d > 1:
        assert result.spectral_radius > 1
    else:
        assert result.spectral_radius == 0
        assert result.worst_cycle_ratio == 1


def test_example_cascade_and_accounting():
    result = calc().evolve(24, initial_state=[[2.5, 2, 1.7]], warmup=18, record_trace=True)
    first_eviction = next(item for item in result.iteration_trace if item["evicted"] > 1e-10)
    assert first_eviction["iteration"] + 1 == 7
    assert first_eviction["state"][0] == pytest.approx([0, 778 / 243, 544 / 243])
    assert result.iteration_trace[15]["state"][0] == pytest.approx([0, 0, 4.8])
    assert result.throughput == pytest.approx(1.6)
    assert result.initial_work + result.executed_work == pytest.approx(
        result.completed_work + result.discarded_work + result.remaining_work
    )
    assert result.w is None and result.p is None


def test_balanced_intermediate_orbit_and_exact_capture():
    model = calc()
    initial = [[48 / 13, 24 / 13, 72 / 65]]
    result = model.evolve(7, initial_state=initial, record_trace=True)
    assert result.iteration_trace[0]["state"][0] == pytest.approx([0, 48 / 13, 24 / 13])
    assert result.iteration_trace[3]["state"][0] == pytest.approx([0, 48 / 13, 24 / 13], abs=1e-12)
    # Not every initial state reaches maximal eviction: exceptional balanced captures exist.
    assert all(sum(value > 1e-10 for value in entry["state"][0]) == 2 for entry in result.iteration_trace)


@pytest.mark.parametrize(
    "inputs,decodes,probabilities",
    [
        ([2], [3], [Fraction(1)]),
        ([2, 5], [3, 4], [Fraction(1, 3), Fraction(2, 3)]),
        ([1, 3, 7], [2, 4, 5], [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]),
    ],
)
@pytest.mark.parametrize("limited", [False, True])
def test_rational_map_matches_all_steps(inputs, decodes, probabilities, limited):
    capacity = Fraction(120)
    raw = [[Fraction(j + k + 1) for j in range(d)] for k, d in enumerate(decodes)]
    used = sum((inputs[k] + 1 + j) * x for k, row in enumerate(raw) for j, x in enumerate(row))
    state = [[x * capacity / used for x in row] for row in raw]
    model = calc(120, inputs, decodes, list(map(float, probabilities)))
    cap = Fraction(3, 2) if limited else None
    result = model.evolve(
        30,
        initial_state=[[float(x) for x in row] for row in state],
        admission_cap=float(cap) if cap is not None else None,
        record_trace=True,
    )
    for entry in result.iteration_trace:
        state = rational_step(state, inputs, decodes, probabilities, capacity, cap)
        for exact, computed in zip(state, entry["state"]):
            assert computed == pytest.approx(list(map(float, exact)), rel=1e-10, abs=1e-10)
        assert entry["memory"] <= 120 + 1e-10


def stage_radius(inputs, decodes, probabilities):
    """Direct physical-state derivative, independent of polynomial coefficients."""
    # Direct differentiated shift + proportional memory-fill, not a companion from F.
    states = [(k, j) for k, d in enumerate(decodes) for j in range(d)]
    indices = {state: index for index, state in enumerate(states)}
    jacobian = np.zeros((len(states), len(states)))
    cost = sum(p * (s + 1) for p, s in zip(probabilities, inputs))
    for column, (kind, stage) in enumerate(states):
        if stage + 1 < decodes[kind]:
            jacobian[indices[kind, stage + 1], column] = 1
            for k, probability in enumerate(probabilities):
                jacobian[indices[k, 0], column] = -probability * (inputs[kind] + stage + 2) / cost
    return max(abs(np.linalg.eigvals(jacobian)))


@pytest.mark.parametrize(
    "inputs,decodes,probabilities",
    [
        ([2], [3], [1]),
        ([50, 50], [2, 3], [0.5, 0.5]),
        ([50, 50], [2, 4], [0.5, 0.5]),
        ([2, 10], [3, 5], [0.3, 0.7]),
        ([2, 7, 11], [2, 4, 7], [0.2, 0.3, 0.5]),
    ],
)
def test_spectrum_matches_full_class_stage_jacobian(inputs, decodes, probabilities):
    radius = stage_radius(inputs, decodes, probabilities)
    result = calc(10000, inputs, decodes, probabilities).run()
    assert result.spectral_radius == pytest.approx(radius, abs=1e-9)
    assert result.root_residual < 1e-10


def test_limiting_polynomials_and_heterogeneous_ratios():
    result = calc(1000, [50, 50], [2, 3], [0.5, 0.5]).run()
    assert result.limiting_coefficients == [1, 1, 0.5]
    assert result.limiting_spectral_radius == pytest.approx(1 / np.sqrt(2))
    result = calc(1000, [50, 50], [2, 4], [0.5, 0.5]).run()
    assert result.limiting_coefficients == [1, 1, 0.5, 0.5]
    assert result.limiting_spectral_radius == pytest.approx(1)
    model = calc(1000, [10, 30], [2, 3], [0.5, 0.5])
    model.set_servers([10, 30], [2, 3], input_ratios=[1, 3])
    assert model.run().limiting_coefficients == [2, 2, 1.5]


def test_finite_input_caption_threshold_requires_independent_arbiters():
    # Eq.(10) with p=.5, (2,7): the literal Fig.3 caption's l0=15 is not Schur-stable.
    statuses = []
    for s in range(20):
        exact = [Fraction(s + 1 + j) * (1 if j < 2 else Fraction(1, 2)) for j in range(7)]
        result = calc(1000, [s, s], [2, 7], [0.5, 0.5]).run()
        assert result.spectral_radius == pytest.approx(stage_radius([s, s], [2, 7], [0.5, 0.5]), abs=1e-9)
        stable = schur(exact)
        assert stable == (result.spectral_radius < 1)
        statuses.append(stable)
    assert statuses.index(True) == 18
    assert not statuses[15]
    assert calc(1000, [15, 15], [2, 7], [0.5, 0.5]).run().decode_gcd == 1


def test_unit_circle_is_inconclusive_and_not_all_finite_inputs_are_stable():
    result = calc(100, [1, 1], [2, 3], [0.5, 0.5]).run()
    assert result.spectral_radius == pytest.approx(1)
    assert result.linear_stability == "near_boundary"
    assert calc(100, [0, 0], [2, 3], [0.5, 0.5]).run().spectral_radius > 1


def test_rate_cap_after_transient_removes_eviction_and_restores_nominal_flow():
    model = calc()
    result = model.evolve(90, initial_state=[[8, 0, 0]], admission_cap=2, warmup=60, record_trace=True)
    assert all(item["evicted"] < 1e-10 for item in result.iteration_trace[3:])
    assert result.throughput == pytest.approx(2)
    assert result.final_state[0] == pytest.approx([2, 2, 2])
    assert all(item["admitted"] <= 2 for item in result.iteration_trace)


def test_poisson_des_manual_protocol_and_integer_not_fractional():
    simulator = KvCacheCongestionSim(24)
    simulator.set_sources([0])
    simulator.set_servers([2], [3])
    result = simulator.run(2, arrivals=[[5], [0]], initial_state=[[1, 1, 2]], initial_queue=[0] * 8, record_trace=True)
    assert result.iteration_trace[0]["state"] == [[5, 1, 1]]
    assert result.iteration_trace[1]["state"] == [[1, 4, 1]]
    assert result.iteration_trace[1]["post_execution_memory"] == 25
    assert result.final_queue == 8
    assert result.completed == 3 and result.evictions == 1
    assert result.initial_jobs + result.arrived == result.completed + result.unfinished
    assert (
        result.initial_work + result.executed_work
        == result.completed_work + result.discarded_work + result.remaining_work
    )


def test_exact_integer_cycle_independent_of_fluid_formula():
    simulator = KvCacheCongestionSim(60)
    simulator.set_sources([1])
    simulator.set_servers([2], [3])
    result = simulator.run(30, saturated=True, initial_state=[[20, 0, 0]], warmup=3, record_trace=True)
    # Manual integer cycle: 20 at footprint3 -> 15 at4 -> 12 at5 -> 20 at3.
    assert result.throughput == 4
    assert result.iteration_trace[0]["state"] == [[0, 15, 0]]
    assert result.iteration_trace[1]["state"] == [[0, 0, 12]]
    assert result.iteration_trace[2]["state"] == [[20, 0, 0]]
    assert result.initial_jobs + result.arrived == result.completed + result.unfinished


@pytest.mark.parametrize("memory", [True, 0, -1, float("nan"), float("inf")])
def test_bad_memory(memory):
    with pytest.raises(ValueError):
        KvCacheCongestionCalc(memory)


@pytest.mark.parametrize("probabilities", [[], [0], [-1], [0.4], [True], [float("nan")]])
def test_bad_proportions(probabilities):
    with pytest.raises(ValueError):
        KvCacheCongestionCalc(100).set_sources(probabilities)


@pytest.mark.parametrize(
    "inputs,decodes",
    [
        ([], []),
        ([1], [0]),
        ([-1], [3]),
        ([1.1], [3]),
        ([True], [3]),
        ([1], [True]),
        ([98], [3]),
        ([1], [300]),
        ([1], [2, 3]),
    ],
)
def test_bad_lengths(inputs, decodes):
    with pytest.raises(ValueError):
        KvCacheCongestionCalc(100).set_servers(inputs, decodes)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(iterations=0),
        dict(iterations=True),
        dict(iterations=2, warmup=2),
        dict(iterations=2, initial_state=[[9, 0, 0]]),
        dict(iterations=2, initial_state=[[1, -1, 0]]),
        dict(iterations=2, initial_state=[[0]]),
        dict(iterations=2, admission_cap=-1),
    ],
)
def test_invalid_fluid_run(kwargs):
    with pytest.raises(ValueError):
        calc().evolve(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(iterations=0),
        dict(iterations=2, admission_cap=1.5),
        dict(iterations=2, warmup=2),
        dict(iterations=2, arrivals=[[1]]),
        dict(iterations=2, arrivals=[[1.5], [0]]),
        dict(iterations=2, arrivals=[[1_000_001], [0]]),
        dict(iterations=2, initial_state=[[9, 0, 0]]),
        dict(iterations=2, initial_queue=[1]),
        dict(iterations=2, saturated=True, arrivals=[[0], [0]]),
    ],
)
def test_invalid_integer_run(kwargs):
    model = KvCacheCongestionSim(24)
    model.set_sources([1])
    model.set_servers([2], [3])
    with pytest.raises(ValueError):
        model.run(**kwargs)


def test_simulator_is_seed_reproducible_and_has_no_theory_import():
    import inspect

    import most_queue.sim.kv_cache_congestion as module

    simulator = KvCacheCongestionSim(100, seed=7)
    simulator.set_sources([1, 0.5])
    simulator.set_servers([2, 5], [3, 4])
    first = simulator.run(100, record_trace=True)
    second = simulator.run(100, record_trace=True)
    assert first.iteration_trace == second.iteration_trace
    assert "most_queue.theory" not in inspect.getsource(module)
    assert first.initial_jobs + first.arrived == first.completed + first.unfinished
    assert first.executed_work == first.completed_work + first.remaining_work + first.discarded_work
    assert first.w is None and first.v is None and first.p is None
