"""
Unit tests for exact offline scheduling of imprecise computations
(roadmap item R6; Shih, Liu, Chung & Gillies 1989; Shih, Liu & Chung 1991;
Shih & Liu 1995; reviewed by Shioura, Shakhlevich & Strusevich, EJOR 2018).

The model and the solved problems are theirs. These tests check OUR linear
program against references that share no code with it:

1. EXHAUSTIVE search over unit-slot assignments on small integer instances --
   there is no cleverness left to be wrong in a brute force;
2. a minimum-cost maximum-flow solved by network simplex (networkx), which is
   the formulation the literature actually states, so agreeing with it also
   confirms the LP is the same problem;
3. closed forms in the degenerate corners (one task; a common deadline with no
   mandatory parts, where the optimum is a weight-ordered greedy fill);
4. the emitted schedule, re-checked against the model by ``verify_schedule``,
   which knows nothing about how it was produced.
"""

import itertools

import networkx as nx
import numpy as np
import pytest

from most_queue.theory.imprecise import (
    ImpreciseComputationScheduler,
    ImpreciseTask,
    verify_schedule,
)


def _solve(tasks, objective="total"):
    return ImpreciseComputationScheduler().set_tasks(tasks).solve(objective)


def _brute_force(tasks):
    """
    Exhaustive search over unit-slot assignments. Integer data only, tiny
    instances only -- which is the point: it cannot be subtly wrong.

    :return: the minimum total weighted error, or None if the mandatory parts
        cannot all be completed.
    """
    horizon = int(max(t.deadline for t in tasks))
    n = len(tasks)
    best = None
    for assignment in itertools.product(range(n + 1), repeat=horizon):
        done = [0] * n
        valid = True
        for slot, job in enumerate(assignment):
            if job == n:  # idle
                continue
            if not (tasks[job].release <= slot and slot + 1 <= tasks[job].deadline):
                valid = False
                break
            done[job] += 1
        if not valid or any(done[j] < tasks[j].mandatory for j in range(n)):
            continue
        error = sum(tasks[j].weight * max(tasks[j].total - done[j], 0) for j in range(n))
        best = error if best is None else min(best, error)
    return best


def _min_cost_flow(tasks):
    """
    The literature's own formulation: a minimum-cost maximum-flow network, with
    the cost of a unit of flow through task j equal to its weight. Solved here
    by network simplex -- a completely different algorithm from the LP.

    Each task is split into a mandatory and an optional source arc sharing the
    same arcs into the intervals. The mandatory arcs carry a cost so large that
    min-cost flow saturates them before it buys any optional work, which is how
    a hard lower bound is emulated: if the resulting flow still leaves mandatory
    arcs unsaturated, no schedule exists. That reroute argument is valid because
    the mandatory and optional arcs of a task have identical successors.

    Integer data only (network simplex needs integral capacities).

    :return: the minimum total weighted error, or None if infeasible.
    """
    points = sorted({t.release for t in tasks} | {t.deadline for t in tasks})
    intervals = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    scale = 1000
    span = sum(int(b - a) for a, b in intervals)
    big = scale * int(max(t.weight for t in tasks) + 1) * (span + 1)

    graph = nx.DiGraph()
    for j, task in enumerate(tasks):
        graph.add_edge("s", f"m{j}", capacity=int(task.mandatory), weight=-big)
        graph.add_edge("s", f"o{j}", capacity=int(task.optional), weight=-int(task.weight * scale))
        for k, (a, b) in enumerate(intervals):
            if task.release <= a and b <= task.deadline:
                graph.add_edge(f"m{j}", f"i{k}", capacity=int(b - a), weight=0)
                graph.add_edge(f"o{j}", f"i{k}", capacity=int(b - a), weight=0)
    for k, (a, b) in enumerate(intervals):
        graph.add_edge(f"i{k}", "t", capacity=int(b - a), weight=0)

    flow = nx.max_flow_min_cost(graph, "s", "t")
    for j, task in enumerate(tasks):
        if flow["s"][f"m{j}"] < int(task.mandatory):
            return None
    return sum(task.weight * (task.optional - flow["s"][f"o{j}"]) for j, task in enumerate(tasks))


def _random_instance(rng, n):
    tasks = []
    for _ in range(n):
        release = int(rng.integers(0, 3))
        deadline = release + int(rng.integers(1, 5))
        tasks.append(
            ImpreciseTask(
                release=release,
                deadline=deadline,
                mandatory=int(rng.integers(0, 3)),
                optional=int(rng.integers(0, 4)),
                weight=int(rng.integers(1, 4)),
            )
        )
    return tasks


@pytest.mark.parametrize("seed", range(12))
def test_matches_exhaustive_search(seed):
    """The optimum, the feasibility verdict and the schedule, against brute force."""
    rng = np.random.default_rng(seed)
    for n in (2, 3):
        tasks = _random_instance(rng, n)
        result = _solve(tasks)
        reference = _brute_force(tasks)

        assert result.feasible == (reference is not None), tasks
        if reference is None:
            continue
        assert np.isclose(result.total_error, float(reference), atol=1e-7), tasks
        executed = verify_schedule(tasks, result.schedule)
        assert np.allclose(executed, result.executed, atol=1e-7)


@pytest.mark.parametrize("seed", range(12))
def test_matches_a_min_cost_flow(seed):
    """
    Against the formulation the literature actually states, solved by network
    simplex. Agreeing with it says the LP encodes the same problem, not merely
    that the LP is solved correctly.
    """
    rng = np.random.default_rng(100 + seed)
    tasks = _random_instance(rng, 3)
    result = _solve(tasks)
    reference = _min_cost_flow(tasks)
    assert result.feasible == (reference is not None), tasks
    if reference is not None:
        assert np.isclose(result.total_error, float(reference), atol=1e-7), tasks


def test_single_task_closed_form():
    """One task: the error is whatever does not fit in its own window."""
    for release, deadline, mandatory, optional in ((0, 5, 1, 2), (0, 2, 1, 4), (3, 4, 0.5, 3)):
        task = ImpreciseTask(release, deadline, mandatory, optional, weight=2.0)
        result = _solve([task])
        assert result.feasible
        expected = 2.0 * max(task.total - (deadline - release), 0.0)
        assert np.isclose(result.total_error, expected, atol=1e-9)


def test_common_deadline_without_mandatory_is_a_weight_ordered_fill():
    """
    All tasks released at 0 with one common deadline and no mandatory part: the
    optimum is simply to spend the available time on the heaviest tasks first,
    which is a closed form the LP is not told.
    """
    horizon = 7.0
    specs = [(3.0, 1.0), (4.0, 5.0), (2.0, 2.0), (6.0, 3.0)]  # (optional, weight)
    tasks = [ImpreciseTask(0.0, horizon, 0.0, opt, w) for opt, w in specs]

    capacity = horizon
    executed = {}
    for index in sorted(range(len(specs)), key=lambda i: -specs[i][1]):
        take = min(specs[index][0], capacity)
        executed[index] = take
        capacity -= take
    expected = sum(w * (opt - executed[i]) for i, (opt, w) in enumerate(specs))

    result = _solve(tasks)
    assert np.isclose(result.total_error, expected, atol=1e-7)


def test_no_optional_work_means_no_error_when_feasible():
    tasks = [ImpreciseTask(0, 4, mandatory=2), ImpreciseTask(1, 5, mandatory=1)]
    result = _solve(tasks)
    assert result.feasible
    assert np.isclose(result.total_error, 0.0, atol=1e-9)
    verify_schedule(tasks, result.schedule)


def test_overloaded_mandatory_set_is_reported_infeasible():
    """Three units of mandatory work in a window of two cannot be scheduled."""
    tasks = [ImpreciseTask(0, 2, mandatory=1.5), ImpreciseTask(0, 2, mandatory=1.5)]
    scheduler = ImpreciseComputationScheduler().set_tasks(tasks)
    assert not scheduler.is_feasible()
    assert not scheduler.solve().feasible
    assert not scheduler.solve("max").feasible
    assert not scheduler.solve("lexicographic").feasible


def test_max_error_objective_balances_and_lexicographic_keeps_the_total():
    """
    Two identical tasks that must shed two units between them: minimising the
    TOTAL allows all of it to land on one task, minimising the MAXIMUM splits it,
    and the lexicographic objective must achieve the best total AND that split.
    """
    tasks = [ImpreciseTask(0, 4, optional=3), ImpreciseTask(0, 4, optional=3)]
    by_total = _solve(tasks, "total")
    by_max = _solve(tasks, "max")
    lexicographic = _solve(tasks, "lexicographic")

    assert np.isclose(by_total.total_error, 2.0, atol=1e-7)
    assert np.isclose(by_max.max_error, 1.0, atol=1e-7)
    assert np.isclose(lexicographic.total_error, by_total.total_error, atol=1e-7)
    assert lexicographic.max_error <= by_total.max_error + 1e-7
    assert np.isclose(lexicographic.max_error, 1.0, atol=1e-7)


def test_max_objective_never_beats_total_on_the_total_and_vice_versa():
    """Each objective must be the best at its own criterion."""
    rng = np.random.default_rng(5)
    for _ in range(8):
        tasks = _random_instance(rng, 3)
        by_total = _solve(tasks, "total")
        by_max = _solve(tasks, "max")
        if not by_total.feasible:
            continue
        assert by_total.total_error <= by_max.total_error + 1e-7
        assert by_max.max_error <= by_total.max_error + 1e-7


def test_a_heavier_task_is_protected():
    """Raising one task's weight must not increase its own error."""
    previous = None
    for weight in (1.0, 2.0, 5.0, 20.0):
        tasks = [
            ImpreciseTask(0, 5, optional=4, weight=weight),
            ImpreciseTask(0, 5, optional=4, weight=3.0),
        ]
        error = _solve(tasks).errors[0]
        if previous is not None:
            assert error <= previous + 1e-7
        previous = error


def test_more_time_never_hurts():
    """Extending every deadline cannot increase the optimal error."""
    previous = None
    for horizon in (3.0, 4.0, 6.0, 10.0):
        tasks = [
            ImpreciseTask(0, horizon, mandatory=1, optional=3, weight=1),
            ImpreciseTask(0, horizon, mandatory=1, optional=3, weight=2),
        ]
        result = _solve(tasks)
        assert result.feasible
        if previous is not None:
            assert result.total_error <= previous + 1e-7
        previous = result.total_error
    assert np.isclose(previous, 0.0, atol=1e-7)  # eventually everything fits


def test_schedule_verification_rejects_a_broken_schedule():
    """verify_schedule must actually catch violations, or it proves nothing."""
    tasks = [ImpreciseTask(0, 4, mandatory=1, optional=1), ImpreciseTask(2, 6, mandatory=1)]
    with pytest.raises(ValueError, match="overlaps"):
        verify_schedule(tasks, [(0.0, 2.0, 0), (1.0, 3.0, 1)])
    with pytest.raises(ValueError, match="outside"):
        verify_schedule(tasks, [(0.0, 1.0, 1)])  # before its release
    with pytest.raises(ValueError, match="less than its mandatory"):
        verify_schedule(tasks, [(0.0, 2.0, 0)])  # task 1 never runs
    with pytest.raises(ValueError, match="more than its total"):
        verify_schedule(tasks, [(0.0, 3.0, 0), (3.0, 4.0, 1)])


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="must exceed release"):
        ImpreciseTask(release=2.0, deadline=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        ImpreciseTask(0.0, 1.0, mandatory=-1.0)
    with pytest.raises(ValueError, match="weight must be positive"):
        ImpreciseTask(0.0, 1.0, weight=0.0)
    with pytest.raises(ValueError, match="at least one task"):
        ImpreciseComputationScheduler().set_tasks([])
    with pytest.raises(ValueError, match="objective must be"):
        _solve([ImpreciseTask(0, 1)], "nonsense")
