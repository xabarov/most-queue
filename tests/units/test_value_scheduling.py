"""
Unit tests for value-based scheduling under overload.

The clairvoyant optimum is checked against exhaustive search over subsets, with
feasibility of each subset decided NOT by Horn's condition but by running
preemptive EDF on it. That keeps the reference independent of the thing being
tested: the integer program's constraints *are* Horn's condition, so checking
them against Horn's condition would prove nothing, whereas EDF's optimality for
feasibility is a separate theorem.
"""

import itertools

import numpy as np
import pytest

from most_queue.sim.value_scheduling import (
    GUARANTEES,
    PRIORITIES,
    ValueJob,
    ValueSchedulingSim,
    compare_with_clairvoyant,
    to_offline_tasks,
)
from most_queue.theory.imprecise.offline import verify_schedule
from most_queue.theory.value_scheduling import ClairvoyantValueScheduler, ValueTask

_TOL = 1e-6


# --------------------------------------------------------------------------
# independent references
# --------------------------------------------------------------------------


def _edf_feasible(tasks: list[ValueTask]) -> bool:
    """
    Decide schedulability of a set that must be completed IN FULL, by running
    preemptive EDF and watching for a deadline miss.

    EDF is optimal for this question (Horn 1974, and Liu & Layland for the
    periodic case), so no miss under EDF means feasible and a miss means
    infeasible. No use is made of the interval formulation.
    """
    if not tasks:
        return True
    remaining = [task.processing for task in tasks]
    order = sorted(range(len(tasks)), key=lambda i: tasks[i].release)
    ready: list[int] = []
    clock = min(task.release for task in tasks)
    nxt = 0
    while nxt < len(order) or ready:
        while nxt < len(order) and tasks[order[nxt]].release <= clock + _TOL:
            ready.append(order[nxt])
            nxt += 1
        if not ready:
            clock = tasks[order[nxt]].release
            continue
        current = min(ready, key=lambda i: tasks[i].deadline)
        events = [clock + remaining[current]]
        if nxt < len(order):
            events.append(tasks[order[nxt]].release)
        step = min(events)
        worked = min(step - clock, remaining[current])
        remaining[current] -= worked
        clock = step
        if remaining[current] <= _TOL:
            if clock > tasks[current].deadline + _TOL:
                return False
            ready.remove(current)
        elif clock > tasks[current].deadline + _TOL:
            return False
    return True


def _brute_force_optimum(tasks: list[ValueTask]) -> float:
    """Best total value over all subsets, feasibility decided by EDF."""
    best = 0.0
    for size in range(len(tasks) + 1):
        for subset in itertools.combinations(range(len(tasks)), size):
            value = sum(tasks[i].value for i in subset)
            if value <= best:
                continue
            if _edf_feasible([tasks[i] for i in subset]):
                best = value
    return best


def _random_tasks(rng, count, span=20.0) -> list[ValueTask]:
    tasks = []
    for _ in range(count):
        release = rng.uniform(0.0, span)
        window = rng.uniform(1.0, span / 2)
        tasks.append(
            ValueTask(
                release=release,
                deadline=release + window,
                processing=rng.uniform(0.2, window),
                value=rng.uniform(1.0, 10.0),
            )
        )
    return tasks


# --------------------------------------------------------------------------
# ValueTask / ValueJob validation
# --------------------------------------------------------------------------


def test_value_task_rejects_inverted_window():
    with pytest.raises(ValueError):
        ValueTask(release=5.0, deadline=5.0, processing=1.0, value=1.0)
    with pytest.raises(ValueError):
        ValueTask(release=5.0, deadline=4.0, processing=1.0, value=1.0)


def test_value_task_rejects_negative_processing_and_value():
    with pytest.raises(ValueError):
        ValueTask(release=0.0, deadline=5.0, processing=-1.0, value=1.0)
    with pytest.raises(ValueError):
        ValueTask(release=0.0, deadline=5.0, processing=1.0, value=-1.0)


def test_value_task_window_and_schedulable_alone():
    task = ValueTask(release=2.0, deadline=7.0, processing=4.0, value=1.0)
    assert task.window == pytest.approx(5.0)
    assert task.schedulable_alone
    assert not ValueTask(release=2.0, deadline=7.0, processing=5.5, value=1.0).schedulable_alone


def test_value_job_rejects_actual_above_worst():
    with pytest.raises(ValueError):
        ValueJob(arrival=0.0, deadline=10.0, worst=2.0, actual=3.0, value=1.0)


def test_empty_task_set_is_rejected():
    with pytest.raises(ValueError):
        ClairvoyantValueScheduler().set_tasks([])
    with pytest.raises(ValueError):
        ClairvoyantValueScheduler().solve()


# --------------------------------------------------------------------------
# the clairvoyant optimum, against exhaustive search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_clairvoyant_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    tasks = _random_tasks(rng, count=8)
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    assert result.optimal_value == pytest.approx(_brute_force_optimum(tasks), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_clairvoyant_matches_brute_force_tight_windows(seed):
    """Narrow windows and heavy processing, so the selection really bites."""
    rng = np.random.default_rng(1000 + seed)
    tasks = []
    for _ in range(9):
        release = rng.uniform(0.0, 6.0)
        window = rng.uniform(1.0, 3.0)
        tasks.append(
            ValueTask(
                release=release,
                deadline=release + window,
                processing=rng.uniform(0.5 * window, window),
                value=rng.uniform(1.0, 10.0),
            )
        )
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    assert result.optimal_value == pytest.approx(_brute_force_optimum(tasks), rel=1e-6, abs=1e-6)


def test_single_task_completes_exactly_when_it_fits():
    fits = ValueTask(release=1.0, deadline=5.0, processing=4.0, value=7.0)
    assert ClairvoyantValueScheduler().set_tasks([fits]).solve().optimal_value == pytest.approx(7.0)
    too_big = ValueTask(release=1.0, deadline=5.0, processing=4.5, value=7.0)
    result = ClairvoyantValueScheduler().set_tasks([too_big]).solve()
    assert result.optimal_value == pytest.approx(0.0)
    assert result.rejected == 1


def test_two_tasks_in_one_window_keeps_the_more_valuable():
    tasks = [
        ValueTask(release=0.0, deadline=10.0, processing=6.0, value=3.0),
        ValueTask(release=0.0, deadline=10.0, processing=6.0, value=5.0),
    ]
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    assert result.optimal_value == pytest.approx(5.0)
    assert result.selected == [False, True]


def test_knapsack_corner_prefers_two_small_over_one_large():
    """Value, not size, decides -- and the optimum is not greedy by value."""
    tasks = [
        ValueTask(release=0.0, deadline=10.0, processing=10.0, value=6.0),
        ValueTask(release=0.0, deadline=10.0, processing=5.0, value=3.5),
        ValueTask(release=0.0, deadline=10.0, processing=5.0, value=3.5),
    ]
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    assert result.optimal_value == pytest.approx(7.0)
    assert result.selected == [False, True, True]


def test_everything_fits_under_light_load():
    tasks = [ValueTask(release=10.0 * i, deadline=10.0 * i + 9.0, processing=1.0, value=2.0) for i in range(6)]
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    assert result.optimal_value == pytest.approx(12.0)
    assert result.hit_value_ratio == pytest.approx(1.0)
    assert result.rejected == 0
    assert ClairvoyantValueScheduler().set_tasks(tasks).is_feasible()


def test_is_feasible_is_false_when_something_must_be_dropped():
    tasks = [
        ValueTask(release=0.0, deadline=5.0, processing=4.0, value=1.0),
        ValueTask(release=0.0, deadline=5.0, processing=4.0, value=1.0),
    ]
    assert not ClairvoyantValueScheduler().set_tasks(tasks).is_feasible()


# --------------------------------------------------------------------------
# the schedule the optimum emits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_emitted_schedule_runs_every_selected_task_in_its_window(seed):
    rng = np.random.default_rng(2000 + seed)
    tasks = _random_tasks(rng, count=10)
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()

    served = {}
    for begin, end, index in result.schedule:
        assert begin >= tasks[index].release - _TOL
        assert end <= tasks[index].deadline + _TOL
        served[index] = served.get(index, 0.0) + (end - begin)
    # the machine runs one task at a time
    spans = sorted((begin, end) for begin, end, _ in result.schedule)
    for (_, end), (begin, _) in zip(spans, spans[1:]):
        assert begin >= end - _TOL
    # exactly the selected tasks are run, and in full
    for index, keep in enumerate(result.selected):
        assert served.get(index, 0.0) == pytest.approx(tasks[index].processing if keep else 0.0, abs=1e-6)


def test_emitted_schedule_passes_the_independent_verifier():
    rng = np.random.default_rng(7)
    tasks = _random_tasks(rng, count=9)
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    picked = [i for i, keep in enumerate(result.selected) if keep]
    remap = {original: local for local, original in enumerate(picked)}
    from most_queue.theory.imprecise.offline import ImpreciseTask

    as_imprecise = [
        ImpreciseTask(
            release=tasks[i].release,
            deadline=tasks[i].deadline,
            mandatory=tasks[i].processing,
            optional=0.0,
        )
        for i in picked
    ]
    local_schedule = [(begin, end, remap[index]) for begin, end, index in result.schedule]
    assert verify_schedule(as_imprecise, local_schedule)


def test_schedule_can_be_suppressed():
    tasks = _random_tasks(np.random.default_rng(3), count=7)
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve(emit_schedule=False)
    assert result.schedule == []
    assert result.optimal_value > 0.0


# --------------------------------------------------------------------------
# block decomposition and row pruning -- speedups that must not change answers
# --------------------------------------------------------------------------


def test_disjoint_blocks_are_detected_and_solved_independently():
    tasks = [
        ValueTask(release=0.0, deadline=5.0, processing=4.0, value=1.0),
        ValueTask(release=0.0, deadline=5.0, processing=4.0, value=3.0),
        ValueTask(release=100.0, deadline=105.0, processing=4.0, value=2.0),
        ValueTask(release=100.0, deadline=105.0, processing=4.0, value=7.0),
    ]
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve()
    assert result.components == 2
    assert result.optimal_value == pytest.approx(3.0 + 7.0)


def test_block_count_is_one_when_windows_chain_together():
    tasks = [ValueTask(release=float(i), deadline=float(i) + 3.0, processing=0.1, value=1.0) for i in range(6)]
    assert ClairvoyantValueScheduler().set_tasks(tasks).solve().components == 1


@pytest.mark.parametrize("seed", range(8))
def test_pruning_and_decomposition_agree_with_the_full_row_set(seed):
    """
    Same integer program, every Horn row kept and no blocks -- the reductions
    in ``_horn_rows``/``_components`` are claimed to be exact, so this must
    land on the same optimum.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    rng = np.random.default_rng(3000 + seed)
    tasks = _random_tasks(rng, count=11)
    processing = np.array([task.processing for task in tasks])
    releases = np.array([task.release for task in tasks])
    deadlines = np.array([task.deadline for task in tasks])
    values = np.array([task.value for task in tasks])

    rows, rhs = [], []
    for lo in np.unique(releases):
        for hi in np.unique(deadlines):
            if hi <= lo:
                continue
            mask = (releases >= lo - 1e-12) & (deadlines <= hi + 1e-12)
            if mask.any():
                rows.append(np.where(mask, processing, 0.0))
                rhs.append(hi - lo)
    reference = milp(
        c=-values,
        constraints=LinearConstraint(np.array(rows), -np.inf, np.array(rhs)),
        integrality=np.ones(len(tasks)),
        bounds=Bounds(0, 1),
    )
    assert reference.success
    result = ClairvoyantValueScheduler().set_tasks(tasks).solve(emit_schedule=False)
    assert result.optimal_value == pytest.approx(-reference.fun, rel=1e-6, abs=1e-6)


def test_oversized_instance_is_refused_with_advice():
    tasks = _random_tasks(np.random.default_rng(5), count=30)
    scheduler = ClairvoyantValueScheduler(max_constraints=10).set_tasks(tasks)
    with pytest.raises(ValueError, match="max_constraints"):
        scheduler.solve()


# --------------------------------------------------------------------------
# the on-line simulator
# --------------------------------------------------------------------------


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        ValueSchedulingSim(priority="srpt")
    with pytest.raises(ValueError):
        ValueSchedulingSim(guarantee="optimistic")
    with pytest.raises(ValueError):
        ValueSchedulingSim(alpha=1.5)


def test_algorithm_names_follow_the_paper():
    assert ValueSchedulingSim(priority="edf", guarantee="plain").algorithm == "EDF"
    assert ValueSchedulingSim(priority="edf", guarantee="guaranteed").algorithm == "GEDF"
    assert ValueSchedulingSim(priority="hdf", guarantee="robust").algorithm == "RHDF"
    assert ValueSchedulingSim(priority="mix", guarantee="guaranteed").algorithm == "GMIX"


def test_generator_rejects_bad_arguments():
    sim = ValueSchedulingSim(seed=0)
    with pytest.raises(ValueError):
        sim.generate_task_set(0.0)
    with pytest.raises(ValueError):
        sim.generate_task_set(1.0, num_streams=0)
    with pytest.raises(ValueError):
        sim.generate_task_set(1.0, value_mode="quadratic")
    with pytest.raises(ValueError):
        sim.generate_task_set(1.0, unused_ratio=1.0)


def test_generated_task_set_follows_the_papers_ranges():
    sim = ValueSchedulingSim(seed=11)
    jobs = sim.generate_task_set(2.0, num_streams=30, horizon=20000.0)
    assert len(jobs) > 50
    for job in jobs:
        assert 50.0 <= job.worst <= 350.0
        assert 0.0 <= job.actual <= job.worst
        relative = job.deadline - job.arrival
        assert job.worst + 150.0 <= relative <= job.worst + 1850.0
        assert 150.0 <= job.value <= 1850.0
    # the mean actual time is half the worst case, as the paper intends
    assert np.mean([j.actual / j.worst for j in jobs]) == pytest.approx(0.5, abs=0.05)


def test_linear_value_mode_ties_value_to_the_relative_deadline():
    sim = ValueSchedulingSim(seed=4)
    jobs = sim.generate_task_set(2.0, num_streams=20, horizon=10000.0, value_mode="linear")
    for job in jobs:
        assert job.value == pytest.approx(job.deadline - job.arrival)


def test_unused_ratio_fixes_the_actual_time():
    sim = ValueSchedulingSim(seed=4)
    jobs = sim.generate_task_set(2.0, num_streams=20, horizon=10000.0, unused_ratio=0.25)
    for job in jobs:
        assert job.actual == pytest.approx(0.75 * job.worst)


def test_nominal_load_is_what_the_generator_claims():
    """Load measured on worst-case times must come out at the requested value."""
    sim = ValueSchedulingSim(seed=2)
    horizon = 400_000.0
    jobs = sim.generate_task_set(2.5, num_streams=40, horizon=horizon)
    nominal = sum(job.worst for job in jobs) / horizon
    assert nominal == pytest.approx(2.5, rel=0.1)
    actual = sum(job.actual for job in jobs) / horizon
    assert actual == pytest.approx(1.25, rel=0.1)


@pytest.mark.parametrize("priority", PRIORITIES)
@pytest.mark.parametrize("guarantee", GUARANTEES)
def test_run_does_not_mutate_the_task_set(priority, guarantee):
    """
    Every algorithm must see the same realisation, so a run may not leave a
    mark on the jobs it was given. (An early version carried executed work
    across runs, which inflated every result after the first.)
    """
    sim = ValueSchedulingSim(priority=priority, guarantee=guarantee, seed=1)
    jobs = sim.generate_task_set(3.0, num_streams=10, horizon=5000.0)
    before = [(j.arrival, j.deadline, j.worst, j.actual, j.value) for j in jobs]
    first = sim.run_on(jobs)
    after = [(j.arrival, j.deadline, j.worst, j.actual, j.value) for j in jobs]
    assert before == after
    second = sim.run_on(jobs)
    assert second.hit_value_ratio == pytest.approx(first.hit_value_ratio)
    assert second.cumulative_value == pytest.approx(first.cumulative_value)


@pytest.mark.parametrize("priority", PRIORITIES)
@pytest.mark.parametrize("guarantee", GUARANTEES)
def test_result_bookkeeping_is_consistent(priority, guarantee):
    sim = ValueSchedulingSim(priority=priority, guarantee=guarantee, seed=3)
    jobs = sim.generate_task_set(3.0, num_streams=12, horizon=6000.0)
    out = sim.run_on(jobs)
    assert out.completed + out.missed + out.rejected == out.jobs == len(jobs)
    assert 0.0 <= out.hit_value_ratio <= 1.0
    assert out.cumulative_value == pytest.approx(out.hit_value_ratio * out.total_value)
    assert out.busy_time >= out.wasted_time - _TOL
    assert out.wasted_time >= -_TOL
    assert out.busy_time <= sum(job.actual for job in jobs) + _TOL


@pytest.mark.parametrize("priority", PRIORITIES)
def test_plain_class_never_rejects(priority):
    sim = ValueSchedulingSim(priority=priority, guarantee="plain", seed=5)
    out = sim.run_on(sim.generate_task_set(3.0, num_streams=12, horizon=6000.0))
    assert out.rejected == 0
    assert out.reclaimed == 0


@pytest.mark.parametrize("priority", PRIORITIES)
@pytest.mark.parametrize("guarantee", ["guaranteed", "robust"])
def test_acceptance_test_admits_nothing_it_cannot_finish(priority, guarantee):
    """
    The guarantee classes test every arrival against worst-case remaining
    times, and the actual times are never larger, so an admitted job must make
    its deadline: these classes trade missed deadlines for rejections.
    """
    sim = ValueSchedulingSim(priority=priority, guarantee=guarantee, seed=6)
    out = sim.run_on(sim.generate_task_set(3.0, num_streams=12, horizon=6000.0))
    assert out.missed == 0
    assert out.rejected > 0


@pytest.mark.parametrize("priority", PRIORITIES)
def test_underload_banks_everything(priority):
    """With ample slack every rule collects the whole value."""
    sim = ValueSchedulingSim(priority=priority, guarantee="plain", seed=7)
    out = sim.run_on(sim.generate_task_set(0.4, num_streams=20, horizon=20000.0))
    assert out.hit_value_ratio == pytest.approx(1.0)
    assert out.missed == 0


def test_reclaiming_happens_and_only_in_the_robust_class():
    jobs = ValueSchedulingSim(seed=8).generate_task_set(3.0, num_streams=15, horizon=8000.0)
    robust = ValueSchedulingSim(priority="edf", guarantee="robust").run_on(jobs)
    guaranteed = ValueSchedulingSim(priority="edf", guarantee="guaranteed").run_on(jobs)
    assert robust.reclaimed > 0
    assert guaranteed.reclaimed == 0
    # reclaiming can only help: same test, strictly more work admitted
    assert robust.rejected < guaranteed.rejected
    assert robust.cumulative_value > guaranteed.cumulative_value


def test_priority_rules_actually_differ_on_a_hand_built_instance():
    """
    Two jobs, both released at zero, the machine able to finish only one.
    The urgent job is cheap and the valuable job is slack, so EDF and HVF must
    make opposite choices.
    """
    loose = [
        ValueJob(arrival=0.0, deadline=10.0, worst=9.0, actual=9.0, value=1.0),
        ValueJob(arrival=0.0, deadline=20.0, worst=9.0, actual=9.0, value=100.0),
    ]
    # Both fit, but only in deadline order -- EDF's optimality in underload.
    assert ValueSchedulingSim(priority="edf").run_on(loose).cumulative_value == pytest.approx(101.0)
    assert ValueSchedulingSim(priority="hvf").run_on(loose).cumulative_value == pytest.approx(100.0)

    tight = [
        ValueJob(arrival=0.0, deadline=10.0, worst=9.0, actual=9.0, value=1.0),
        ValueJob(arrival=0.0, deadline=11.0, worst=9.0, actual=9.0, value=100.0),
    ]
    # Now only one can finish, and the two rules pick opposite jobs.
    assert ValueSchedulingSim(priority="edf").run_on(tight).cumulative_value == pytest.approx(1.0)
    assert ValueSchedulingSim(priority="hvf").run_on(tight).cumulative_value == pytest.approx(100.0)


def test_hdf_prefers_value_density_over_raw_value():
    """
    One fat job worth 100 against five lean ones worth 30 each, all due at 10
    and together demanding 15. Raw value runs the fat job and banks 100;
    density runs the lean ones first and banks 150.
    """
    jobs = [ValueJob(arrival=0.0, deadline=10.0, worst=10.0, actual=10.0, value=100.0)]
    jobs += [ValueJob(arrival=0.0, deadline=10.0, worst=1.0, actual=1.0, value=30.0) for _ in range(5)]
    assert ValueSchedulingSim(priority="hvf").run_on(jobs).cumulative_value == pytest.approx(100.0)
    assert ValueSchedulingSim(priority="hdf").run_on(jobs).cumulative_value == pytest.approx(150.0)


def test_mix_alpha_interpolates_between_value_and_deadline():
    jobs = [
        ValueJob(arrival=0.0, deadline=10.0, worst=9.0, actual=9.0, value=1.0),
        ValueJob(arrival=0.0, deadline=11.0, worst=9.0, actual=9.0, value=100.0),
    ]
    by_value = ValueSchedulingSim(priority="mix", guarantee="plain", alpha=1.0).run_on(jobs)
    by_deadline = ValueSchedulingSim(priority="mix", guarantee="plain", alpha=0.0).run_on(jobs)
    assert by_value.cumulative_value == pytest.approx(100.0)
    assert by_deadline.cumulative_value == pytest.approx(1.0)


def test_a_job_needing_no_time_does_not_idle_the_machine():
    """
    A zero-work job banks its value for free, and must not hold the machine
    until the next event while real work is waiting. Here the free job is the
    most urgent, so a scheduler that let it block would lose the second job.
    """
    jobs = [
        ValueJob(arrival=0.0, deadline=1.0, worst=0.0, actual=0.0, value=5.0),
        ValueJob(arrival=0.0, deadline=10.0, worst=10.0, actual=10.0, value=7.0),
    ]
    out = ValueSchedulingSim(priority="edf", guarantee="plain").run_on(jobs)
    assert out.cumulative_value == pytest.approx(12.0)
    assert out.completed == 2
    assert out.busy_time == pytest.approx(10.0)


def test_empty_task_set_is_rejected_by_the_simulator():
    with pytest.raises(ValueError):
        ValueSchedulingSim().run_on([])


# --------------------------------------------------------------------------
# the two halves against each other
# --------------------------------------------------------------------------


def test_offline_tasks_carry_the_actual_times():
    jobs = ValueSchedulingSim(seed=9).generate_task_set(2.0, num_streams=5, horizon=3000.0)
    tasks = to_offline_tasks(jobs)
    assert len(tasks) == len(jobs)
    for job, task in zip(jobs, tasks):
        assert task.processing == pytest.approx(job.actual)
        assert task.release == pytest.approx(job.arrival)
        assert task.deadline == pytest.approx(job.deadline)
        assert task.value == pytest.approx(job.value)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("priority", PRIORITIES)
@pytest.mark.parametrize("guarantee", GUARANTEES)
def test_no_algorithm_can_beat_the_clairvoyant_optimum(seed, priority, guarantee):
    """
    The structural check that matters. The optimum is an upper bound over every
    scheduler, clairvoyant or not, so a ratio above one would mean the two
    halves are not solving the same problem.
    """
    sim = ValueSchedulingSim(priority=priority, guarantee=guarantee, seed=seed)
    jobs = sim.generate_task_set(3.0, num_streams=8, horizon=4000.0)
    report = compare_with_clairvoyant(sim, jobs, emit_schedule=False)
    assert report["competitive_ratio"] <= 1.0 + 1e-9
    assert report["hit_value_ratio"] <= report["clairvoyant_hit_value_ratio"] + 1e-9
    assert report["algorithm"] == sim.algorithm


def test_clairvoyant_banks_everything_when_edf_does():
    """
    If plain EDF loses nothing then the instance was never overloaded, so the
    optimum must also be the whole task set -- and conversely the optimum
    cannot be full while EDF, which is optimal in underload, misses something.
    """
    sim = ValueSchedulingSim(priority="edf", guarantee="plain", seed=10)
    jobs = sim.generate_task_set(1.0, num_streams=8, horizon=4000.0)
    report = compare_with_clairvoyant(sim, jobs, emit_schedule=False)
    assert report["hit_value_ratio"] == pytest.approx(1.0)
    assert report["clairvoyant_hit_value_ratio"] == pytest.approx(1.0)
    assert report["competitive_ratio"] == pytest.approx(1.0)
