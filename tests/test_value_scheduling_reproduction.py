"""
Reproduction of the experiments of Buttazzo, Spuri & Sensini (RTSS 1995), and
the comparison against the exact clairvoyant optimum that the paper lacks.

Each test here asserts one of the paper's published findings, so a regression
in the simulator shows up as a contradiction of the literature rather than as a
changed number. The task-set generator follows the paper: 100 streams,
worst-case times uniform on [50, 350], laxity uniform on [150, 1850], actual
execution time uniform on [0, worst], and a nominal load swept from 0.5 to 3.5.
The horizon is shorter than the paper's 300 000 time units and the averages are
over fewer runs, which is why the assertions are stated as orderings and gaps
rather than as exact values.
"""

import numpy as np
import pytest

from most_queue.sim.value_scheduling import (
    GUARANTEES,
    PRIORITIES,
    ValueSchedulingSim,
    to_offline_tasks,
)
from most_queue.theory.value_scheduling import ClairvoyantValueScheduler

NUM_STREAMS = 100
HORIZON = 30_000.0
SEEDS = 6


def _hvr(priority, guarantee, nominal_load, seeds=SEEDS, **kwargs):
    """Mean Hit Value Ratio over several independent task sets."""
    values = []
    for seed in range(seeds):
        sim = ValueSchedulingSim(priority=priority, guarantee=guarantee, seed=seed)
        jobs = sim.generate_task_set(nominal_load, num_streams=NUM_STREAMS, horizon=HORIZON, **kwargs)
        values.append(sim.run_on(jobs).hit_value_ratio)
    return float(np.mean(values))


def _row(guarantee, nominal_load, **kwargs):
    return {rule: _hvr(rule, guarantee, nominal_load, **kwargs) for rule in PRIORITIES}


# --------------------------------------------------------------------------
# Observation 1 -- plain class: value density wins, EDF collapses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nominal_load", [2.5, 3.0, 3.5])
def test_observation1_hdf_is_best_without_a_guarantee_mechanism(nominal_load):
    """
    "Without any guarantee mechanism, the most effective priority assignment in
    overload conditions is the one based on value density, namely HDF."
    """
    row = _row("plain", nominal_load)
    assert max(row, key=row.get) == "hdf", row


@pytest.mark.parametrize("value_mode", ["random", "linear"])
def test_observation1_edf_collapses_while_the_others_degrade_gracefully(value_mode):
    """
    EDF is optimal in underload and falls apart once the actual load passes
    one -- the domino effect. The value-based rules lose far less.
    """
    light = _row("plain", 1.0, value_mode=value_mode)
    heavy = _row("plain", 3.5, value_mode=value_mode)
    assert light["edf"] == pytest.approx(1.0, abs=0.01)
    edf_drop = light["edf"] - heavy["edf"]
    for rule in ("hvf", "hdf", "mix"):
        assert light[rule] - heavy[rule] < edf_drop, (rule, light, heavy)
    assert edf_drop > 0.3


def test_observation1_mix_beats_both_of_its_ingredients():
    """
    "Although MIX is defined as a linear combination of EDF and HVF, in
    overload conditions its behaviour is not an average of their performance.
    On the contrary, MIX performs better than both for any load."
    """
    for nominal_load in (2.0, 2.5, 3.0, 3.5):
        row = _row("plain", nominal_load)
        assert row["mix"] > row["edf"], (nominal_load, row)
        assert row["mix"] > row["hvf"], (nominal_load, row)


def test_observation1_edf_and_hdf_are_least_sensitive_to_the_task_set():
    """
    "The least sensitive algorithms with respect to task set variations are EDF
    and HDF, whereas HVF and MIX are more influenced."
    """
    random_set = _row("plain", 3.5, value_mode="random")
    linear_set = _row("plain", 3.5, value_mode="linear")
    shift = {rule: abs(random_set[rule] - linear_set[rule]) for rule in PRIORITIES}
    assert max(shift["edf"], shift["hdf"]) < min(shift["hvf"], shift["mix"]), shift


# --------------------------------------------------------------------------
# Observation 2 -- guaranteed class: EDF wins everywhere
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nominal_load", [1.0, 2.0, 2.5, 3.0, 3.5])
def test_observation2_gedf_is_best_at_every_load(nominal_load):
    """
    "If the system workload is controlled at each task activation by an
    acceptance test which under overload conditions rejects the newly arrived
    task, then the most effective priority assignment is EDF."
    """
    row = _row("guaranteed", nominal_load)
    assert max(row, key=row.get) == "edf", (nominal_load, row)


@pytest.mark.parametrize("nominal_load", [1.0, 2.0, 2.5, 3.0, 3.5])
def test_observation2_also_holds_against_the_value_based_rules_on_the_linear_set(nominal_load):
    """
    The substance of Observation 2 on the linear task set: against the two
    genuinely value-based orderings, GEDF wins at every load.

    The paper's figure 3b shows GEDF ahead of GMIX as well, and here it is not,
    at deep overload. That is a degeneracy of the MIX rule rather than a
    disagreement about EDF -- see
    :func:`test_mix_degenerates_to_fifo_on_the_linear_task_set`.
    """
    row = _row("guaranteed", nominal_load, value_mode="linear")
    assert row["edf"] > row["hvf"], (nominal_load, row)
    assert row["edf"] > row["hdf"], (nominal_load, row)


def test_mix_degenerates_to_fifo_on_the_linear_task_set():
    """
    A deviation from the paper's figure 3b, and the reason for it.

    MIX scores a job by ``alpha * value - (1 - alpha) * deadline``, which adds
    a value to a time and so is not invariant to how the two are scaled. On the
    linear task set the value IS the relative deadline, so at ``alpha = 0.5``
    the key becomes ``0.5 * (d - a) - 0.5 * d = -0.5 * a`` and the rule reduces
    exactly to first-come first-served -- it stops consulting value or urgency
    at all. "GMIX" on the linear set is therefore guaranteed FCFS, and at a
    nominal load of 3.5 that beats GEDF by about two points, a real effect
    rather than noise.
    """
    sim = ValueSchedulingSim(priority="mix", alpha=0.5, seed=0)
    jobs = sim.generate_task_set(3.0, num_streams=10, horizon=4000.0, value_mode="linear")
    remaining_worst = [job.worst for job in jobs]
    keys = [sim._priority_key(i, jobs, remaining_worst) for i in range(len(jobs))]
    assert keys == pytest.approx([0.5 * job.arrival for job in jobs])
    assert sorted(range(len(jobs)), key=lambda i: keys[i]) == sorted(range(len(jobs)), key=lambda i: jobs[i].arrival)
    # ... and on the random set, where value and deadline are independent, it
    # does not collapse: the order is neither arrival order nor deadline order.
    other = sim.generate_task_set(3.0, num_streams=10, horizon=4000.0, value_mode="random")
    other_worst = [job.worst for job in other]
    other_keys = [sim._priority_key(i, other, other_worst) for i in range(len(other))]
    by_key = sorted(range(len(other)), key=lambda i: other_keys[i])
    assert by_key != sorted(range(len(other)), key=lambda i: other[i].arrival)
    assert by_key != sorted(range(len(other)), key=lambda i: other[i].deadline)


# --------------------------------------------------------------------------
# Observation 3 -- robust class: close together, REDF ahead until deep overload
# --------------------------------------------------------------------------


def test_observation3_robust_algorithms_are_all_close_together():
    """
    "All robust algorithms show a graceful degradation as the load increases
    and achieve a similar behaviour in the range of overload conditions we have
    simulated."
    """
    for nominal_load in (2.0, 2.5, 3.0, 3.5):
        row = _row("robust", nominal_load)
        assert max(row.values()) - min(row.values()) < 0.07, (nominal_load, row)


def test_observation3_redf_leads_until_deep_overload_then_rhdf_takes_over():
    """
    "REDF shows the best performance for almost any load. However, for high
    overloads (nominal load greater than 3) RHDF performs slightly better than
    REDF."
    """
    for nominal_load in (1.5, 2.0, 2.5, 3.0):
        row = _row("robust", nominal_load)
        assert max(row, key=row.get) == "edf", (nominal_load, row)
    deep = _row("robust", 3.5)
    assert deep["hdf"] > deep["edf"], deep
    assert max(deep, key=deep.get) == "hdf", deep


# --------------------------------------------------------------------------
# Observation 4 -- the rows experiment: what each mechanism does to a rule
# --------------------------------------------------------------------------


def test_observation4_acceptance_test_hurts_every_value_based_ordering():
    """
    "The acceptance test executed at each arrival time by the guarantee
    mechanism worsens the performance of all priority assignments that consider
    importance values in their ordering discipline."

    Rejecting the newcomer regardless of its value is what does the damage, and
    EDF is the one rule that does not mind, since it never cared about value.
    """
    for rule in ("hvf", "hdf", "mix"):
        plain = _hvr(rule, "plain", 3.0)
        guaranteed = _hvr(rule, "guaranteed", 3.0)
        assert guaranteed < plain, (rule, plain, guaranteed)
    assert _hvr("edf", "guaranteed", 3.0) > _hvr("edf", "plain", 3.0)


def test_observation4_reclaiming_repairs_the_damage():
    """
    "The robust scheduling schemes perform very well both in overload and in
    underload conditions, proving that the reclaiming strategy is effective for
    increasing the Cumulative Value in all practical situations."

    For the value-based rules the robust class claws back essentially all of
    what the bare acceptance test cost; for EDF it goes well beyond plain.
    """
    for rule in PRIORITIES:
        plain = _hvr(rule, "plain", 3.0)
        guaranteed = _hvr(rule, "guaranteed", 3.0)
        robust = _hvr(rule, "robust", 3.0)
        assert robust > guaranteed, (rule, guaranteed, robust)
        assert robust > plain - 0.01, (rule, plain, robust)
    assert _hvr("edf", "robust", 3.0) > _hvr("edf", "plain", 3.0) + 0.1


def test_figure5_guarantee_helps_under_overload_and_hurts_under_underload():
    """
    The paper's last experiment sweeps the unused computation time ratio
    ``beta = 1 - actual / worst`` at a fixed nominal load of 3, so the ACTUAL
    load falls as beta rises. "For small unused computation time, GEDF and REDF
    are able to obtain a significant improvement compared to the plain EDF
    scheduling. Increasing the unused computation time, however, the actual
    load falls down and the plain EDF performs better and better, reaching the
    optimality in underload conditions. Notice that, as the system becomes
    underloaded (beta ~ 0.7), GEDF becomes [worse than plain EDF]."
    """
    tight = {g: _hvr("edf", g, 3.0, unused_ratio=0.125) for g in GUARANTEES}
    assert tight["guaranteed"] > tight["plain"] + 0.1, tight
    assert tight["robust"] > tight["guaranteed"], tight

    slack = {g: _hvr("edf", g, 3.0, unused_ratio=0.875) for g in GUARANTEES}
    assert slack["plain"] == pytest.approx(1.0, abs=0.005), slack
    assert slack["guaranteed"] < slack["plain"], slack  # the crossover past beta ~ 0.7


# --------------------------------------------------------------------------
# against the exact clairvoyant optimum -- not in the paper
# --------------------------------------------------------------------------

OPT_STREAMS = 20
OPT_HORIZON = 5_000.0
OPT_SEEDS = 5


def _clairvoyant_sweep(nominal_load):
    """Mean clairvoyant HVR and per-algorithm competitive ratios."""
    optima, ratios = [], {(r, g): [] for r in PRIORITIES for g in GUARANTEES}
    for seed in range(OPT_SEEDS):
        jobs = ValueSchedulingSim(seed=seed).generate_task_set(
            nominal_load, num_streams=OPT_STREAMS, horizon=OPT_HORIZON
        )
        best = ClairvoyantValueScheduler().set_tasks(to_offline_tasks(jobs)).solve(emit_schedule=False)
        optima.append(best.hit_value_ratio)
        for rule in PRIORITIES:
            for guarantee in GUARANTEES:
                banked = ValueSchedulingSim(priority=rule, guarantee=guarantee).run_on(jobs).cumulative_value
                ratios[(rule, guarantee)].append(banked / best.optimal_value)
    return float(np.mean(optima)), {key: float(np.mean(value)) for key, value in ratios.items()}


def test_nominal_load_two_is_where_overload_actually_begins():
    """
    The paper observes that a nominal load of two "corresponds to an actual
    load of one", since actual times average half the worst case. The exact
    optimum makes that precise: below it nothing need be lost at all.
    """
    assert _clairvoyant_sweep(2.0)[0] == pytest.approx(1.0, abs=0.005)
    assert _clairvoyant_sweep(3.5)[0] < 0.95


@pytest.mark.parametrize("nominal_load", [2.5, 3.0, 3.5])
def test_no_algorithm_beats_the_optimum_and_the_robust_ones_come_close(nominal_load):
    """
    The paper suggests "the performance of the robust algorithms is close to
    the best one achievable by an on-line algorithm" without measuring the best
    achievable. Against the exact clairvoyant optimum the robust class really
    does bank most of it, while plain EDF leaves a great deal on the table.
    """
    optimum, ratios = _clairvoyant_sweep(nominal_load)
    assert max(ratios.values()) <= 1.0 + 1e-9, ratios
    assert ratios[("edf", "robust")] > 0.85, ratios
    assert ratios[("hdf", "robust")] > 0.85, ratios
    assert ratios[("edf", "robust")] > ratios[("edf", "plain")], ratios
    assert optimum < 1.0 + 1e-9


def test_measured_competitive_ratios_sit_above_the_adversarial_bound():
    """
    No on-line algorithm can guarantee better than 1/4 of the clairvoyant
    optimum under overload (Baruah et al., Real-Time Systems 4(2), 1992). That
    bound is adversarial; on the paper's random task sets the measured ratios
    are far above it, which is the useful thing to know and is not something
    the bound can tell you.
    """
    _, ratios = _clairvoyant_sweep(3.5)
    assert min(ratios.values()) > 0.25, ratios
