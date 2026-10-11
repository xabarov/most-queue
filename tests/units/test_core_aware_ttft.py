"""Independent overshoot, return-system and old-bound containment arbiters."""

import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import poisson

from most_queue.theory.continuous_batching import CoreAwareTtftCalc, PhaseAwareTtftCalc, StageAwareTtftCalc

LAW = ((1, 1, 0.6), (3, 2, 0.4))
COSTS = (0.1, 0.02, 0.003)


def configured(cls=CoreAwareTtftCalc, cap=2, budget=4, rate=1.3, law=LAW, **kwargs):
    """Same physical synthetic queue; no fitted costs."""
    calc = cls(cap, budget, **kwargs)
    calc.set_servers(law, *COSTS)
    calc.set_sources(rate)
    return calc


@pytest.mark.parametrize("budget", [2, 4, None])
@pytest.mark.parametrize("law", [LAW, ((2, 2, 1.0),)])
def test_ablations_nested_and_finer_answers_contained(budget, law):
    global_result = configured(PhaseAwareTtftCalc, budget=budget, law=law).run()
    phase = configured(budget=budget, law=law, mode="phase").run()
    core = configured(budget=budget, law=law).run()
    fine = configured(budget=budget, law=law, truncation=32).run()
    for key in ("w", "v", "ttft"):
        glo, ghi = global_result.mean_bounds[key]
        plo, phi = phase.mean_bounds[key]
        clo, chi = core.mean_bounds[key]
        flo, fhi = fine.mean_bounds[key]
        assert glo <= plo + 1e-12 <= clo + 2e-12 <= flo + 3e-12
        assert fhi <= chi + 1e-12 <= phi + 2e-12 <= ghi + 3e-12
    checks = core.continuation_diagnostics
    assert checks["restart_exit_probability_max"] < 1
    assert checks["absorption_residual"] < 1e-12
    assert np.all(np.array(checks["core_upper"]) <= np.array(checks["global_core_upper"]))
    assert core.phase_certificate["drift_verified"]
    assert not core.roundoff_certified


@pytest.mark.parametrize("cap,budget", [(2, 2), (2, 4), (3, 3)])
def test_exit_phase_moments_against_direct_poisson_word_sum(cap, budget):
    calc = configured(cap=cap, budget=budget, rate=0.4)
    calc.run()
    probs = calc._mark_probabilities()
    values = calc._constants
    eps = values["epsilon"]
    for phase in calc._phase_calc._phases[:12]:
        duration, survivors = calc._iteration(phase)
        fill = cap - len(survivors)
        mean, offset, threshold = calc.l * duration, 2, 4
        actual = calc._exit_remainder(survivors, calc._overshoot_moments(mean, threshold, offset), None, probs)
        expected = np.zeros(4)
        for count in range(threshold, 100):
            for word in itertools.product(range(len(probs)), repeat=fill):
                mass = poisson.pmf(count, mean) * np.prod([probs[k] for k in word])
                target = calc._canonical(survivors + tuple((k, 0) for k in word))
                y = count + offset + calc._phase_shifts[target]
                steps = y / eps
                area_steps = (y**2 + values["increment_second_max"] * y / eps) / (2 * eps)
                area = values["duration_max"] * area_steps
                population = area + values["population_step_constant"] * steps
                expected += mass * (
                    np.array(
                        [
                            values["duration_max"] * steps,
                            population,
                            area + values["waiting_step_constant"] * steps,
                            population,
                        ]
                    )
                    + calc._core_upper
                )
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_finite_core_closure_against_explicit_exterior_chain():
    # States: idle, core1, core2, exterior. Independent finite continuation
    # arbiter; exterior returns randomly to the core, not to idle or a buffer.
    full = np.array([[0, 0.5, 0.5, 0], [0.2, 0.3, 0.2, 0.3], [0.4, 0.1, 0.2, 0.3], [0, 0.4, 0.6, 0]])
    rewards = np.array([[0, 0, 0, 0], [1, 2, 3, 4], [2, 1, 4, 3], [1, 3, 2, 4]], dtype=float)
    matrix = csr_matrix(full[:3, :3])
    exits = np.column_stack((full[:3, 3], full[:3, 3, None] * rewards[3]))
    bound, diagnostics = CoreAwareTtftCalc._return_system(matrix, rewards[:3], exits, [0, 1])
    exact = np.linalg.solve(np.eye(3) - full[1:, 1:], rewards[1:])
    assert np.all(exact[:2].max(axis=0) <= bound)
    assert np.all(bound <= np.array(diagnostics["separate_max_geometric_upper"]))
    assert diagnostics["restart_idle_probability_min"] > 0
    # Dropping exterior reward yields a different, too optimistic calculation.
    wrong, _ = CoreAwareTtftCalc._return_system(matrix, rewards[:3], np.c_[exits[:, 0], np.zeros((3, 4))], [0, 1])
    assert np.any(wrong < exact[:2].max(axis=0))


def test_core_solve_preserves_full_return_even_after_many_exits():
    calc = configured(rate=4.0, core_truncation=1)
    result = calc.run()
    d = result.continuation_diagnostics
    assert d["restart_exit_probability_max"] > 0.5
    assert d["core_upper"][0] > 0
    assert d["core_upper"][2] > 0
    # Auxiliary excursions exclude the core term; production MUST include it.
    moments = np.array([0.2, 2.0, 25.0])
    survivors = calc._iteration(calc._phase_calc._phases[0])[1]
    full = calc._exit_remainder(survivors, moments, None, calc._mark_probabilities())
    calc._building_core = True
    exterior = calc._exit_remainder(survivors, moments, None, calc._mark_probabilities())
    assert full - exterior == pytest.approx(moments[0] * calc._core_upper)


@pytest.mark.parametrize("budget", [1, 2, None])
def test_b1_pk_no_auxiliary_solve(budget):
    result = configured(cap=1, budget=budget).run()
    pk = configured(StageAwareTtftCalc, cap=1, budget=budget).run()
    assert result.w == pytest.approx(pk.w)
    assert result.ttft_mean == pytest.approx(pk.ttft_mean)
    assert "core_states" not in result.continuation_diagnostics
    assert result.truncation is None


@pytest.mark.parametrize(
    "kwargs", [{"mode": "reflect"}, {"core_truncation": True}, {"core_truncation": 0}, {"truncation": 0}]
)
def test_invalid_domain_or_mode(kwargs):
    with pytest.raises(ValueError):
        configured(**kwargs).run()


@pytest.mark.parametrize("kwargs", [{"max_states": 200}, {"max_transition_words": 2000}])
def test_combined_budgets_and_restoration(kwargs):
    calc = configured(**kwargs)
    original = calc.truncation, calc.max_states, calc.max_transition_words
    with pytest.raises(ValueError, match="budget"):
        calc.run()
    assert (calc.truncation, calc.max_states, calc.max_transition_words) == original
    assert not calc._building_core
    calc.max_states, calc.max_transition_words = 20000, 2000000
    result = calc.run()
    d = result.continuation_diagnostics
    assert d["total_count_states"] == result.states + d["core_states"]
    assert d["total_count_transition_words"] == result.transition_words + d["core_transition_words"]
    assert d["total_kernel_storage_bytes"] == result.kernel_storage_bytes + d["core_kernel_storage_bytes"]


def test_reconfiguration_no_stale_core_or_shifts():
    calc = configured()
    first = calc.run()
    calc.set_sources(2)
    calc.set_servers(((2, 2, 1.0),), *COSTS)
    second = calc.run()
    fresh = configured(law=((2, 2, 1.0),), rate=2).run()
    assert second.mean_bounds == fresh.mean_bounds
    assert second.continuation_diagnostics["core_upper"] != first.continuation_diagnostics["core_upper"]
    calc.set_sources(100)
    with pytest.raises(ValueError, match="positive_drift_escape"):
        calc.run()
    calc.set_sources(2)
    assert calc.run().mean_bounds == second.mean_bounds


def test_bad_solve_mass_and_contraction_are_refused():
    rewards = np.ones((2, 4))
    with pytest.raises(ValueError, match="singular"):
        CoreAwareTtftCalc._return_system(csr_matrix([[0, 1], [0, 1]]), rewards, np.zeros((2, 5)), [0])
    with pytest.raises(ValueError, match="balance"):
        CoreAwareTtftCalc._return_system(csr_matrix([[0, 1], [0.2, 0.2]]), rewards, np.zeros((2, 5)), [0])
    with pytest.raises(ValueError, match="contraction"):
        CoreAwareTtftCalc._return_system(
            csr_matrix([[0, 1], [0, 0.5]]), rewards, np.array([[0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0]]), [0]
        )
    with patch("most_queue.theory.continuous_batching.core_aware_ttft.spsolve", return_value=np.full((1, 6), np.nan)):
        with pytest.raises(ValueError, match="ill-conditioned"):
            CoreAwareTtftCalc._return_system(csr_matrix([[0, 1], [1, 0]]), rewards, np.zeros((2, 5)), [0])


def test_legacy_selected_answers_unchanged_by_exit_hook():
    historical = json.loads(Path("docs/research/phase-aware-stability-results-2026.json").read_text())
    for row in historical["rows"]:
        calc = PhaseAwareTtftCalc(row["batch_cap"], row["token_budget"], row["result"]["truncation"] or 8)
        calc.set_servers(row["law"], *row["costs"])
        calc.set_sources(row["arrival_rate"])
        actual = calc.run()
        for key, bounds in row["result"]["mean_bounds"].items():
            assert actual.mean_bounds[key] == pytest.approx(bounds, rel=1e-12, abs=1e-13)


def test_mutated_parameters_revalidated_before_derived_state():
    calc = configured()
    calc.core_truncation = 0
    with pytest.raises(ValueError, match="core_truncation"):
        calc.run()
    calc.core_truncation = 8
    calc.mode = "reflect"
    with pytest.raises(ValueError, match="mode"):
        calc.run()


def test_dominating_load_above_one_is_still_supported():
    calc = configured()
    capacity = configured(PhaseAwareTtftCalc).run().phase_certificate["capacity"]
    calc.set_sources(0.85 * capacity)
    result = calc.run()
    assert result.dominating_load > 1
    assert result.phase_certificate["stable"]
