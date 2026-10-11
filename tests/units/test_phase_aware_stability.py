"""Independent reachability, periodic classes, rational drift and mean checks."""

from fractions import Fraction

import numpy as np
import pytest
import yaml
from scipy.sparse import csr_matrix

from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import (
    InfiniteStageAwareTtftCalc,
    PhaseAwareStabilityCalc,
    PhaseAwareTtftCalc,
    StageAwareTtftCalc,
)

LAW = ((1, 1, 0.6), (3, 2, 0.4))
COSTS = (0.1, 0.02, 0.003)


def configured(cls=PhaseAwareStabilityCalc, cap=2, budget=2, rate=1.3, law=LAW, **kwargs):
    """Configure the original frozen iid policy without fitted parameters."""
    calc = cls(cap, budget, **kwargs)
    calc.set_servers(law, *COSTS)
    calc.set_sources(rate)
    return calc


@pytest.mark.parametrize("cap,budget", [(1, 1), (2, 2), (2, 4), (2, None), (3, 3)])
def test_reachability_against_independent_count_kernel(cap, budget):
    calc = configured(cap=cap, budget=budget, rate=0.1)
    result = calc.run()
    other = configured(InfiniteStageAwareTtftCalc, cap, budget, rate=0.1, truncation=0)
    states, *_ = other._killed_kernel(np.array([0.6, 0.4]), other.dominating_work())
    assert set(calc._phases) == {active for active, _ in states if len(active) == cap}
    assert result.reachable_active_count == len(states)
    assert np.asarray(calc._matrix.sum(axis=1)).ravel() == pytest.approx(np.ones(result.phase_count))
    for active, exact in zip(calc._phases, calc._exact_durations):
        assert float(exact) == pytest.approx(other._iteration(active)[0])


@pytest.mark.parametrize("budget", [2, 4])
def test_periodic_multiclass_and_critical_refusal(budget):
    calc = configured(budget=budget, law=((2, 2, 1.0),))
    result = calc.run()
    assert len(result.closed_classes) == 2
    assert result.transient_phase_count == (1 if budget == 2 else 0)
    if budget == 4:
        # Two independent 3-cycles process four prompt/output tokens per job.
        capacity = 2 / (3 * COSTS[0] + 8 * COSTS[1] + 18 * COSTS[2])
        assert result.capacity == pytest.approx(capacity)
        assert all(len(item["indices"]) == 3 for item in result.closed_classes)
    for load, status in (
        (0.95, "positive_recurrence"),
        (1, "critical_or_numerically_unresolved"),
        (1.05, "positive_drift_escape"),
    ):
        calc.set_sources(result.capacity * load)
        checked = calc.run()
        assert checked.classification == status
        assert checked.drift_verified == (load != 1)
        assert checked.stable == (load < 1)


@pytest.mark.parametrize("budget", [1, 2, None])
def test_b1_physical_time_threshold_and_pk(budget):
    calc = configured(cap=1, budget=budget)
    result = calc.run()
    times, _ = calc.isolated_times()
    assert result.capacity == pytest.approx(1 / (np.array([0.6, 0.4]) @ times))
    phase = configured(PhaseAwareTtftCalc, cap=1, budget=budget).run()
    pk = configured(StageAwareTtftCalc, cap=1, budget=budget).run()
    assert phase.ttft_mean == pytest.approx(pk.ttft_mean)
    assert phase.w == pytest.approx(pk.w)
    assert phase.truncation is None
    assert phase.return_bound_constants == {}


def test_exact_binary_drift_verification():
    calc = configured()
    result = calc.run()
    b = [Fraction(value) for value in result.corrector]
    drifts = [
        Fraction(calc.l) * duration - int(v) + sum(p * b[j] for j, p in row.items()) - b[i]
        for i, (duration, v, row) in enumerate(zip(calc._exact_durations, calc._departures, calc._exact_rows))
    ]
    assert result.row_drifts_exact == [str(value) for value in drifts]
    assert Fraction(result.drift_margin_exact) == -max(drifts) > 0
    assert result.w is None and result.v is None
    assert result.poisson_residual < 1e-12


def test_transient_harmonic_extension_and_worst_class():
    matrix = csr_matrix([[0, 0.25, 0.75], [0, 1, 0], [0, 0, 1]])
    classes, _, delta, count = PhaseAwareStabilityCalc._class_analysis(
        matrix, np.array([1.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]), 0.6
    )
    assert count == 1
    assert sorted(item["capacity"] for item in classes) == [0.5, 1.0]
    assert delta == pytest.approx([0.05, -0.4, 0.2])


@pytest.mark.parametrize("budget", [2, 4])
def test_continuation_contains_finer_solution_and_old_interval(budget):
    coarse = configured(PhaseAwareTtftCalc, budget=budget, truncation=8).run()
    fine = configured(PhaseAwareTtftCalc, budget=budget, truncation=32).run()
    old = configured(InfiniteStageAwareTtftCalc, budget=budget, truncation=32).run()
    for key in ("w", "v", "ttft"):
        lo, hi = coarse.mean_bounds[key]
        flo, fhi = fine.mean_bounds[key]
        olo, ohi = old.mean_bounds[key]
        assert lo <= flo <= fhi <= hi
        assert max(flo, olo) <= min(fhi, ohi) + 1e-13
    assert fine.phase_certificate["drift_verified"]
    assert not fine.roundoff_certified
    assert fine.return_bound_constants["core_area_upper"] > 0


@pytest.mark.parametrize("budget,law", [(2, ((2, 2, 1.0),)), (4, ((2, 2, 1.0),)), (4, LAW)])
def test_old_stability_refusal_is_not_instability(budget, law):
    calc = configured(PhaseAwareTtftCalc, budget=budget, law=law, truncation=32)
    calc.set_sources(0.8 / (np.array([row[2] for row in law]) @ calc.isolated_times()[0]))
    result = calc.run()
    assert result.dominating_load >= 1
    assert result.phase_certificate["stable"]
    assert all(hi - lo <= 0.001 * ((hi + lo) / 2) + 1e-8 for lo, hi in result.mean_bounds.values())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_states": 2}, "state budget"),
        ({"max_transition_words": 2}, "word budget"),
        ({"cap": 10**12, "budget": None}, "word budget"),
    ],
)
def test_complexity_guards(kwargs, match):
    with pytest.raises(ValueError, match=match):
        configured(**kwargs).run()


def test_positive_edges_cannot_be_pruned_and_reconfiguration():
    with pytest.raises(ValueError, match="underflow"):
        configured(law=((1, 1, 1.0), (2, 1, 1e-300))).run()
    calc = configured(PhaseAwareTtftCalc)
    calc.set_sources(100)
    with pytest.raises(ValueError, match="positive_drift_escape"):
        calc.run()
    calc.set_sources(1.3)
    assert calc.run().phase_certificate["stable"]


def test_normalization_matches_certificate_model():
    law = ((1, 1, 0.6), (3, 2, 0.4000000000001))
    calc = configured(PhaseAwareTtftCalc, law=law)
    expected = [Fraction(p) / sum(Fraction(row[2]) for row in law) for _, _, p in law]
    assert calc._mark_probabilities().tolist() == [float(p) for p in expected]
    assert calc.run().phase_certificate["drift_verified"]


@pytest.mark.parametrize(
    "cls,alpha,beta,rate,match",
    [
        (PhaseAwareStabilityCalc, 1e308, 1e308, 0.1, "iteration duration overflow"),
        (PhaseAwareTtftCalc, 1e200, 0, 1e-201, "continuation potential overflow"),
    ],
)
def test_extreme_intermediate_overflow_is_clear_refusal(cls, alpha, beta, rate, match):
    calc = cls(2, 2)
    calc.set_servers(((1, 1, 1.0),), alpha, beta, 0)
    calc.set_sources(rate)
    with pytest.raises(ValueError, match=match):
        calc.run()


def test_exterior_quadratic_potential_row_inequality():
    calc = configured(PhaseAwareTtftCalc, truncation=8)
    result = calc.run()
    values = result.return_bound_constants
    b = np.array(result.phase_certificate["corrector"])
    shifts = b - b.min() + 1
    eps, m2 = values["epsilon"], values["increment_second_max"]
    for h in (2, 7, 100):
        for i, row in enumerate(calc._phase_calc._exact_rows):
            mu = calc.l * calc._phase_calc._durations[i]
            r = np.array([mu - calc._phase_calc._departures[i] + shifts[j] - shifts[i] for j in row])
            probabilities = np.array([float(p) for p in row.values()])
            drift = probabilities @ r
            second = mu + probabilities @ (r**2)
            y = h + shifts[i]
            g_drift = (2 * y * drift + second + m2 * drift / eps) / (2 * eps)
            assert g_drift <= -h + 1e-10


@pytest.mark.parametrize("budget", [2, 4])
def test_independent_high_load_token_replay(budget):
    with open("tests/default_params.yaml", encoding="utf-8") as stream:
        tolerance = yaml.safe_load(stream)["phase_aware_stability"]
    capacity = configured(budget=budget).run().capacity
    rate = 0.85 * capacity
    samples = []
    for seed in range(96000, 96003):
        rng = np.random.default_rng(seed)
        arrivals = np.cumsum(rng.exponential(1 / rate, 50000))
        marks = np.array([item[:2] for item in LAW])[rng.choice(2, len(arrivals), p=[0.6, 0.4])]
        sim = PrefillDecodeSim(2, budget)
        sim.set_servers(1, 1, *COSTS)
        reference = sim.replay(arrivals, *marks.T, warmup_fraction=0.2, record_iterations=False)
        samples.append((reference.w[0], reference.v[0], reference.ttft_mean))
    means = np.mean(samples, axis=0)
    result = configured(PhaseAwareTtftCalc, budget=budget, truncation=128, rate=rate).run()
    for actual, predicted in zip(means, (result.w[0], result.v[0], result.ttft_mean)):
        assert predicted == pytest.approx(actual, rel=tolerance["mean_rel"])
