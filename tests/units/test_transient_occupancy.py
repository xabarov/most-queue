"""Analytic and event-driven oracles for transient occupancy projection."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import poisson

from most_queue.theory.continuous_batching import TransientOccupancyQueueCalc
from most_queue.theory.fifo.erlang import ErlangCCalc


def configured(k=2, cutoff=30, schedule=1.2, mu=1.0):
    """Create a configured calculator without invoking its private methods."""
    calc = TransientOccupancyQueueCalc(k, cutoff)
    calc.set_sources(schedule)
    calc.set_servers(mu)
    return calc


def test_two_kept_states_have_closed_form_with_escape():
    result = configured(k=1, cutoff=1, schedule=1).run([0, 0.4, 2])
    t = result.times
    root = math.sqrt(5)
    zero = np.exp(-1.5 * t) * (np.cosh(root * t / 2) + np.sinh(root * t / 2) / root)
    one = np.exp(-1.5 * t) * 2 * np.sinh(root * t / 2) / root
    np.testing.assert_allclose(result.state_probabilities, np.column_stack((zero, one)), atol=2e-15)
    np.testing.assert_allclose(result.escape_probability, 1 - zero - one, atol=2e-15)
    assert result.escape_probability[-1] > 0.3
    assert result.state_probabilities[-1].sum() < 0.7  # must not reflect or normalize
    assert result.p is None and result.w is None and result.v is None
    assert not result.roundoff_certified


def test_no_arrivals_pure_death_queue_matches_poisson_departures():
    result = configured(k=1, cutoff=6, schedule=0, mu=1.7).run([0, 0.5, 2, 5], initial=6)
    for i, t in enumerate(result.times):
        expected = np.r_[poisson.sf(5, 1.7 * t), poisson.pmf(6 - np.arange(1, 7), 1.7 * t)]
        np.testing.assert_allclose(result.state_probabilities[i], expected, atol=4e-15)
    np.testing.assert_array_equal(result.escape_probability, 0)


def test_immediate_wait_and_erlang_tail_from_known_initial_population():
    result = configured(k=3, cutoff=5, mu=lambda occupancy: 2 / occupancy).run([0], initial=5)
    # Three completions at total rate 2 are needed to reach a free slot.
    expected = math.exp(-0.8) * (1 + 0.8 + 0.8**2 / 2)
    np.testing.assert_allclose(result.waiting_tail_bounds(0.4), [[expected, expected]])
    np.testing.assert_array_equal(result.waiting_tail_bounds(0), [[1, 1]])
    free = configured(k=3, cutoff=5).run([0], initial=2)
    np.testing.assert_array_equal(free.waiting_tail_bounds(0), [[0, 0]])


def test_initial_mixture_and_population_support_are_not_conditioned():
    result = configured(k=2, cutoff=3).run([0], initial=[0.2, 0.3, 0.1, 0.4])
    np.testing.assert_allclose(result.waiting_tail_bounds(0), [[0.5, 0.5]])
    np.testing.assert_allclose(result.population_tail_bounds(3), [[0.4, 0.4]])
    np.testing.assert_array_equal(result.population_tail_bounds(0), [[1, 1]])
    np.testing.assert_array_equal(result.population_tail_bounds(9), [[0, 0]])


def test_finite_projection_bounds_contain_immigration_death_anchor():
    # Linear deaths before exit: the unbounded M/M/infinity Poisson law is
    # another process coupled to this projection until the first exit.
    result = configured(k=8, cutoff=8, schedule=4.0).run([0.2, 1, 3])
    for level in [1, 4, 8, 9]:
        actual = poisson.sf(level - 1, 4 * (1 - np.exp(-result.times)))
        bounds = result.population_tail_bounds(level)
        assert np.all(bounds[:, 0] <= actual + 1e-13)
        assert np.all(actual <= bounds[:, 1] + 1e-13)


def test_refinement_nests_bounds_and_escape_is_cumulative():
    schedule = [(0, 1), (1.0, 4), (2.5, 0)]
    small = configured(cutoff=5, schedule=schedule).run(np.linspace(0, 8, 25))
    large = configured(cutoff=35, schedule=schedule).run(small.times)
    for deadline in [0, 0.5, 2]:
        lo, hi = small.waiting_tail_bounds(deadline).T
        lower, upper = large.waiting_tail_bounds(deadline).T
        assert np.all(lo <= lower + 1e-12)
        assert np.all(upper <= hi + 1e-12)
    assert np.min(np.diff(small.escape_probability)) >= -1e-14
    assert small.escape_probability[-1] > 0.2  # cannot reset error at the rate drop
    assert large.population_tail_bounds(6)[-1, 1] < small.escape_probability[-1]


def test_rate_change_is_split_even_without_an_observation_at_change():
    schedule = [(0, 0.5), (0.13, 5), (0.71, 0), (1.1, 1)]
    sparse = configured(schedule=schedule).run([0.8, 1.3])
    dense = configured(schedule=schedule).run([0, 0.13, 0.4, 0.71, 0.8, 1.1, 1.3])
    np.testing.assert_allclose(sparse.state_probabilities, dense.state_probabilities[[4, 6]], atol=1e-14)
    np.testing.assert_allclose(sparse.escape_probability, dense.escape_probability[[4, 6]], atol=1e-14)
    np.testing.assert_array_equal(dense.arrival_rates, [0.5, 5, 5, 0, 0, 1, 1])


def test_long_time_constant_rate_recovers_erlang_c_waiting_tail():
    # Independent infinite-buffer Erlang C formula, not a second projection.
    result = configured(k=3, cutoff=70, schedule=1.7).run([90])
    anchor = ErlangCCalc(n=3)
    anchor.set_sources(1.7)
    anchor.set_servers(1)
    probability = anchor.get_waiting_probability()
    for deadline in [0, 0.5, 2]:
        exact = probability * math.exp(-(3 - 1.7) * deadline)
        np.testing.assert_allclose(result.waiting_tail_bounds(deadline), exact, atol=2e-11)


def test_independent_unbounded_event_simulation_under_a_burst():
    with (Path(__file__).parents[1] / "default_params.yaml").open(encoding="utf-8") as stream:
        parameters = yaml.safe_load(stream)["transient_occupancy"]
    rng = np.random.default_rng(72914)
    times = [1, 2, 4]
    counts = np.zeros(len(times))
    replications = parameters["replications"]
    for _ in range(replications):
        population = 0
        now = 0.0
        for index, target in enumerate(times):
            while now < target:
                rate = 0.7 if now < 1 else (4 if now < 2 else 0.7)
                stop = min(target, 1 if now < 1 else (2 if now < 2 else math.inf))
                occupancy = min(population, 2)
                death = occupancy / (0.5 + 0.2 * occupancy)
                event = now + rng.exponential(1 / (rate + death))
                if event >= stop:
                    now = stop
                else:
                    now = event
                    population += 1 if rng.random() < rate / (rate + death) else -1
            counts[index] += population >= 2
    result = configured(schedule=[(0, 0.7), (1, 4), (2, 0.7)], mu=lambda j: 1 / (0.5 + 0.2 * j)).run(times)
    np.testing.assert_allclose(
        counts / replications, result.waiting_tail_bounds(0)[:, 0], atol=parameters["probability_atol"], rtol=0
    )


def test_overload_is_allowed_without_inventing_a_stationary_distribution():
    result = configured(k=1, cutoff=12, schedule=5, mu=1).run([1, 4])
    assert result.escape_probability[-1] > 0.5
    assert result.mass_residual < 1e-12
    assert result.pi is None and result.utilization is None


def test_reconfiguration_changes_results_without_stale_cached_rates():
    calc = configured(schedule=2)
    old = calc.run([2])
    calc.set_sources(0)
    new = calc.run([2])
    np.testing.assert_array_equal(new.waiting_tail_bounds(0), [[0, 0]])
    assert old.waiting_tail_bounds(0)[0, 0] > 0
    calc.set_sources(2)
    calc.set_servers(4)
    assert calc.run([2]).waiting_tail_bounds(0)[0, 0] < old.waiting_tail_bounds(0)[0, 0]


@pytest.mark.parametrize("schedule", [[], [(1, 1)], [(0, 1), (0, 2)], [(0, 1), (-1, 2)], -1, np.nan, [(0, np.inf)]])
def test_rejects_invalid_schedules(schedule):
    with pytest.raises(ValueError):
        configured(schedule=schedule)


@pytest.mark.parametrize("times", [[], [1, 1], [2, 1], [-1], [np.nan], [np.inf]])
def test_rejects_invalid_sample_times(times):
    with pytest.raises(ValueError):
        configured().run(times)


@pytest.mark.parametrize("mu", [0, -1, np.inf, np.nan, lambda j: 0 if j == 2 else 1])
def test_rejects_invalid_service_rates(mu):
    with pytest.raises(ValueError):
        configured(mu=mu)


@pytest.mark.parametrize("initial", [-1, 31, True, [0.5, 0.5], [0.0] * 31])
def test_rejects_invalid_initial_laws(initial):
    with pytest.raises(ValueError):
        configured().run([0], initial=initial)


@pytest.mark.parametrize("k,cutoff", [(True, 2), (0, 2), (1.5, 2), (3, 2), (1, 2.5)])
def test_rejects_invalid_state_domains(k, cutoff):
    with pytest.raises(ValueError):
        configured(k=k, cutoff=cutoff)


def test_budgets_configuration_and_observable_validation():
    with pytest.raises(ValueError, match="max_states"):
        TransientOccupancyQueueCalc(2, 100, max_states=50)
    with pytest.raises(ValueError):
        TransientOccupancyQueueCalc(2).run([0])
    result = configured().run([0])
    for deadline in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            result.waiting_tail_bounds(deadline)
    for level in [-1, 0.5, True]:
        with pytest.raises(ValueError):
            result.population_tail_bounds(level)
