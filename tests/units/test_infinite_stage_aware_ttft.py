"""Killed-kernel mass, analytic containment and independent infinite replay."""

import itertools

import numpy as np
import pytest
import yaml
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.stats import poisson

from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import InfiniteStageAwareTtftCalc, StageAwareTtftCalc

LAW = ((1, 1, 0.6), (3, 2, 0.4))
COSTS = (0.1, 0.02, 0.003)


def configured(cap=2, budget=2, truncation=8, rate=1.3, **kwargs):
    """Use frozen costs and iid marks, never fitted to the replay."""
    calc = InfiniteStageAwareTtftCalc(cap, budget, truncation, **kwargs)
    calc.set_servers(LAW, *COSTS)
    calc.set_sources(rate)
    return calc


@pytest.mark.parametrize("budget", [2, 4, None])
@pytest.mark.parametrize("truncation", [0, 2, 8])
def test_kernel_mass_and_real_infinite_rewards(budget, truncation):
    calc = configured(budget=budget, truncation=truncation)
    states, matrix, rewards, exits, tail, _ = calc._killed_kernel(np.array([0.6, 0.4]), calc.dominating_work())
    assert np.asarray(matrix.sum(axis=1)).ravel() + exits[:, 0] == pytest.approx(np.ones(len(states)), abs=2e-14)
    assert exits[:, 0].max() > 0
    assert np.all(exits >= 0)
    assert np.all(tail >= 0)
    for i, (active, waiting) in enumerate(states[1:], start=1):
        duration, _ = calc._iteration(active)
        assert rewards[i, 2] == pytest.approx(waiting * duration + calc.l * duration**2 / 2)
        assert waiting <= truncation
    assert np.max(matrix.sum(axis=1)) <= 1 + 2e-14


@pytest.mark.parametrize("mean,threshold,offset", [(0.01, 3, -2), (0.3, 9, -1), (2.7, 4, 2), (0.7, 1, 0)])
def test_overshoot_against_explicit_positive_series(mean, threshold, offset):
    counts = np.arange(threshold, 180)
    h = counts + offset
    expected = np.array([np.sum(poisson.pmf(counts, mean) * h**power) for power in range(3)])
    assert configured()._overshoot_moments(mean, threshold, offset) == pytest.approx(expected, rel=1e-12, abs=1e-30)


@pytest.mark.parametrize("budget", [1, 2, None])
def test_killed_bounds_contain_independent_b1_pk(budget):
    # B1 run takes the shortcut; exercise its killed kernel separately to test
    # the error derivation against a genuinely analytic infinite reference.
    calc = configured(cap=1, budget=budget, truncation=2)
    probs = np.array([0.6, 0.4])
    states, matrix, rewards, exits, _, _ = calc._killed_kernel(probs, calc.dominating_work())
    visits = spsolve(eye(len(states) - 1, format="csc") - matrix[1:, 1:].T.tocsc(), matrix[0, 1:].toarray().ravel())
    visits = np.r_[1.0, visits]
    retained, remainder = visits @ rewards, (visits @ exits)[1:]
    ref = StageAwareTtftCalc(1, budget)
    ref.set_servers(LAW, *COSTS)
    ref.set_sources(1.3)
    result = ref.run()
    for i, exact in ((1, result.ttft_mean), (2, result.w[0]), (3, result.v[0])):
        lower = retained[i] / (calc.l * (retained[0] + remainder[0]))
        upper = (retained[i] + remainder[i]) / (calc.l * retained[0])
        assert lower <= exact <= upper


def test_credit_domination_for_active_states():
    calc = configured(truncation=2)
    states, *_ = calc._killed_kernel(np.array([0.6, 0.4]), calc.dominating_work())
    for active, _ in states[1:]:
        duration, survivors = calc._iteration(active)
        # Independently compute all consumed token credits, including jobs that
        # finish. Decoder reordering is irrelevant to this sum.
        remaining = calc.token_budget - sum(q >= LAW[k][0] for k, q in active)
        credits = 0.0
        for k, q in active:
            prompt, output, _ = LAW[k]
            amount = 1 if q >= prompt else min(prompt - q, remaining)
            if q < prompt:
                remaining -= amount
            credits += sum(COSTS[0] + COSTS[1] + COSTS[2] * attained for attained in range(q + 1, q + amount + 1))
            assert q + amount <= prompt + output
        assert duration <= credits + 1e-14
        assert all(q < LAW[k][0] + LAW[k][1] for k, q in survivors)


@pytest.mark.parametrize("budget", [2, 4, None])
def test_refinement_contains_finer_means(budget):
    coarse = configured(budget=budget, truncation=4).run()
    fine = configured(budget=budget, truncation=16).run()
    for key in ("w", "v", "ttft"):
        lo, hi = coarse.mean_bounds[key]
        flo, fhi = fine.mean_bounds[key]
        assert lo <= flo <= fhi <= hi
        assert fhi - flo < (hi - lo) / 1000
    assert fine.waiting_tail_upper < coarse.waiting_tail_upper
    assert fine.cycle_exit_probability < coarse.cycle_exit_probability
    assert fine.accepted_rate == 1.3
    assert fine.blocking_probability == 0
    assert not fine.roundoff_certified


@pytest.mark.parametrize("budget", [2, 4])
def test_independent_infinite_token_replay(budget):
    with open("tests/default_params.yaml", encoding="utf-8") as stream:
        tolerances = yaml.safe_load(stream)["infinite_stage_aware_ttft"]
    rng = np.random.default_rng(95000)
    arrivals = np.cumsum(rng.exponential(1 / 1.3, 30000))
    marks = np.array([item[:2] for item in LAW])[rng.choice(2, len(arrivals), p=[0.6, 0.4])]
    sim = PrefillDecodeSim(2, budget)
    sim.set_servers(1, 1, *COSTS)
    reference = sim.replay(arrivals, *marks.T, warmup_fraction=0.2)
    result = configured(budget=budget, truncation=16).run()
    assert result.ttft_mean == pytest.approx(reference.ttft_mean, rel=tolerances["mean_rel"])
    assert result.v[0] == pytest.approx(reference.v[0], rel=tolerances["mean_rel"])
    assert result.w[0] == pytest.approx(reference.w[0], abs=tolerances["wait_abs"])


def test_b1_shortcut_and_low_load():
    result = configured(cap=1, truncation=0).run()
    assert result.states == 0
    assert result.waiting_tail_upper is None  # PK shortcut has NO queue-tail estimate.
    assert result.truncation is None
    assert result.mean_bounds["ttft"] == (result.ttft_mean, result.ttft_mean)
    low = configured(rate=1e-5).run()
    assert low.ttft_mean == pytest.approx(0.6 * 0.249 + 0.4 * 0.407, rel=1e-4)


@pytest.mark.parametrize("rate", [3, 100])
def test_sufficient_certificate_refusal_is_not_instability(rate):
    with pytest.raises(ValueError, match="NOT instability"):
        configured(rate=rate).run()
    if rate == 100:
        with pytest.raises(ValueError, match="requires"):
            configured(cap=1, rate=rate).run()


@pytest.mark.parametrize(
    "kwargs,match", [({"max_states": 2}, "state budget"), ({"max_transition_words": 2}, "word budget")]
)
def test_complexity_guards(kwargs, match):
    with pytest.raises(ValueError, match=match):
        configured(**kwargs).run()
    with pytest.raises(ValueError, match="word budget"):
        configured(truncation=10**12, max_transition_words=10).run()


@pytest.mark.parametrize("truncation", [-1, True, 1.2])
def test_invalid_truncation(truncation):
    with pytest.raises(ValueError):
        configured(truncation=truncation)


def test_configuration_required_and_reconfigurable():
    with pytest.raises(ValueError):
        InfiniteStageAwareTtftCalc().run()
    with pytest.raises(ValueError):
        InfiniteStageAwareTtftCalc().dominating_work()
    calc = configured()
    previous = calc.run().ttft_mean
    calc.set_sources(0.5)
    assert calc.run().ttft_mean < previous
    for costs in itertools.product((0.1,), (0.0, 0.02), (0.0, 0.003)):
        calc.set_servers(LAW, *costs)
        assert np.all(calc.dominating_work() > 0)
