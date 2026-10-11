"""Weighted-row identity, symmetry counterexamples and independent replay."""

import itertools
import math
from collections import defaultdict

import numpy as np
import pytest
import yaml
from scipy.stats import poisson

from examples.stage_aware_ttft_oracle import replay_loss
from most_queue.theory.continuous_batching import ReducedStageAwareTtftCalc, StageAwareTtftCalc
from most_queue.theory.continuous_batching.reduced_stage_aware_ttft import ReducedStageAwareTtftResults

LAW = ((1, 1, 0.6), (3, 2, 0.4))
COSTS = (0.1, 0.02, 0.003)


def configured(cls=ReducedStageAwareTtftCalc, cap=2, budget=2, queue=2, rate=1.3, **kwargs):
    """Configure identical laws without fitting any costs to the arbiter."""
    calc = cls(cap, budget, queue, **kwargs)
    calc.set_sources(rate)
    calc.set_servers(LAW, *COSTS)
    return calc


def advance(active, cap, budget):
    """Independent discrete token accounting for a single iteration."""
    assert len(active) <= cap
    remaining = math.inf if budget is None else budget - sum(q >= LAW[k][0] for k, q in active)
    updated, duration = [], COSTS[0]
    for k, q in active:
        prompt, output, _ = LAW[k]
        tokens = 1 if q >= prompt else min(prompt - q, remaining)
        if q < prompt:
            remaining -= tokens
        if tokens:
            duration += COSTS[1] * tokens + COSTS[2] * (q + tokens)
        if q + tokens < prompt + output:
            updated.append((k, q + tokens))
    return duration, tuple(updated)


def full_row(active, queue, capacity, rate=1.3):
    """Enumerate the unreduced marked queue independently of both solvers."""
    targets = defaultdict(float)
    if not active:
        for k, (_, _, p) in enumerate(LAW):
            targets[(((k, 0),), ())] += p
        return targets
    duration, survivors = advance(active, 2, 2)
    space = capacity - len(queue)
    for count in range(space + 1):
        mass = poisson.pmf(count, rate * duration) if count < space else poisson.sf(count - 1, rate * duration)
        for word in itertools.product(range(2), repeat=count):
            probability = mass * math.prod(LAW[k][2] for k in word)
            waiting = queue + word
            fill = min(2 - len(survivors), len(waiting))
            target = (survivors + tuple((k, 0) for k in waiting[:fill]), waiting[fill:])
            targets[target] += probability
    return targets


def test_product_lifting_identity_every_reachable_row():
    # L P_full = P_reduced L, WITHOUT decoder symmetry. This is stronger than
    # matching stationary means and is deliberately not called strong lumping.
    calc = configured(canonical_decoders=False)
    states, matrix, rewards, _ = calc._kernel(np.array([0.6, 0.4]))
    assert np.allclose(matrix.sum(axis=1), 1)
    assert np.all(rewards >= 0)
    for i, (active, length) in enumerate(states):
        left, right = defaultdict(float), defaultdict(float)
        for waiting in itertools.product(range(2), repeat=length):
            weight = math.prod(LAW[k][2] for k in waiting)
            for target, probability in full_row(active, waiting, 2).items():
                left[target] += weight * probability
        row = matrix.getrow(i)
        for j, probability in zip(row.indices, row.data):
            next_active, next_length = states[j]
            for waiting in itertools.product(range(2), repeat=next_length):
                right[next_active, waiting] += probability * math.prod(LAW[k][2] for k in waiting)
        for target in left.keys() | right.keys():
            assert left[target] == pytest.approx(right[target], abs=2e-14)


@pytest.mark.parametrize("budget", [2, 4, None])
@pytest.mark.parametrize("queue", [0, 2, 4])
@pytest.mark.parametrize("canonical", [False, True])
def test_same_stationary_rewards_as_full(budget, queue, canonical):
    full = configured(StageAwareTtftCalc, budget=budget, queue=queue).run()
    reduced = configured(budget=budget, queue=queue, canonical_decoders=canonical).run()
    for field in ("w", "v", "ttft_mean", "blocking_probability", "accepted_rate", "utilization"):
        assert getattr(reduced, field) == pytest.approx(getattr(full, field), rel=1e-10, abs=1e-10)
    assert reduced.states <= full.states
    assert reduced.stationary_residual < 1e-11
    assert isinstance(reduced, ReducedStageAwareTtftResults)


def test_cap_three_and_decoder_symmetry():
    full = configured(StageAwareTtftCalc, cap=3, budget=3).run()
    queue_only = configured(cap=3, budget=3, canonical_decoders=False).run()
    reduced = configured(cap=3, budget=3).run()
    assert reduced.ttft_mean == pytest.approx(full.ttft_mean, rel=1e-10)
    assert reduced.w == pytest.approx(full.w, rel=1e-10)
    assert reduced.states <= queue_only.states < full.states


def test_decoder_permutations_preserve_transition_not_prefill_order():
    calc = configured(cap=3, budget=3)
    active = ((1, 3), (0, 0), (0, 1))
    for permutation in itertools.permutations(active):
        assert calc._canonical(permutation) == calc._canonical(active)
        duration, survivors = calc._iteration(permutation)
        original_duration, original_survivors = calc._iteration(active)
        assert duration == pytest.approx(original_duration)
        assert calc._canonical(survivors) == calc._canonical(original_survivors)
    prefills = ((0, 0), (1, 0))
    assert calc._canonical(prefills) != calc._canonical(prefills[::-1])
    small = configured()
    assert small._iteration(prefills)[1] != small._iteration(prefills[::-1])[1]
    # Duplicate decoder multiplicities cannot be discarded.
    assert len(calc._canonical(((0, 1), (0, 1)))) == 2


def test_waiting_count_is_not_strong_lumpability():
    # Same active state and length, different known head => different admission.
    first = full_row(((0, 1),), (0,), 1)
    second = full_row(((0, 1),), (1,), 1)
    assert first[(((0, 0),), ())] == pytest.approx(1)
    assert second[(((0, 0),), ())] == 0


@pytest.mark.parametrize("queue", [0, 2, 4])
def test_independent_loss_replay(queue):
    with open("tests/default_params.yaml", encoding="utf-8") as stream:
        tolerances = yaml.safe_load(stream)["reduced_stage_aware_ttft"]
    rng = np.random.default_rng(94001)
    arrivals = np.cumsum(rng.exponential(0.5, 30000))
    marks = np.array([item[:2] for item in LAW])[rng.choice(2, 30000, p=[0.6, 0.4])]
    oracle = replay_loss(arrivals, marks, 2, 2, queue, COSTS, warmup=3000)
    result = configured(queue=queue, rate=2).run()
    assert result.ttft_mean == pytest.approx(oracle["ttft"], rel=tolerances["mean_rel"])
    assert result.v[0] == pytest.approx(oracle["v"], rel=tolerances["mean_rel"])
    assert result.w[0] == pytest.approx(oracle["w"], abs=tolerances["wait_abs"])
    assert result.blocking_probability == pytest.approx(oracle["loss"], abs=tolerances["loss_abs"])


def test_b1_pk_is_preserved_and_zero_load():
    full = configured(StageAwareTtftCalc, cap=1).run()
    reduced = configured(cap=1).run()
    assert reduced.ttft_mean == full.ttft_mean
    assert reduced.states == reduced.transition_words == 0
    assert "infinite" in reduced.model
    low = configured(rate=1e-5).run()
    assert low.ttft_mean == pytest.approx(0.6 * 0.249 + 0.4 * 0.407, rel=1e-4)
    with pytest.raises(ValueError, match="requires"):
        configured(cap=1, rate=100).run()
    assert configured(rate=100).run().blocking_probability > 0.8


@pytest.mark.parametrize(
    "kwargs,match", [({"max_states": 2}, "state budget"), ({"max_transition_words": 2}, "word budget")]
)
def test_explicit_budgets(kwargs, match):
    with pytest.raises(ValueError, match=match):
        configured(**kwargs).run()
    # Capacity must not cause a huge allocation before either guard fires.
    with pytest.raises(ValueError, match=match):
        configured(queue=10**12, **kwargs).run()


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_symmetry_flag_is_not_truthiness(value):
    with pytest.raises(ValueError, match="bool"):
        configured(canonical_decoders=value)


def test_reconfigure_does_not_reuse_previous_kernel():
    calc = configured()
    calc.run()
    calc.set_sources(0.7)
    calc.set_servers([(2, 2, 1)], *COSTS)
    ref = StageAwareTtftCalc(2, 2, 2)
    ref.set_sources(0.7)
    ref.set_servers([(2, 2, 1)], *COSTS)
    assert calc.run().ttft_mean == pytest.approx(ref.run().ttft_mean)
