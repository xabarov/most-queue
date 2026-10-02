"""Deterministic empirical-CDF, fallback and circular-block contracts."""

import numpy as np
import pytest

from most_queue.random.trace_resampling import ConditionalEmpirical, circular_block_indices, lag_correlation


def test_exact_coarse_and_pooled_fallback():
    model = ConditionalEmpirical.fit(
        [1] * 4 + [2] * 4 + [3] * 2, [1, 2, 3, 4, 10, 20, 30, 40, 100, 200], exact=True, minimum=4
    )
    assert model.distribution(1)[1] == "exact"
    assert model.distribution(2)[0].mean == 25
    assert model.distribution(3)[1] == "coarse"
    assert model.distribution(8)[0].samples == (10, 20, 30, 40, 100, 200)
    assert model.distribution(33)[1] == "pooled"
    assert model.distribution(33)[0].mean == 41


@pytest.mark.parametrize("exact", [False, True])
def test_training_midrank_inverse_roundtrip_with_ties(exact):
    needs = np.array([1, 2, 1, 2, 3, 3, 1, 2, 33, 33])
    durations = np.array([1, 1, 1, 8, 2, 2, 3, 8, 100, 100])
    model = ConditionalEmpirical.fit(needs, durations, exact=exact, minimum=3)
    ranks = model.ranks(needs, durations)
    assert np.all((ranks > 0) & (ranks < 1))
    np.testing.assert_array_equal(model.quantiles(needs, ranks), durations)
    assert ranks[0] == ranks[2]


def test_heldout_outside_support_is_diagnostic_only():
    model = ConditionalEmpirical.fit([1, 1], [2, 4], minimum=2)
    np.testing.assert_array_equal(model.ranks([1, 1], [1, 10]), [0, 1])
    with pytest.raises(ValueError):
        model.quantiles([1, 1], [0, 1])


@pytest.mark.parametrize(
    "needs,services",
    [
        ([], []),
        ([1.5, 2], [1, 2]),
        ([True, False], [1, 2]),
        ([0, 1], [1, 2]),
        ([1, 2], [1]),
        ([1, 2], [-1, 2]),
        ([1, 2], [np.nan, 2]),
        ([[1, 2]], [[1, 2]]),
    ],
)
def test_invalid_history(needs, services):
    with pytest.raises(ValueError):
        ConditionalEmpirical.fit(needs, services)


@pytest.mark.parametrize("minimum", [0, 1, True, 2.5])
def test_bad_minimum(minimum):
    with pytest.raises(ValueError):
        ConditionalEmpirical.fit([1, 1], [1, 2], minimum=minimum)


def test_invalid_exact_and_mismatched_query():
    with pytest.raises(ValueError):
        ConditionalEmpirical.fit([1, 1], [1, 2], exact="yes")
    model = ConditionalEmpirical.fit([1, 1], [1, 2])
    for need in (True, 0, 2.5):
        with pytest.raises(ValueError):
            model.distribution(need)
    with pytest.raises(ValueError):
        model.quantiles([1, 2], [0.5])
    with pytest.raises(ValueError):
        model.ranks([1, 2], [1])


def test_circular_wrap_last_partial_block_and_iid():
    uniforms = np.array([0.91, 0.2, 0.3, 0.41, 0.2, 0.1, 0.61])
    np.testing.assert_array_equal(circular_block_indices(5, uniforms, 3), [4, 0, 1, 2, 3, 4, 3])
    np.testing.assert_array_equal(circular_block_indices(5, uniforms, 1), np.floor(uniforms * 5))


@pytest.mark.parametrize("length", [1, 2, 3, 5])
def test_every_offset_has_identical_uniform_donor_marginal(length):
    # Enumerate all possible starts: exact invariant, not a stochastic tolerance.
    columns = []
    for start in range(5):
        uniforms = np.full(length, (start + 0.5) / 5)
        columns.append(circular_block_indices(5, uniforms, length))
    for values in np.array(columns).T:
        np.testing.assert_array_equal(np.sort(values), np.arange(5))


@pytest.mark.parametrize(
    "size,uniforms,length",
    [
        (0, [0.5], 1),
        (True, [0.5], 1),
        (5, [0.5], 0),
        (5, [0.5], 6),
        (5, [0.5], 1.5),
        (5, [], 1),
        (5, [0], 1),
        (5, [1], 1),
        (5, [np.nan], 1),
        (5, [[0.5]], 1),
    ],
)
def test_invalid_block_arguments(size, uniforms, length):
    with pytest.raises(ValueError):
        circular_block_indices(size, uniforms, length)


def test_block_sampling_carries_dependence_on_constructed_history():
    tape = np.tile(np.repeat([0.1, 0.9], 50), 10)
    u = np.random.default_rng(5).uniform(0.0001, 0.9999, 20_000)
    iid = tape[circular_block_indices(len(tape), u)]
    blocks = tape[circular_block_indices(len(tape), u, 20)]
    assert abs(lag_correlation(iid)) < 0.04
    assert lag_correlation(blocks) > 0.8


def test_lag_degenerate_and_finite_validation():
    assert lag_correlation([1, 1, 1]) is None
    assert lag_correlation([1, 2]) is None
    assert lag_correlation([1, 2, 3, 4]) == pytest.approx(1)
    for values, lag in (([1, np.nan], 1), ([1, 2, 3], 0), ([[1, 2]], 1)):
        with pytest.raises(ValueError):
            lag_correlation(values, lag)
