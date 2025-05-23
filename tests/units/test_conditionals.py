"""
Test the conditional moments calculation functions.
"""
import numpy as np
import pytest

from most_queue.theory.utils.conditional import (
    calc_b_min_h2_and_exp,
    moments_exp_less_than_h2,
    moments_h2_less_than_exp,
)
from most_queue.general.tables import times_print
from most_queue.rand_distribution import ExpDistribution, H2Distribution, H2Params


@pytest.fixture
def setup_params():
    """Fixture to set up and return common test parameters and generated data"""
    total = 100_000
    gamma = 2.0

    p1 = 0.7
    mu1 = 0.2
    mu2 = 1.4

    h2_params = H2Params(p1=p1, mu1=mu1, mu2=mu2)

    exp_rand = ExpDistribution(gamma)
    h2_rand = H2Distribution(h2_params)

    # Pre-allocate arrays for better performance
    h2_values = np.zeros(total)
    exp_values = np.zeros(total)

    # Generate all random values upfront
    for i in range(total):
        h2_values[i] = h2_rand.generate()
        exp_values[i] = exp_rand.generate()

    return {
        'total': total,
        'gamma': gamma,
        'h2_params': h2_params,
        'exp_rand': exp_rand,
        'h2_rand': h2_rand,
        'h2_values': h2_values,
        'exp_values': exp_values
    }


def test_moments_exp_less_than_h2(setup_params):
    """
    Test moments for Exp < H2 condition
    """
    params = setup_params
    h2_values = params['h2_values']
    exp_values = params['exp_values']

    # Calculate which H2 values are greater than corresponding Exp values
    mask = h2_values > exp_values

    # Get valid Exp values where H2 > Exp (since we're testing Y < X for H2)
    valid_exp = exp_values[mask]

    if len(valid_exp) == 0:
        sim_moments = [0, 0, 0]
    else:
        sim_moments = [
            np.mean(valid_exp),
            np.mean(valid_exp ** 2),
            np.mean(valid_exp ** 3)
        ]

    calc_moments = moments_exp_less_than_h2(
        gamma=params['gamma'],
        h2_params=params['h2_params']
    )

    times_print(sim_moments, calc_moments, is_w=False,
                header='Results for exp < h2')

    # Add assertions
    np.testing.assert_allclose(sim_moments, calc_moments[:3], atol=1e-2)


def test_moments_h2_less_than_exp(setup_params):
    """
    Test moments for H2 < Exp condition
    simulation vs calculation for H2 < Exp condition
    """
    params = setup_params
    h2_values = params['h2_values']
    exp_values = params['exp_values']

    # Calculate which H2 values are less than corresponding Exp values
    mask = h2_values < exp_values

    # Get valid H2 values where H2 < Exp
    valid_h2 = h2_values[mask]

    if len(valid_h2) == 0:
        sim_moments = [0, 0, 0]
    else:
        sim_moments = [
            np.mean(valid_h2),
            np.mean(valid_h2 ** 2),
            np.mean(valid_h2 ** 3)
        ]

    calc_moments = moments_h2_less_than_exp(
        gamma=params['gamma'],
        h2_params=params['h2_params']
    )

    times_print(sim_moments, calc_moments, is_w=False,
                header='Results for h2 < exp')

    # Add assertions
    np.testing.assert_allclose(sim_moments, calc_moments[:3], atol=1e-2)

def test_min_h2_and_exp(setup_params):
    """
    Test min of H2 and Exp values simulation vs calculation
    """
    params = setup_params
    h2_values = params['h2_values']
    exp_values = params['exp_values']

    # Calculate min of H2 and Exp values using numpy
    min_h2_exp = np.minimum(h2_values, exp_values)

    if len(min_h2_exp) == 0:
        sim_moments = [0, 0, 0]
    else:
        sim_moments = [
            np.mean(min_h2_exp),
            np.mean(min_h2_exp ** 2),
            np.mean(min_h2_exp ** 3)
        ]

    calc_moments = calc_b_min_h2_and_exp(
        h2_params=params['h2_params'],
        mu=params['gamma']
    )

    times_print(sim_moments, calc_moments, is_w=False,
                header='Results for min(H2, Exp)')

    # Add assertions
    np.testing.assert_allclose(sim_moments, calc_moments[:3], atol=1e-2)
    