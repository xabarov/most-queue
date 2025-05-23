"""
Conditional moments of random variables.
"""
import math

import numpy as np

from most_queue.general.distribution_params import H2Params
from most_queue.rand_distribution import H2Distribution


def moments_h2_less_than_exp(gamma: float, h2_params: H2Params):
    """
    Compute the initial moments mean of Y given Y < X,
    where Y is a hyperexponential random variable with parameters p1, mu1, and mu2.
    X is exponentially distributed with rate gamma.
    """
    y = [h2_params.p1, 1-h2_params.p1]
    mu = [h2_params.mu1, h2_params.mu2]
    ps = [y[i]*mu[i] for i in range(2)]

    coef = sum([y[i]*mu[i]/(mu[i]+gamma) for i in range(2)])

    b = [sum([math.factorial(k+1)*ps[i]/pow(mu[i]+gamma, k+2)
              for i in range(2)])/coef for k in range(3)]

    return np.array([mom.real for mom in b])


def moments_exp_less_than_h2(gamma: float, h2_params: H2Params):
    """
    Compute the initial moments mean of X given X < Y,
    where X is exponentially distributed with rate gamma.
    Y is a hyperexponential random variable with parameters p1, mu1, and mu2.

    1. f_{X|X<Y}(x) = f_X(x) * P(Y > x) / P(X < Y)
    2. P(Y > x) = ∫_{x}^{∞} [p1mu1exp(-mu1y) + (1-p1)mu2exp(-mu2y)] dy
    3. P(X < Y) =  ∫_{0}^{∞} f_X(x) * P(Y > x) dx
    4. f_{X|X<Y}(x) = gamma * exp(-gamma*x) * P(Y > x) / P(X < Y)
    """
    p1, mu1, mu2 = h2_params.p1, h2_params.mu1, h2_params.mu2
    y = [p1, 1-p1]
    mu = [mu1, mu2]
    denominator = np.sum([y[i]/(mu[i]+gamma) for i in range(2)])
    b = []

    # Step 2: Compute the Numerator
    for k in range(3):
        numerator = math.factorial(
            k+1) * np.sum([y[i] / (mu[i]+gamma)**(k+2) for i in range(2)])
        b.append(numerator / denominator)

    return np.array([mom.real for mom in b])


def calc_b_min_h2_and_exp(h2_params: H2Params, mu: float) -> list[float]:
    """
    Calculate the minimum of H2 and exponential distribution moments.
    Parameters
    ----------
    h2_params : H2Params
    Parameters for the H2 distribution.
    mu : float
        Arrival rate of negative
    """
    b = H2Distribution.calc_theory_moments(H2Params(p1=h2_params.p1,
                                                    mu1=mu + h2_params.mu1,
                                                    mu2=mu + h2_params.mu2))

    return b
