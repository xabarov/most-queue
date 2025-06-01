"""
Laplace-Stieltjes transforms
"""
import math

from most_queue.rand_distribution import GammaParams, H2Params


def lst_exp(mu, s):
    """
    Calculates the Laplace-Stieltjes transform of an exponential distribution.
    """
    return mu / (mu + s)


def lst_h2(h2_params: H2Params, s: float) -> float:
    """
    Calculate the Laplace-Stieltjes transform of an H2 distribution.
    """
    return h2_params.p1*lst_exp(h2_params.mu1, s) + (1.0 - h2_params.p1)*lst_exp(h2_params.mu2, s)


def lst_gamma(gamma_params: GammaParams, s: float) -> float:
    """
    Calculate the Laplace-Stieltjes transform of a gamma distribution.
    """
    return math.pow(gamma_params.mu/(s+gamma_params.mu), gamma_params.alpha)



