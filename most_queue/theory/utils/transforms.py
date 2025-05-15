import numpy as np


def laplace_stieltjes_exp_transform(mu, s):
    """
    Calculates the Laplace-Stieltjes transform of an exponential distribution.
    """
    return mu / (mu + s)