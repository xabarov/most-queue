import math

import numpy as np
from scipy.misc import derivative

from most_queue.theory.utils.transforms import lst_gamma, lst_h2
from most_queue.rand_distribution import GammaDistribution, H2Distribution

def busy_calc(l: float, b: list[float], num: int = 5):
    """
    Calculation of initial moments of continuous busy period for M/G/1 queue
    By default, the first five are calculated.
    :param l: - intensity of input stream
    :param b: [j], j=1..num, initial moments of service time
    :param num: number of initial moments to calculate
    :return: list of initial moments of continuous busy period for M/G/1 queue
    """
    num = min(num, len(b))
    busy_moments = []
    ro = l * b[0]
    busy_moments.append(b[0] / (1 - ro))
    z = 1 + l * busy_moments[0]
    if num > 1:
        busy_moments.append(b[1] / math.pow(1 - ro, 3))
    if num > 2:
        busy_moments.append(b[2] / math.pow(1 - ro, 4) + 3 *
                            l * b[1] * b[1] / math.pow(1 - ro, 5))
    if num > 3:
        chisl = b[3] * math.pow(z, 4) + 6 * b[2] * l * busy_moments[1] * z * z + b[1] * (
                3 * math.pow(l * busy_moments[1], 2) + 4 * l * busy_moments[2] * z)
        busy_moments.append(chisl / (1 - ro))
    if num > 4:
        chisl = b[4] * math.pow(z, 5) + 10 * b[3] * l * busy_moments[1] * math.pow(z, 3) + \
                b[2] * (15 * math.pow(l * busy_moments[1], 2) * z + 10 * l * busy_moments[2 * z * z]) + b[1] * (
            5 * l * busy_moments[3] * z + 10 * l * l * busy_moments[1] * busy_moments[2])
        busy_moments.append(chisl / (1 - ro))

    return busy_moments


def busy_calc_lst(l: float, b: list[float], lst_function='gamma'):
    """
    Calculate the busy period moments using the Laplace-Stieltjes transform.
    Parameters
    ----------
    l : float
        Arrival rate.
    b : list[float]
        service time initial moments.
    lst_function : str
        The function to use for the Laplace-Stieltjes transform.
    Returns
    -------
    busy_moments : list[float]
        The first three moments of the busy period
    """
    if lst_function == 'gamma':
        lst_function = lst_gamma
        params = GammaDistribution.get_params(b)
    elif lst_function == 'h2':
        lst_function = lst_h2
        params = H2Distribution.get_params(b)
    else:
        raise ValueError('lst_function must be one of "gamma" or "h2"')

    def calc_busy_pls(s):
        y = 0
        while True:
            y_new = lst_function(params, s + l-l*y)
            if abs(y_new - y) < 1e-6:
                return y_new
            y = y_new
    busy = [0, 0, 0]
    for i in range(3):
        busy[i] = derivative(calc_busy_pls, 0,
                             dx=1e-3 / b[0], n=i + 1, order=9)
    return [-busy[0], busy[1].real, -busy[2]]


def busy_calc_warm_up(l: float, f: list[float], busy_moments: list[float], num: int = 5):
    """
    Calculate the initial moments of continuous busy period for M/G/1 queue with warm-up
    By default, the first three are calculated.
    :param l: - input flow intensity
    :param f: - initial service time moments
    :param busy_moments: - initial moments of busy period
    :param num: - number of moments to calculate
    :return: busy_moments_warm_up

    """
    num = min(num, len(f))

    busy_moments_warm_up = []
    z = 1 + l * busy_moments[0]
    busy_moments_warm_up.append(f[0] * z)
    if num > 1:
        busy_moments_warm_up.append(f[0] * l * busy_moments[1] + f[1] * z * z)
    if num > 2:
        busy_moments_warm_up.append(f[0] * l * busy_moments[2] + 3 * f[1] *
                                    l * busy_moments[1] * z + f[2] * math.pow(z, 3))
    if num > 3:
        busy_moments_warm_up.append(f[0] * l * busy_moments[3] + f[1] * (3 * math.pow(l * busy_moments[1], 2) + 4 * l * busy_moments[2] * z)
                                    + 6 * f[2] * l * busy_moments[1] * z * z + f[3] * math.pow(z, 4))
    if num > 4:
        busy_moments_warm_up.append(f[0] * l * busy_moments[4] + f[1] * (5 * l * busy_moments[3] * z + 10 * l * l * busy_moments[1] * busy_moments[2]) +
                                    f[2] * (15 * math.pow(l * busy_moments[1], 2) * z + 10 * f[3] * l * busy_moments[1] * math.pow(z, 3) + f[
                                        4] * math.pow(z, 5)))

    return busy_moments_warm_up
