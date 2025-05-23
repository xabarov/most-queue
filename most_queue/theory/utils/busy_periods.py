import math

from scipy.misc import derivative

from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


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
        numerator = b[3] * math.pow(z, 4)
        numerator += 6 * b[2] * l * busy_moments[1] * z * z
        numerator += b[1] * (3 * math.pow(l * busy_moments[1],
                             2) + 4 * l * busy_moments[2] * z)
        busy_moments.append(numerator / (1 - ro))
    if num > 4:
        numerator = b[4] * math.pow(z, 5)
        numerator += 10 * b[3] * l * busy_moments[1] * math.pow(z, 3)
        numerator += b[2] * (15 * math.pow(l * busy_moments[1], 2)
                             * z + 10 * l * busy_moments[2]*z*z)
        numerator += b[1] * (5 * l * busy_moments[3] * z +
                             10 * l * l * busy_moments[1] * busy_moments[2])
        busy_moments.append(numerator / (1 - ro))

    return busy_moments


def calc_busy_pls(b_lst_function, b_params, l: float, s: float, tolerance=1e-12):
    """
    Calculate the busy period Laplace-Stieltjes transform.
     Parameters
     ----------
     b_lst_function : callable
         The function to use for the Laplace-Stieltjes transform.
     b_params : list[float]
         The parameters for service time distribution.
     l : float
         Arrival rate.
     s : float
         The value at which to evaluate the Laplace-Stieltjes transform.

     Returns
     -------
         The value of the busy period  Laplace-Stieltjes transform at the given value of s.
    """
    y = 0
    while True:
        y_new = b_lst_function(b_params, s + l-l*y)
        if abs(y_new - y) < tolerance:
            return y_new
        y = y_new


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

    def _calc_busy_pls(s):
        y = 0
        while True:
            y_new = lst_function(params, s + l-l*y)
            if abs(y_new - y) < 1e-6:
                return y_new
            y = y_new
    busy = [0, 0, 0]
    for i in range(3):
        busy[i] = derivative(_calc_busy_pls, 0,
                             dx=1e-3 / b[0], n=i + 1, order=9)
    return [-busy[0], busy[1].real, -busy[2]]


def busy_calc_warm_up(l: float, f: list[float], b_busy: list[float], num: int = 5):
    """
    Calculate the initial moments of continuous busy period for M/G/1 queue with warm-up
    By default, the first three are calculated.
    :param l: - input flow intensity
    :param f: - initial service time moments
    :param b_busy: - initial moments of busy period
    :param num: - number of moments to calculate
    :return: - initial moments of continuous busy period
    """
    num = min(num, len(f))

    gs = []
    z = 1 + l * b_busy[0]
    fzs = [f[i] * z ** (i+1) for i in range(num)]
    g1 = fzs[0]
    fl = f[0] * l
    gs.append(g1)
    if num > 1:
        g2 = fl * b_busy[1] + fzs[1]
        gs.append(g2)
    if num > 2:
        g3 = fl * b_busy[2]
        g3 += 3 * f[1] * l * b_busy[1] * z + fzs[2]
        gs.append(g3)
    if num > 3:
        g4 = fl * b_busy[3]
        g4 += f[1] * (3 * math.pow(l * b_busy[1], 2) + 4 * l * b_busy[2] * z)
        g4 += 6 * f[2] * l * b_busy[1] * z * z + fzs[3]
        gs.append(g4)
    if num > 4:
        g5 = fl * b_busy[4]
        g5 += f[1] * (5 * l * b_busy[3] * z + 10 *
                      l * l * b_busy[1] * b_busy[2])
        g5 += f[2] * (15 * math.pow(l * b_busy[1], 2) *
                      z + 10 * l * b_busy[2] * math.pow(z, 2))
        g5 += 10 * f[3] * l * b_busy[1] * math.pow(z, 3) + fzs[4]
        gs.append(g5)

    return gs
