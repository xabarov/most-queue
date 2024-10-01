"""
Utils for distribution from random_distributions 

"""
import math

from most_queue.rand_distribution import (
    Cox_dist,
    Det_dist,
    Erlang_dist,
    Exp_dist,
    Gamma,
    H2_dist,
    Normal_dist,
    Pareto_dist,
    Uniform_dist,
)
from most_queue.sim.utils.exceptions import QsSourseSettingException


def create_distribution(params, kendall_notation: str, generator):
    """ Creates distribution from random_distributions 

    --------------------------------------------------------------------
    Distribution                    kendall_notation    params
    --------------------------------------------------------------------
    Exponential                           'М'             [mu]
    Hyperexponential of the 2nd order     'Н'         [y1, mu1, mu2]
    Erlang                                'E'           [r, mu]
    Cox 2nd order                         'C'         [y1, mu1, mu2]
    Pareto                                'Pa'         [alpha, K]
    Deterministic                         'D'         [b]
    Uniform                            'Uniform'     [mean, half_interval]
    Gaussian                             'Norm'    [mean, standard_deviation]

    Args:
        params (_type_): params of distribution. 
                         For "M": one single value "mu". 
                         For "H": [y1, mu1, mu2]
        kendall_notation (str): like "M", "H", "E"
        generator (_type_): random numbers generator, for ex np.random.default_rng()

    Raises:
        QsSourseSettingException: Incorrect distribution type specified

    Returns:
        _type_: distribution from random_distributions 
    """
    dist = None
    if kendall_notation == "M":
        dist = Exp_dist(params, generator=generator)
    elif kendall_notation == "H":
        dist = H2_dist(params, generator=generator)
    elif kendall_notation == "E":
        dist = Erlang_dist(params, generator=generator)
    elif kendall_notation == "Gamma":
        dist = Gamma(params, generator=generator)
    elif kendall_notation == "C":
        dist = Cox_dist(params, generator=generator)
    elif kendall_notation == "Pa":
        dist = Pareto_dist(params, generator=generator)
    elif kendall_notation == "Uniform":
        dist = Uniform_dist(params, generator=generator)
    elif kendall_notation == "Norm":
        dist = Normal_dist(params, generator=generator)
    elif kendall_notation == "D":
        dist = Det_dist(params)
    else:
        raise QsSourseSettingException(
            "Incorrect distribution type specified. Options \
             М, Н, Е, С, Pa, Uniform, Norm, D")

    return dist


def calc_qs_load(source_types: str, source_params,
                 server_types: str, server_params, n) -> float:
    """Calculates the utilization (load factor) of the QS

    Args:
        source_types: str,  Kendall notation of source 
        server_types: str, Kendall notation of source 
        n (int): number of QS channels

    Returns:
        float: utilization (load factor) of the QS
    """

    l = 0
    if source_types == "M":
        l = source_params
    elif source_types == "D":
        l = 1.00 / source_params
    elif source_types == "Uniform":
        l = 1.00 / source_params[0]
    elif source_types == "H":
        y1 = source_params[0]
        y2 = 1.0 - source_params[0]
        mu1 = source_params[1]
        mu2 = source_params[2]

        f1 = y1 / mu1 + y2 / mu2
        l = 1.0 / f1

    elif source_types == "E":
        r = source_params[0]
        mu = source_params[1]
        l = mu / r

    elif source_types == "Gamma":
        mu = source_params[0]
        alpha = source_params[1]
        l = mu / alpha

    elif source_types == "C":
        y1 = source_params[0]
        y2 = 1.0 - source_params[0]
        mu1 = source_params[1]
        mu2 = source_params[2]

        f1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
        l = 1.0 / f1
    elif source_types == "Pa":
        if source_params[0] < 1:
            return None
        else:
            a = source_params[0]
            k = source_params[1]
            f1 = a * k / (a - 1)
            l = 1.0 / f1

    b1 = 0
    if server_types == "M":
        mu = server_params
        b1 = 1.0 / mu
    elif server_types == "D":
        b1 = server_params
    elif server_types == "Uniform":
        b1 = server_params[0]

    elif server_types == "H":
        y1 = server_params[0]
        y2 = 1.0 - server_params[0]
        mu1 = server_params[1]
        mu2 = server_params[2]

        b1 = y1 / mu1 + y2 / mu2

    elif server_types == "Gamma":
        mu = server_params[0]
        alpha = server_params[1]
        b1 = alpha / mu

    elif server_types == "E":
        r = server_params[0]
        mu = server_params[1]
        b1 = r / mu

    elif server_types == "C":
        y1 = server_params[0]
        y2 = 1.0 - server_params[0]
        mu1 = server_params[1]
        mu2 = server_params[2]

        b1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
    elif server_types == "Pa":
        if server_params[0] < 1:
            return math.inf
        else:
            a = server_params[0]
            k = server_params[1]
            b1 = a * k / (a - 1)

    return l * b1 / n
