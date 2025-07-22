"""
Utils for distribution from random_distributions

"""

from most_queue.random.distributions import (
    CoxDistribution,
    DeterministicDistribution,
    ErlangDistribution,
    ExpDistribution,
    GammaDistribution,
    H2Distribution,
    NormalDistribution,
    ParetoDistribution,
    UniformDistribution,
)


def create_distribution(params, kendall_notation: str, generator):
    """Creates distribution from random_distributions

    --------------------------------------------------------------------
    Distribution                    kendall_notation    params
    --------------------------------------------------------------------
    Exponential                           'М'             [mu]
    Hyperexponential of the 2nd order     'Н'         H2Params dataclass
    Erlang                                'E'           [r, mu]
    Cox 2nd order                         'C'         Cox2Params dataclass
    Pareto                                'Pa'         [alpha, K]
    Deterministic                         'D'         [b]
    Uniform                            'Uniform'     [mean, half_interval]
    Gaussian                             'Norm'    [mean, standard_deviation]

    Args:
        params (_type_): params of distribution.
                         For "M": one single value "mu".
                         For "H": H2Params
        kendall_notation (str): like "M", "H", "E"
        generator (_type_): random numbers generator, for ex np.random.default_rng()

    Raises:
        QsSourseSettingException: Incorrect distribution type specified

    Returns:
        _type_: distribution from random_distributions
    """
    dist = None
    if kendall_notation == "M":
        dist = ExpDistribution(params, generator=generator)
    elif kendall_notation == "H":
        dist = H2Distribution(params, generator=generator)
    elif kendall_notation == "E":
        dist = ErlangDistribution(params, generator=generator)
    elif kendall_notation == "Gamma":
        dist = GammaDistribution(params, generator=generator)
    elif kendall_notation == "C":
        dist = CoxDistribution(params, generator=generator)
    elif kendall_notation == "Pa":
        dist = ParetoDistribution(params, generator=generator)
    elif kendall_notation == "Uniform":
        dist = UniformDistribution(params, generator=generator)
    elif kendall_notation == "Norm":
        dist = NormalDistribution(params, generator=generator)
    elif kendall_notation == "D":
        dist = DeterministicDistribution(params)
    else:
        raise ValueError(
            "Incorrect distribution type specified. See most_queue.random.distributions.print_supported_distributions() for list of supported distributions"
        )

    return dist
