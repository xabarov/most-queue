"""
Utils for distribution from random_distributions

"""

import math

import yaml
from colorama import Fore, Style

from most_queue.rand_distribution import (
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
from most_queue.sim.utils.exceptions import QsSourseSettingException


def print_supported_distributions():
    """Prints supported distributions loaded from a YAML file."""
    print(f"{Fore.GREEN}Supported distributions:{Style.RESET_ALL}")
    separator = f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}"

    dist_path = "most_queue/sim/utils/distributions.yaml"
    try:
        with open(dist_path, "r", encoding="utf-8") as file:
            distributions = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"{Fore.RED}Error: distributions.yaml not found.{Style.RESET_ALL}")
        return
    except yaml.YAMLError as e:
        print(f"{Fore.RED}Error parsing distributions.yaml: {e}{Style.RESET_ALL}")
        return

    header = (
        f"{separator}\n"
        f"{Fore.CYAN}Distribution{'-':20}"
        f"{Fore.GREEN}Kendall notation{'-':15}"
        f"{Fore.RED}Parameters{Style.RESET_ALL}\n"
        f"{separator}"
    )
    print(header)

    for dist in distributions:
        name = dist.get("name", "N/A")
        kendall = dist.get("kendall_notation", "N/A")
        params = dist.get("params", "N/A")
        print(f"{Fore.MAGENTA}{name:<38}" f"{Fore.GREEN}{kendall:<19}" f"{Fore.RED}{params}{Style.RESET_ALL}")
    print(separator)


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
        raise QsSourseSettingException(
            "Incorrect distribution type specified. Options \
             М, Н, Е, С, Pa, Uniform, Norm, D"
        )

    return dist


def calc_qs_load(
    source_kendall_notation: str,
    source_params,
    server_kendall_notation: str,
    server_params,
    n,
) -> float:
    """Calculates the utilization (load factor) of the QS

    Args:
        source_kendall_notation: str,  Kendall notation of source
        server_kendall_notation: str, Kendall notation of source
        n (int): number of QS channels

    Returns:
        float: utilization (load factor) of the QS
    """

    l = 0
    if source_kendall_notation == "M":
        l = source_params
    elif source_kendall_notation == "D":
        l = 1.00 / source_params
    elif source_kendall_notation == "Uniform":
        l = 1.00 / source_params.mean
    elif source_kendall_notation == "H":
        y1 = source_params.p1
        y2 = 1.0 - y1
        mu1 = source_params.mu1
        mu2 = source_params.mu2

        f1 = y1 / mu1 + y2 / mu2
        l = 1.0 / f1

    elif source_kendall_notation == "E":
        r = source_params.r
        mu = source_params.mu
        l = mu / r

    elif source_kendall_notation == "Gamma":
        mu = source_params.mu
        alpha = source_params.alpha
        l = mu / alpha

    elif source_kendall_notation == "C":
        y1 = source_params.p1
        y2 = 1.0 - y1
        mu1 = source_params.mu1
        mu2 = source_params.mu2

        f1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
        l = 1.0 / f1
    elif source_kendall_notation == "Pa":
        if source_params[0] < 1:
            return None

        a = source_params.alpha
        k = source_params.K
        f1 = a * k / (a - 1)
        l = 1.0 / f1

    b1 = 0
    if server_kendall_notation == "M":
        mu = server_params
        b1 = 1.0 / mu
    elif server_kendall_notation == "D":
        b1 = server_params
    elif server_kendall_notation == "Uniform":
        b1 = server_params.mean

    elif server_kendall_notation == "H":
        y1 = server_params.p1
        y2 = 1.0 - y1
        mu1 = server_params.mu1
        mu2 = server_params.mu2

        b1 = y1 / mu1 + y2 / mu2

    elif server_kendall_notation == "Gamma":
        mu = server_params.mu
        alpha = server_params.alpha
        b1 = alpha / mu

    elif server_kendall_notation == "E":
        r = server_params.r
        mu = server_params.mu
        b1 = r / mu

    elif server_kendall_notation == "C":
        y1 = server_params.p1
        y2 = 1.0 - y1
        mu1 = server_params.mu1
        mu2 = server_params.mu2

        b1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
    elif server_kendall_notation == "Pa":
        if server_params[0] < 1:
            return math.inf

        a = server_params.alpha
        k = server_params.K
        b1 = a * k / (a - 1)

    return l * b1 / n


if __name__ == "__main__":
    # Example usage
    print_supported_distributions()
