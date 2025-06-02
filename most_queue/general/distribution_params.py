"""
Distribution parameters classes.
"""
from dataclasses import dataclass


@dataclass
class H2Params:
    """
    H2 distribution parameters.
    """
    mu1: float | complex  # Mean of the first phase
    mu2: float | complex  # Mean of the second phase
    p1: float | complex  # Probability of being in the first phase


@dataclass
class Cox2Params:
    """
    Coxian second order distribution parameters.
    """
    mu1: float | complex  # Mean of the first phase
    mu2: float | complex  # Mean of the second phase
    # probability for transition between phase 1 and phase 2.
    p1: float | complex


@dataclass
class GammaParams:
    """
    Gamma distribution parameters.
    """
    mu: float  # 1/theta
    alpha: float
    g: list[float] | None = None


@dataclass
class WeibullParams:
    """
     Weibull distribution parameters.
    """
    k: float  # Shape parameter
    W: float  # Scale parameter


@dataclass
class GaussianParams:
    """
    Normal distribution parameters.
    """
    mean: float  # Mean of the distribution
    std_dev: float  # Standard deviation of the distribution


@dataclass
class UniformParams:
    """
    Uniform  distribution parameters.
    """
    mean: float  # Mean of the distribution
    half_interval: float  # Half interval of the distribution


@dataclass
class ParetoParams:
    """
    Pareto  distribution parameters.
    """
    alpha: float  # alpha of the distribution
    K: float  # K of the distribution (or x_m, minimum value of x)


@dataclass
class ErlangParams:
    """
    Erlang distribution parameters.
    """
    r: int
    mu: float
