"""
Abstract base class for all distributions.
"""
from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    Abstract base class for all distributions.
    """

    @abstractmethod
    def __init__(self, params, generator=None):
        """
        Initializes a new instance of the Distribution class.
        :param params: Parameters for the distribution.
        :param generator: Random number generator. If None, np.random is used.
        """

    @abstractmethod
    def generate(self) -> float:
        """
        Generates a random number according to the specified distribution.
        :return: float
        """

    @staticmethod
    @abstractmethod
    def generate_static(params, generator=None) -> float:
        """
        Generates a static random number according to the specified distribution and parameters.
        :param params: Parameters for the distribution.
        :param generator: Random number generator. If None, np.random is used.
        :return: float
        """

    @staticmethod
    @abstractmethod
    def calc_theory_moments(params, num: int) -> list[float]:
        """
        Calculates theoretical moments of the distribution up to the specified order.
        :param params: Parameters for the distribution.
        :param num: number of moments to calculate.
        :return: list[float]
        """

    @staticmethod
    @abstractmethod
    def get_params(moments: list[float]):
        """
        :param moments: list of theoretical moments.
        :return: Parameters for the distribution that correspond to the given moments.
        """

    @staticmethod
    @abstractmethod
    def get_params_by_mean_and_coev(f1: float, coev: float):
        """
        :param f1: mean of the distribution.
        :param coev: coefficient of variation (std_dev / mean).
        :return: Parameters for the distribution that correspond to the given mean and coefficient of variation.
        """
