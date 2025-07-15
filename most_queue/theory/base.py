"""
Base class for queueing systems.
"""

from most_queue.theory.calc_params import CalcParams


class BaseQueue:
    """
    Base class for queueing systems.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None, buffer: int | None = None):
        """
        Initializes the base queue class
        :param n: number of channels
        :param calc_params: calculation parameters
        :param buffer: buffer size
        """

        self.n = n
        self.calc_params = calc_params if calc_params else CalcParams()
        self.buffer = buffer

    def get_p(self) -> list[float]:
        """
        Returns the probability distribution of the number of customers in the system.
        """

    def get_w(self) -> list[float]:
        """
        Returns the waiting time initial moments
        """

    def get_v(self) -> list[float]:
        """
        Returns the sojourn time initial moments
        """
