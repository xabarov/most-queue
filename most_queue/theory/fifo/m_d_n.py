"""
Numerical calculation of an M/D/n system.
"""

import cmath
import math
import time

import numpy as np

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class MDn(BaseQueue):
    """
    Numerical calculation of an M/D/n system.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        Parameters:
        n - number of channels
        calc_params - parameters for calculation
        """

        super().__init__(n=n, calc_params=calc_params)

        self.p_num = self.calc_params.p_num
        self.e = self.calc_params.tolerance
        self.p = None

        self.l = None
        self.b = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rate of markovian arrival distribution.
        :param l: arrival rate
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: float):  # pylint: disable=arguments-differ
        """
        Set service time of deterministic service.
        """
        self.b = b
        self.is_servers_set = True

    def run(self) -> QueueResults:
        """
        Run calculation of queueing system.
        """
        start = time.process_time()
        self.p = self.p or self.calc_p()
        utilization = self.l * self.b / self.n

        return QueueResults(p=self.p, utilization=utilization, duration=time.process_time() - start)

    def calc_p(self) -> list[float]:
        """
        Calculate the probabilities of states.
        """

        self._check_if_servers_and_sources_set()

        self.p = [0.0] * self.calc_params.p_num

        p_up_to_n = self._calc_p_up_to_n()
        qs = self._calc_q()
        summ = 0
        for i, prob in enumerate(p_up_to_n):
            self.p[i] = prob
            summ += self.p[i]

        self.p[self.n] = self.p[0] / qs[0] - summ
        u = summ + self.p[self.n]

        is_negative = False
        for k in range(self.n + 1, self.p_num):
            if is_negative:
                break
            summ = 0
            for j in range(1, k - self.n):
                summ += qs[j] * self.p[k - j]
            value = (self.p[k - self.n] - u * qs[k - self.n] - summ) / qs[0]
            if value < 0:
                is_negative = True
            else:
                self.p[k] = value

        return self.p

    def _calc_p_up_to_n(self):

        zs = self._get_z()
        A = np.zeros((self.n, self.n), dtype=complex)
        B = np.zeros(self.n, dtype=complex)
        row_num = 0
        for z_value in zs:
            for j in range(self.n):
                right_z = np.power(z_value, self.n).real
                delta_z = np.power(z_value, j).real - right_z
                A[row_num, j] = delta_z
            row_num += 1
        for z_value in zs:
            for j in range(self.n):
                right_z = np.power(z_value, self.n).imag
                delta_z = np.power(z_value, j).imag - right_z
                A[row_num, j] = delta_z
            row_num += 1

        for j in range(self.n):
            A[self.n - 1, j] = self.n - j

        B[self.n - 1] = self.n - self.l * self.b
        p = np.linalg.lstsq(A, B, rcond=1e-8)
        p_real = []
        for i in range(len(p[0])):
            p_real.append(p[0][i].real)
        return p_real

    def _calc_q(self):

        q0 = math.exp(-self.b * self.l)
        qs = [0.0] * self.p_num
        qs[0] = q0
        for i in range(1, self.p_num):
            qs[i] = qs[i - 1] * (self.l * self.b) / i

        return qs

    def _get_z(self) -> list[float]:
        """
        Find the roots of z.
        """
        z_num = math.floor(self.n / 2)

        zs = [complex(0, 0) for _ in range(z_num)]

        for m in range(z_num):
            zs[m] = 0.5 * cmath.exp(2.0 * (m + 1) * cmath.pi * complex(0, 1) / self.n)
            z_old = zs[m]
            is_close = False
            while not is_close:
                left = 2 * (m + 1) * cmath.pi * complex(0, 1) / self.n
                right = self.l * self.b * (1.0 - z_old) / self.n
                z_new = cmath.exp(left - right)
                if math.fabs(z_new.real - z_old.real) < self.e:
                    is_close = True
                z_old = z_new
            zs[m] = z_old
        return zs
