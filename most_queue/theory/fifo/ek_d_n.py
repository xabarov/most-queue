"""
Numerical calculation of a multi-channel system Ek/D/n
"""

import cmath
import math

import numpy as np

from most_queue.constants import DEFAULT_TOLERANCE, LSTSQ_RCOND
from most_queue.random.distributions import ErlangParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams


class EkDn(BaseQueue):
    """
    Numerical calculation of a multi-channel system Ek/D/n
    with deterministic service
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        Initializes the EkDn class.
        :param erlang_params: parameters of the Erlang distribution.
        :param b: service rate of the server, constant value.
        :param n: number of channels.
        :param calc_params: calculation parameters.
        """

        super().__init__(n=n, calc_params=calc_params, buffer=None)

        self.e = self.calc_params.tolerance
        self.p_num = self.calc_params.p_num

        self.p = [0.0] * self.p_num
        self.q_ = [0.0] * self.p_num
        self.z_ = 0

        self.x = [0.0] * (2 * self.p_num)

        self.l = None
        self.k = None
        self.b = None

    def set_sources(self, erlang_params: ErlangParams):  # pylint: disable=arguments-differ
        """
        Set sources of the system.
        :param erlang_params: parameters of the arrival process.
        """
        self.l = erlang_params.mu
        self.k = erlang_params.r
        self.is_sources_set = True

    def set_servers(self, b: float):  # pylint: disable=arguments-differ
        """
        Set service rate of the server.
        :param b: service rate of the server, constant time.
        """
        self.b = b
        self.is_servers_set = True

    def run(self) -> QueueResults:
        """
        Run calculation.

        Returns:
            QueueResults with calculated values.
        """
        start = self._measure_time()

        utilization = self.l * self.b / (self.k * self.n)
        p = self.get_p()
        result = QueueResults(p=p, utilization=utilization)
        self._set_duration(result, start)
        return result

    def get_p(self):
        """
        Calc probabilities of system states
        """
        w = self._calc_x()

        is_zero = False
        for j in range(len(self.x)):
            if is_zero:
                break
            summ = 0
            for m in range(j * self.k, (j + 1) * self.k):
                summ += w[m]
            if math.fabs(summ) < DEFAULT_TOLERANCE or summ > 1:
                is_zero = True
            else:
                self.p[j] = summ

        return self.p

    def _calc_x(self):

        self._check_if_servers_and_sources_set()

        self._calc_q()
        w_up_to_nk = self._calc_w_up_to_nk()
        self.x[: len(w_up_to_nk)] = w_up_to_nk[:]
        for i in range(self.k):
            summ1 = 0
            for m in range(i + 1):
                q = self.q_[i - m]
                summ2 = 0
                for j in range(self.n):
                    summ2 += self.x[j * self.k + m]
                summ1 += q * summ2
            summ3 = 0
            for m in range(i):
                summ3 += self.q_[i - m] * self.x[self.n * self.k + m]
            self.x[self.n * self.k + i] = (self.x[i] - summ1 - summ3) / self.q_[0]

        is_negative = False

        for i in range(self.k, self.p_num):
            if is_negative:
                break
            summ1 = 0
            for m in range(self.k):
                summ1 += self._get_big_w(self.n, m) * self.q_[i - m]
            summ2 = 0
            for m in range(1, i - self.k + 1):
                summ2 += self.q_[m] * self.x[self.n * self.k + i - m]
            value = (self.x[i] - summ1 - summ2) / self.q_[0]
            if value < 0:
                is_negative = True
            else:
                self.x[self.n * self.k + i] = value

        return self.x

    def _get_big_w(self, n, i):
        summ = 0
        for j in range(n + 1):
            summ += self.x[j * self.k + i]
        return summ

    def _calc_w_up_to_nk(self):
        self._get_z()
        A = np.zeros((self.n * self.k, self.n * self.k), dtype=complex)
        B = np.zeros(self.n * self.k, dtype=complex)
        row_num = 0
        for z_val in self.z_:
            for j in range(self.n):
                for i in range(self.k):
                    right_z = np.power(z_val, self.n * self.k + i).real
                    delta_z = np.power(z_val, j * self.k + i).real - right_z
                    A[row_num, j * self.k + i] = delta_z
            row_num += 1
        for z_val in self.z_:
            for j in range(self.n):
                for i in range(self.k):
                    right_z = np.power(z_val, self.n * self.k + i).imag
                    delta_z = np.power(z_val, j * self.k + i).imag - right_z
                    A[row_num, j * self.k + i] = delta_z
            row_num += 1

        for j in range(self.n):
            coef = self.n - j
            for i in range(j * self.k, (j + 1) * (self.k)):
                A[self.n * self.k - 1, i] = coef

        B[self.n * self.k - 1] = self.n - self.l * self.b / self.k
        w = np.linalg.lstsq(A, B, rcond=LSTSQ_RCOND)
        w_real = []
        for i in range(len(w[0])):
            w_real.append(w[0][i].real)
        return w_real

    def _calc_q(self):
        q0 = math.exp(-self.b * self.l)
        self.q_ = [0.0] * self.p_num
        self.q_[0] = q0
        for i in range(1, self.p_num):
            self.q_[i] = self.q_[i - 1] * (self.l * self.b) / i

    def _get_z(self):
        z = []
        z_num = math.floor((self.n * self.k) / 2)

        z = [complex(0, 0)] * z_num

        for m in range(z_num):
            z[m] = 0.5 * cmath.exp(2.0 * (m + 1) * cmath.pi * complex(0, 1) / (self.n * self.k))
            z_old = z[m]
            is_close = False
            while not is_close:
                left = 2 * (m + 1) * cmath.pi * complex(0, 1) / (self.n * self.k)
                right = self.l * self.b * (1.0 - z_old) / (self.n * self.k)
                z_new = cmath.exp(left - right)
                if math.fabs(z_new.real - z_old.real) < self.e:
                    is_close = True
                z_old = z_new
            z[m] = z_old
        self.z_ = z
