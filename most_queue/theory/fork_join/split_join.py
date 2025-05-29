"""
Numerical calculation of Fork-Join queuing systems
"""
import math

import numpy as np

from most_queue.rand_distribution import (
    ErlangDistribution,
    GammaDistribution,
    H2Distribution,
)
from most_queue.theory.fifo.mg1 import MG1Calculation
from most_queue.theory.utils.conv import conv_moments, get_self_conv_moments
from most_queue.theory.vacations.mg1_warm_calc import MG1WarmCalc


class SplitJoinCalc:
    """
    Numerical calculation of Split-Join queueuing systems

    In a fork-join queueing system, a job is forked into n sub-tasks
    when it arrives at a control node, and each sub-task is sent to a
    single node to be conquered. 

    A basic fork-join queue considers a job is done after all results
    of the job have been received at the join node

    Split-Join queue differs from a basic fork-join queueing system 
    in that it has blocking behavior. 

    New jobs are not allowed to enter the system, until current job has finished.

    """

    def __init__(self, l: float, n: int, b: list[float], approximation='gamma'):
        """
        :param l: arrival rate
        :param n: number of servers
        :param b: list of initial moments of service time
        :param approximation: str : type of approximation 
            for the initial moments of service time, can be 'gamma', 'h2' or 'erlang'
            default is 'gamma'
        """
        self.n = n
        self.b = b
        self.l = l

        self.b_max = None
        self.b_max_warm = None
        self.approximation = approximation

        if approximation not in ['gamma', 'h2', 'erlang']:
            raise ValueError(
                "Approximation must be one of 'gamma', 'h2' or 'erlang'.")

        self.a_big = [1.37793470540E-1, 7.29454549503E-1, 1.808342901740E0,
                      3.401433697855E0, 5.552496140064E0, 8.330152746764E0,
                      1.1843785837900E1, 1.6279257831378E1, 2.1996585811981E1,
                      2.9920697012274E1]
        self.g = [3.08441115765E-1, 4.01119929155E-1, 2.18068287612E-1,
                  6.20874560987E-2, 9.50151697518E-3, 7.53008388588E-4,
                  2.82592334960E-5, 4.24931398496E-7, 1.83956482398E-9,
                  9.91182721961E-13]

    def get_v(self) -> list[float]:
        """
        Calculate sojourn time initial moments for Split-Join queueing systems

        :return: list[float] : initial moments of sojourn time distribution 
        """

        # Calc Split-Join max of n channels service time distribution

        self.b_max = self.get_max_moments()

        # Further calculation as in a regular M/G/1 queueing system with
        # initial moments of the distribution maximum of the random variable
        mg1 = MG1Calculation(self.l, self.b_max)
        return mg1.get_v()

    def get_v_delta(self, b_delta: list[float] | float) -> list[float]:
        """
        Calculate sojourn time initial moments for Split-Join queueing systems with delta
        :param b_delta:  If delta is a list, it should contain the moments of
        time delay caused by reception and restoration operations for each part. 
        If delta is a float, delay is determistic and equal to delta.
        :return: list[float] : initial moments of sojourn time distribution 
        """

        self.b_max_warm = self.get_max_moments_delta(b_delta)
        self.b_max = self.get_max_moments()
        mg1_approx = 'gamma' if self.approximation == 'erlang' else self.approximation
        mg1_warm = MG1WarmCalc(self.l, self.b_max, self.b_max_warm, approximation=mg1_approx)
        return mg1_warm.get_v()

    def get_ro(self):
        """
        Calculate the utilization factor for Split-Join queueing systems
        """

        if self.b_max is None:
            self.get_v()

        return self.l * self.b_max[0]

    def get_max_moments(self):
        """
        Calculate the maximum value of lambda for a given number of channels and service rate.
        The maximum utilization is set to 0.8 by default.
        :param n: number of channels
        :param b: service rate in a channel
        :param num: number of output initial moments of the maximum SV, 
        by default one less than the number of initial moments of b
        :return: maximum value of lambda for a given number of channels and service rate.
        """

        if self.approximation == 'gamma':
            return self._calc_f_gamma()
        elif self.approximation == 'h2':
            return self._calc_f_h2()

        return self._calc_f_erlang()

    def _calc_f_h2(self):
        num = len(self.b)
        f = [0] * num
        params = H2Distribution.get_params(self.b)

        for j in range(10):
            p = self.g[j] * \
                self._dfr_h2_mult(
                    params, self.a_big[j]) * math.exp(self.a_big[j])
            f[0] += p
            for i in range(1, num):
                p = p * self.a_big[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= (i + 2)
        return f

    def _calc_f_gamma(self):
        num = len(self.b)
        f = [0] * num
        params = GammaDistribution.get_params(self.b)

        for j in range(10):
            p = self.g[j] * \
                self._dfr_gamma_mult(
                    params, self.a_big[j]) * math.exp(self.a_big[j])
            f[0] += p
            for i in range(1, num):
                p = p * self.a_big[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= (i + 2)
        return f

    def _calc_f_erlang(self):
        num = len(self.b)

        f = [0] * num

        params = ErlangDistribution.get_params(self.b)

        for j in range(10):
            p = self.g[j] * \
                self._dfr_erl_mult(
                    params, self.a_big[j]) * math.exp(self.a_big[j])
            f[0] += p
            for i in range(1, num):
                p = p * self.a_big[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= (i + 2)
        return f

    def get_max_moments_delta(self, delta=0):
        """
        Calculation of the initial moments of the maximum of a random variable with delay delta.
        :param n: number of identically distributed random variables
        :param b: initial moments of the random variable
        :param num: number of initial moments of the random variable
        :return: initial moments of the maximum of the random variable.
        """
        b = self.b

        num = len(self.b)

        f = [0] * num

        if delta:
            params = GammaDistribution.get_params(b)

            for j in range(10):
                p = self.g[j] * self._dfr_gamma_mult(params, self.a_big[j],
                                                delta) * math.exp(self.a_big[j])
                f[0] += p
                for i in range(1, num):
                    p = p * self.a_big[j]
                    f[i] += p

            for i in range(num - 1):
                f[i + 1] *= (i + 2)

        return f

    def _get_lambda(self, min_value, max_value):
        """
        Generate a random number between min and max.
        """
        l = np.random.randn()
        while l < min_value or l > max_value:
            l = np.random.randn()
        return l

    def _get_1ambda_max(self, ro_max=0.8):
        """
        Calculate the maximum value of lambda for a given number of channels and service rate.
        The maximum utilization is set to 0.8 by default.
        :param b: service rate in a channel
        :param n: number of channels
        :param ro_max: maximum utilization of the system
        :return: maximum value of lambda for a given number of channels and service rate.
        """
        b1_max = self.get_max_moments()[0]
        return ro_max / b1_max

    def _calc_error_percentage(self, real_val, est_val):
        """
        Calculate the error percentage between a real value and an estimated value.
        :param real_val: real value
        :param est_val: estimated value
        :return: error percentage between a real value and an estimated value.
        """
        max_val = max(real_val, est_val)
        return 100 * math.fabs(real_val - est_val) / max_val

    def _dfr_h2_mult(self, params, t, delta=None):
        """
        Calculation of the derivative of the multivariate H2 distribution function.
        params - parameters of the distribution (mu, alpha)
        t - time
        n - number of channels
        delta - optional parameter for multivariate distributions
        """
        res = 1.0
        if not delta:
            for i in range(self.n):
                res *= H2Distribution.get_cdf(params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= H2Distribution.get_cdf(params, t - i * delta)
        return 1.0 - res

    def _dfr_erl_mult(self, params, t, delta=None):
        """
        Calculation of the derivative of the multivariate Erlang distribution function.
        params - parameters of the distribution (mu, alpha)
        t - time
        n - number of channels
        delta - optional parameter for multivariate distributions
        """
        res = 1.0
        if not delta:
            for i in range(self.n):
                res *= ErlangDistribution.get_cdf(params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= ErlangDistribution.get_cdf(params, t - i * delta)
        return 1.0 - res

    def _dfr_gamma_mult(self, params, t, delta=None):
        """
        Calculation of the derivative of the multivariate Gamma distribution function.
        params - parameters of the distribution (mu, alpha)
        t - time
        n - number of channels
        delta - optional parameter for multivariate distributions
        """
        res = 1.0
        if not delta:
            for i in range(self.n):
                res *= GammaDistribution.get_cdf(params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= GammaDistribution.get_cdf(params, t - i * delta)
            else:
                b = GammaDistribution.calc_theory_moments(params)

                for i in range(self.n):
                    b_delta = get_self_conv_moments(delta, i)
                    b_summ = conv_moments(b, b_delta)
                    params_summ = GammaDistribution.get_params(b_summ)
                    res *= GammaDistribution.get_cdf(params_summ, t)

        return 1.0 - res
