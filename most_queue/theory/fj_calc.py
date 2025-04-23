"""
Numerical calculation of Fork-Join queuing systems
"""
import math

import numpy as np
import scipy.special as sp

from most_queue.general_utils.conv import get_moments, get_self_conv_moments
from most_queue.rand_distribution import ErlangDistribution, GammaDistribution, H2Distribution
from most_queue.theory.mg1_calc import MG1Calculation
from most_queue.theory.mg1_warm_calc import MG1WarmCalc


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

    def __init__(self, l: float, n: int, b: list[float]):
        """
        :param l: arrival rate
        :param n: number of servers
        :param b: list of initial moments of service time
        """
        self.n = n
        self.b = b
        self.l = l

        self.b_max = None
        self.b_max_warm = None

    def get_v(self) -> list[float]:
        """
        Calculate soujourn time initial moments for Split-Join queueing systems

        :return: list[float] : initial moments of soujourn time distribution 
        """

        # Calc Split-Join max of n channels service time distribution

        self.b_max = self.get_max_moments()

        # Further calculation as in a regular M/G/1 queueing system with
        # initial moments of the distribution maximum of the random variable
        mg1 = MG1Calculation(self.l, self.b_max)
        return mg1.get_v()

    def get_v_delta(self, b_delta: list[float] | float) -> list[float]:
        """
        Calculate soujourn time initial moments for Split-Join queueing systems with delta
        :param b_delta:  If delta is a list, it should contain the moments of
        time delay caused by reception and restoration operations for each part. 
        If delta is a float, delay is determistic and equal to delta.
        :return: list[float] : initial moments of soujourn time distribution 
        """

        self.b_max_warm = self.get_max_moments_delta(b_delta)
        self.b_max = self.get_max_moments()
        mg1_warm = MG1WarmCalc(self.l, self.b_max, self.b_max_warm)
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

        b = self.b
        num = len(b)

        f = [0] * num
        variance = b[1] - b[0] * b[0]
        coev = math.sqrt(variance) / b[0]
        a_big = [1.37793470540E-1, 7.29454549503E-1, 1.808342901740E0,
                 3.401433697855E0, 5.552496140064E0, 8.330152746764E0,
                 1.1843785837900E1, 1.6279257831378E1, 2.1996585811981E1,
                 2.9920697012274E1]
        g = [3.08441115765E-1, 4.01119929155E-1, 2.18068287612E-1,
             6.20874560987E-2, 9.50151697518E-3, 7.53008388588E-4,
             2.82592334960E-5, 4.24931398496E-7, 1.83956482398E-9,
             9.91182721961E-13]

        if len(b) >= 3:

            if coev < 1:
                params = ErlangDistribution.get_params(b)

                for j in range(10):
                    p = g[j] * \
                        self._dfr_erl_mult(
                            params, a_big[j]) * math.exp(a_big[j])
                    f[0] += p
                    for i in range(1, num):
                        p = p * a_big[j]
                        f[i] += p
            else:
                params = H2Distribution.get_params(b)

                for j in range(10):
                    p = g[j] * \
                        self._dfr_h2_mult(
                            params, a_big[j]) * math.exp(a_big[j])
                    f[0] += p
                    for i in range(1, num):
                        p = p * a_big[j]
                        f[i] += p
        else:
            params = GammaDistribution.get_mu_alpha(b)

            for j in range(10):
                p = g[j] * \
                    self._dfr_gamma_mult(params, a_big[j]) * math.exp(a_big[j])
                f[0] += p
                for i in range(1, num):
                    p = p * a_big[j]
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
        a_big = [1.37793470540E-1, 7.29454549503E-1, 1.808342901740E0,
                 3.401433697855E0, 5.552496140064E0, 8.330152746764E0,
                 1.1843785837900E1, 1.6279257831378E1, 2.1996585811981E1,
                 2.9920697012274E1]
        g = [3.08441115765E-1, 4.01119929155E-1, 2.18068287612E-1,
             6.20874560987E-2, 9.50151697518E-3, 7.53008388588E-4,
             2.82592334960E-5, 4.24931398496E-7, 1.83956482398E-9,
             9.91182721961E-13]

        if delta:
            params = GammaDistribution.get_mu_alpha(b)

            for j in range(10):
                p = g[j] * self._dfr_gamma_mult(params, a_big[j],
                                                delta) * math.exp(a_big[j])
                f[0] += p
                for i in range(1, num):
                    p = p * a_big[j]
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
                res *= GammaDistribution.get_cdf(*params, t)
        else:
            if not isinstance(delta, list):
                for i in range(self.n):
                    res *= GammaDistribution.get_cdf(*params, t - i * delta)
            else:
                b = GammaDistribution.calc_theory_moments(*params)

                for i in range(self.n):
                    b_delta = get_self_conv_moments(delta, i)
                    b_summ = get_moments(b, b_delta)
                    params_summ = GammaDistribution.get_mu_alpha(b_summ)
                    res *= GammaDistribution.get_cdf(*params_summ, t)

        return 1.0 - res


class ForkJoinMarkovianCalc:
    """
    Numerical calculation of Fork-Join queueuing systems

    In a fork-join queueing system, a job is forked into n sub-tasks
    when it arrives at a control node, and each sub-task is sent to a
    single node to be conquered. 

    A basic fork-join queue considers a job is done after all results
    of the job have been received at the join node

    The (n, k) fork-join queues, only require the job’s any k out of n sub-tasks to be finished,
    and thus have performance advantages in such scenarios. 

    There are mainly two versions of (n, k) fork-join queues: 
    The purging one removes all the remaining sub-tasks of a job from both sub-queues
    and service stations once it receives the job’s k the answer.
    As a contrast, the non-purging one keeps queuing and executing remaining sub-tasks

    """

    def __init__(self, l, mu, n, k=None):
        """
        :param l: Arrival rate
        :param mu: Service rate
        :param n: Number of servers
        :param k: Number of sub-tasks that need to be completed before the job is considered done (default is n)
        """
        self.l = l
        self.n = n
        self.mu = mu

        if k is None:
            k = n

        self.k = k
        self.ro = self.l / self.mu

    def get_v1_fj2(self):
        """
        Calculation of mean soujourn time for FJ system with n=2 channels
        :param mu: service rate
        :param l: arrival rate
        :return: mean soujourn time for FJ system with n=2 channels
        """

        assert self.n == 2, "n must be equal to 2"

        h2 = 1.5
        v1_m = 1 / (self.mu - self.l)

        return (h2 - self.ro / 8) * v1_m

    def get_v1_fj_varma(self):
        """
        Calculation of mean soujourn time for FJ system with n channels
        using approximation from:
        S. Varma and A. M. Makowski, “Interpolation approximations for
        symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265,
        1994

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :return: mean soujourn time for FJ system with n channels
        """

        Vn = self._get_v_big(self.n)
        Hn = self._get_h_big(self.n)
        v1 = (Hn + (Vn - Hn) * self.ro) / (self.mu - self.l)
        return v1

    def get_v1_fj_nelson_tantawi(self):
        """
        Calculation of mean soujourn time for FJ system with n channels
        using approximation from:
        R. D. Nelson and A. N. Tantawi, “Approximate analysis of fork/join
        synchronization in parallel queues.” IEEE Trans. Computers, vol. 37,
        no. 6, pp. 739–743, 1988.

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :return: mean soujourn time for FJ system with n channels
        """

        Hn = self._get_h_big(self.n)
        v1 = (Hn / 1.5) + (4.0 / 11) * (1.0 - Hn / 1.5) * self.ro
        v1 *= (12 - self.ro) / (8 * (self.mu - self.l))
        return v1

    def get_v1_fj_nelson_nk(self):
        """
        Calculation of mean soujourn time for FJ (n, k) system with n channels
        using approximation from:
        R. D. Nelson and A. N. Tantawi, “Approximate analysis of fork/join
        synchronization in parallel queues.” IEEE Trans. Computers, vol. 37,
        no. 6, pp. 739–743, 1988.

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :param k: number of sub-tasks that need to be completed before a job is considered done
                if n = k, then it's a basic fork-join queueing system.
        :return: mean soujourn time for FJ system with n channels
        """
        res = 0
        coeff = (12 - self.ro) / (88 * self.mu * (1 - self.ro))

        if self.k == 1:
            res += self.n / (self.mu * (1 - self.ro))
            summ = 0

            for i in range(2, self.n + 1):
                summ += self._get_w_big(self.n, 1, i) * (11 * self._get_h_big(i) + 4 * self.ro *
                                                         (self._get_h_big(2) - self._get_h_big(i))) / self._get_h_big(2)

            res += coeff * summ
        else:
            summ = 0
            for i in range(self.k, self.n + 1):
                summ += self._get_w_big(self.n, self.k, i) * (11 * self._get_h_big(i) + 4 * self.ro *
                                                              (self._get_h_big(2) - self._get_h_big(i))) / self._get_h_big(2)
            res = coeff * summ

        return res

    def get_v1_varma_nk(self):
        """
        Calculation of mean soujourn time for FJ (n, k) system with n channels
        using approximation from:
        S. Varma and A. M. Makowski, “Interpolation approximations for
        symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265,
        1994.

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :param k: number of sub-tasks that need to be completed before a job is considered done
                if n = k, then it's a basic fork-join queueing system.
        :return: mean soujourn time for FJ system with n channels
        """
        summ = 0

        for i in range(self.k, self.n + 1):
            Hn = self._get_h_big(i)
            Vn = self._get_v_big(i)
            delta_ro = (Vn - Hn)*self.ro

            summ += self._get_w_big(self.n, self.k, i) * \
                (Hn + delta_ro) / (self.mu - self.l)

        return summ

    def get_v1_fj_varki_merchant(self):
        """
        Calculation of mean soujourn time for FJ system with n channels
        using approximation from:
        E. Varki, “Response time analysis of parallel computer and storage
        systems.” IEEE Trans. Parallel Distrib. Syst., 2001.
        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :return: mean soujourn time for FJ system with n channels
        """
        Hn = self._get_h_big(self.n)
        summ = 0
        summ2 = 0
        for i in range(1, self.n + 1):
            summ += 1 / (i - self.ro)
            summ2 += 1 / (i * (i - self.ro))

        a = self.ro / (2 * (1 - self.ro))

        v1 = (1 / self.mu) * (Hn + a * (summ + (1 - 2 * self.ro) * summ2))

        return v1

    def _get_v_big(self, n):
        """
        Coefficient V[k] from paper
        S. Varma and A. M. Makowski, “Interpolation approximations for
        symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265,
        1994.
        """
        Vn = 0
        for r in range(1, n + 1):
            elem = sp.binom(n, r) * pow(-1, r - 1)
            summ2 = 0
            for m in range(1, r + 1):
                summ2 += sp.binom(r, m) * math.factorial(m - 1) / pow(r, m + 1)
            elem *= summ2
            Vn += elem
        return Vn

    def _get_a_big(self, n, k, i):
        """
        Coefficient A from paper
        Wang H. et al. Approximations and bounds for (n, k) fork-join queues: 
        a linear transformation approach //2018 18th IEEE/ACM International Symposium on Cluster,
        Cloud and Grid Computing (CCGRID). – IEEE, 2018. – С. 422-431.
        """
        if i == k:
            return 1
        else:
            summ = 0
            for j in range(1, i - k + 1):
                summ += sp.binom(n - i + j, j) * self._get_a_big(n, k, i - j)

            return (-1) * summ

    def _get_w_big(self, n, k, i):
        """
        Coefficient W from paper
        Wang H. et al. Approximations and bounds for (n, k) fork-join queues: 
        a linear transformation approach //2018 18th IEEE/ACM International Symposium on Cluster,
        Cloud and Grid Computing (CCGRID). – IEEE, 2018. – С. 422-431.
        """
        summ = 0
        for j in range(k, i + 1):
            summ += sp.binom(n, j) * self._get_a_big(n, j, i)

        return summ

    def _get_h_big(self, n):
        """
        Coefficient H[k] from paper
        S. Varma and A. M. Makowski, “Interpolation approximations for
        symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265,
        1994.
        """
        summ = 0
        for i in range(1, n + 1):
            summ += 1 / i
        return summ

    def test_h_v_big(self):
        """
        Testing the calculation of Hk and Vk
        """
        print("| K | Hk | Vk |\n")
        print("|---|----|----|\n")
        for k in (x for x in range(2, 21)):
            hk = self._get_h_big(k)
            vk = self._get_v_big(k)
            print(f"| {k} | {hk} | {vk} |\n")
