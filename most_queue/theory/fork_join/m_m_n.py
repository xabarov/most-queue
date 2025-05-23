"""
Numerical calculation of Fork-Join queuing systems
"""
import math

import scipy.special as sp


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
        Calculation of mean sojourn time for FJ system with n=2 channels
        :param mu: service rate
        :param l: arrival rate
        :return: mean sojourn time for FJ system with n=2 channels
        """

        assert self.n == 2, "n must be equal to 2"

        h2 = 1.5
        v1_m = 1 / (self.mu - self.l)

        return (h2 - self.ro / 8) * v1_m

    def get_v1_fj_varma(self):
        """
        Calculation of mean sojourn time for FJ system with n channels
        using approximation from:
        S. Varma and A. M. Makowski, “Interpolation approximations for
        symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265,
        1994

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :return: mean sojourn time for FJ system with n channels
        """

        Vn = self._get_v_big(self.n)
        Hn = self._get_h_big(self.n)
        v1 = (Hn + (Vn - Hn) * self.ro) / (self.mu - self.l)
        return v1

    def get_v1_fj_nelson_tantawi(self):
        """
        Calculation of mean sojourn time for FJ system with n channels
        using approximation from:
        R. D. Nelson and A. N. Tantawi, “Approximate analysis of fork/join
        synchronization in parallel queues.” IEEE Trans. Computers, vol. 37,
        no. 6, pp. 739–743, 1988.

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :return: mean sojourn time for FJ system with n channels
        """

        Hn = self._get_h_big(self.n)
        v1 = (Hn / 1.5) + (4.0 / 11) * (1.0 - Hn / 1.5) * self.ro
        v1 *= (12 - self.ro) / (8 * (self.mu - self.l))
        return v1

    def get_v1_fj_nelson_nk(self):
        """
        Calculation of mean sojourn time for FJ (n, k) system with n channels
        using approximation from:
        R. D. Nelson and A. N. Tantawi, “Approximate analysis of fork/join
        synchronization in parallel queues.” IEEE Trans. Computers, vol. 37,
        no. 6, pp. 739–743, 1988.

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :param k: number of sub-tasks that need to be completed before a job is considered done
                if n = k, then it's a basic fork-join queueing system.
        :return: mean sojourn time for FJ system with n channels
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
        Calculation of mean sojourn time for FJ (n, k) system with n channels
        using approximation from:
        S. Varma and A. M. Makowski, “Interpolation approximations for
        symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265,
        1994.

        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :param k: number of sub-tasks that need to be completed before a job is considered done
                if n = k, then it's a basic fork-join queueing system.
        :return: mean sojourn time for FJ system with n channels
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
        Calculation of mean sojourn time for FJ system with n channels
        using approximation from:
        E. Varki, “Response time analysis of parallel computer and storage
        systems.” IEEE Trans. Parallel Distrib. Syst., 2001.
        :param mu: service rate
        :param l: arrival rate
        :param n: number of channels
        :return: mean sojourn time for FJ system with n channels
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
