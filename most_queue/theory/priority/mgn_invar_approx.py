"""
Approximation of the initial moments of waiting and sojourn times 
for a multi-channel queue with priorities
and general service times.
The approximation is based on method of invariant moments for M/G/n queues.
"""
import math

from most_queue.theory.fifo.mg1 import MG1Calculation
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemtiveCalculation
from most_queue.theory.priority.preemptive.mg1 import MG1PreemtiveCalculation
from most_queue.theory.utils.conv import conv_moments


class MGnInvarApproximation:
    """
    Approximation of the initial moments of waiting and sojourn times 
    for a multi-channel queue with priorities and general service times.
      The approximation is based on method of invariant moments for M/G/n queues.
    """

    def __init__(self, l: list[float], b: list[list[float]], n: int):
        """
        :param l: list of input intensities l[k], k - class number
        :param b: b[k][j] - initial moments of service time, j - moment number
        :param n: number of channels
        """
        self.l = l
        self.b = b
        self.n = n

    def get_w(self, priority='NP', N: int = 150, num=3) -> list[list[float]]:
        """
        Approximation of the initial moments of waiting time 
        for a multi-channel queue with priorities
        based on the invariant relation M*|G*|n = M*|G*|1 * (M|G|n / M|G|1)

        :param priority:  type of priority - "PR" or "NP", 
        default is "NP" (preemptive, non-preemptive)
        :param N: number of levels for the Takahashi-Takagi method, also the number of probabilities
        calculated for M/G/1
        :return: w[k][j] - initial moments of waiting time for all classes
        """
        w = []
        k_num = len(self.l)
        j_num = len(self.b[0])

        for k in range(k_num):
            w.append([0.0] * num)

        # M*/G*/1 PRTY calculation:
        b1 = [] * k_num
        for k in range(k_num):
            b1.append([0.0] * j_num)

        for k in range(k_num):
            for j in range(j_num):
                b1[k][j] = self.b[k][j] / math.pow(self.n, j + 1)

        if priority == 'NP':
            calc_np1 = MG1NonPreemtiveCalculation(self.l, b1)
            w1_prty = calc_np1.get_w()
        elif priority == 'PR':
            calc_pr1 = MG1PreemtiveCalculation(self.l, b1)
            pr_prty_calc = calc_pr1.calc_all(num=num)
            w1_prty = pr_prty_calc['w_with_pr']
        else:
            return 'Wrong PRTY type. Should be "PR" or "NP"'

        # M/G/1 calculation:

        b_sr = [0.0] * j_num
        l_sum = 0
        for j in range(j_num):
            for k in range(k_num):
                b_sr[j] += b1[k][j]
            b_sr[j] /= k_num
        for k in range(k_num):
            l_sum += self.l[k]

        mg1_num = MG1Calculation(l_sum, b_sr)
        p1 = mg1_num.get_p(N)
        q1 = 0
        for i in range(1, N):
            q1 += (i - 1) * p1[i]

        # M/G/n calculation:

        b_sr = [0.0] * j_num
        l_sum = 0
        for j in range(j_num):
            for k in range(k_num):
                b_sr[j] += self.b[k][j]
            b_sr[j] /= k_num
        for k in range(k_num):
            l_sum += self.l[k]

        tt_n = MGnCalc(self.n, l_sum, b_sr)
        tt_n.run()
        p_n = tt_n.get_p()
        qn = 0
        for i in range(self.n + 1, N):
            qn += (i - self.n) * p_n[i]

        for k in range(k_num):
            for j in range(num):
                w[k][j] = w1_prty[k][j] * qn / q1

        return w

    def get_v(self, priority='NP', num=3) -> list[list[float]]:
        """
        Approximation of the initial moments of sojourn time 
        for a multi-channel queue with priorities
        based on the invariant relation M*|G*|n = M*|G*|1 * (M|G|n / M|G|1)

        :param priority:  type of priority - "PR" or "NP",
          default is "NP" (preemptive, non-preemptive)
        :param N: number of levels for the Takahashi-Takagi method, also the number of probabilities
        calculated for M/G/1
        :return: v[k][j] - initial moments of sojourn time for all classes
        """
        w = self.get_w(priority, num=num)
        v = []
        k = len(self.l)
        v = [conv_moments(w[i], self.b[i], num=num) for i in range(k)]

        return v
