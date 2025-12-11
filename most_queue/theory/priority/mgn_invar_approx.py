"""
Approximation of the raw moments of waiting and sojourn times
for a multi-channel queue with priorities
and general service times.
The approximation is based on method of invariant moments for M/G/n queues.
"""

import math
import time

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptiveCalc
from most_queue.theory.priority.preemptive.mg1 import MG1PreemptiveCalc
from most_queue.theory.utils.conv import conv_moments


class MGnInvarApproximation(BaseQueue):
    """
    Approximation of the raw moments of waiting and sojourn times
    for a multi-channel queue with priorities and general service times.
      The approximation is based on method of invariant moments for M/G/n queues.
    """

    def __init__(self, n: int, priority="NP", calc_params: TakahashiTakamiParams | None = None):
        """
        Initialize the MGnInvarApproximation class.
        :param n: number of channels.
        :param priority: type of priority system. Default is "NP" (non-preemptive).
                         can be "NP" or "PR" (preemptive).
        :param calc_params: calculation parameters.
        """

        super().__init__(n=n, calc_params=calc_params)
        self.l = None
        self.calc_params = calc_params or TakahashiTakamiParams()
        self.b = None
        self.priority = priority

    def set_sources(self, l: list[float]):  # pylint: disable=arguments-differ
        """
        Set arrival rates for each class.
        :param l: list of arrival rates.
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[list[float]]):  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution for each class.
        :param b: list of lists where each sublist contains raw moments of service time.
        """
        self.b = b
        self.is_servers_set = True

    def run(self) -> PriorityResults:
        """
        Run the calculation.
        """

        start = time.process_time()

        w = self.get_w()
        v = self.get_v()
        utilization = self.get_utilization()

        return PriorityResults(v=v, w=w, utilization=utilization, duration=time.process_time() - start)

    def get_utilization(self) -> float:
        """
        Calc utilization factor
        """
        b_ave = sum(b[0] for b in self.b) / len(self.b)
        l_sum = sum(self.l)

        return l_sum * b_ave / self.n

    def get_w(self) -> list[list[float]]:
        """
        Approximation of the raw moments of waiting time
        for a multi-channel queue with priorities
        based on the invariant relation M*|G*|n = M*|G*|1 * (M|G|n / M|G|1)
        """
        self._check_if_servers_and_sources_set()

        num = len(self.b[0]) - 1

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

        if self.priority == "NP":
            calc_np1 = MG1NonPreemptiveCalc()
            calc_np1.set_sources(self.l)
            calc_np1.set_servers(b1)
            w1_prty = calc_np1.get_w()
        elif self.priority == "PR":
            calc_pr1 = MG1PreemptiveCalc()
            calc_pr1.set_sources(self.l)
            calc_pr1.set_servers(b1)
            pr_prty_calc = calc_pr1.run(num=num)
            w1_prty = pr_prty_calc.w_with_pr
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

        mg1_num = MG1Calc()
        mg1_num.set_sources(l_sum)
        mg1_num.set_servers(b_sr)

        p1 = mg1_num.get_p()
        q1 = 0
        for i in range(1, self.calc_params.N):
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

        tt_n = MGnCalc(self.n, calc_params=self.calc_params)
        tt_n.set_sources(l_sum)
        tt_n.set_servers(b_sr)
        tt_n.run()
        p_n = tt_n.get_p()
        qn = 0
        for i in range(self.n + 1, self.calc_params.N):
            qn += (i - self.n) * p_n[i]

        for k in range(k_num):
            for j in range(num):
                w[k][j] = w1_prty[k][j] * qn / q1

        return w

    def get_v(self) -> list[list[float]]:
        """
        Approximation of the raw moments of sojourn time
        for a multi-channel queue with priorities
        based on the invariant relation M*|G*|n = M*|G*|1 * (M|G|n / M|G|1)
        """
        w = self.get_w()
        v = []
        k = len(self.l)
        for i in range(k):
            num = min(len(w[i]), len(self.b[i]))
            v.append(conv_moments(w[i], self.b[i], num=num))

        return v
