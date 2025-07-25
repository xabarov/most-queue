"""
Input flows sum simulation
"""

import copy
import math

from tqdm import tqdm

from most_queue.random.distributions import (
    ErlangDistribution,
    GammaDistribution,
    H2Distribution,
    ParetoDistribution,
)


class FlowSumSim:
    """
    Input flows sum simulation
    """

    def __init__(
        self, a, distr="Gamma", verbose=True, num_of_moments=4, num_of_jobs=1000000
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments
        self.n = len(a)
        self.a = a
        self.distr = distr
        self.verbose = verbose
        self.num_of_moments = num_of_moments
        self.num_of_jobs = num_of_jobs
        self.cvs = []
        self.result_flow = []
        self.flows_ = []
        self.a1_sum = []
        self.a2_sum = []

    def sum_flows(self):
        """
        суммирование n потоков
        a[i][j] - i - номер потока, j номер начального момента
        интервалов между соседникми заявками i потока
        """
        n = len(self.a)  # число суммируемых потоков

        if self.distr == "Gamma":
            distr_str = "Gamma"
        elif self.distr == "H":
            distr_str = "H2"
        elif self.distr == "E":
            distr_str = "Erlang"
        elif self.distr == "Pa":
            distr_str = "Pareto"
        else:
            print("Sum of flows. ERROR: Unknown distribution type.")
            return

        for i in range(n - 1):
            if self.verbose:
                print(f"Summation of flows. Start sim {i + 1} from {n - 1}. Dist: {distr_str}")
            if self.distr == "Gamma":
                f1 = FlowSumSim.sum_2_Gamma_flows(
                    self.a[0],
                    self.a[1],
                    num_of_sim=self.num_of_jobs,
                    num_of_moments=self.num_of_moments,
                )
            elif self.distr == "H":
                f1 = FlowSumSim.sum_2_H2_flows(
                    self.a[0],
                    self.a[1],
                    num_of_sim=self.num_of_jobs,
                    num_of_moments=self.num_of_moments,
                )
            elif self.distr == "Pa":
                f1 = FlowSumSim.sum_2_Pa_flows(
                    self.a[0],
                    self.a[1],
                    num_of_sim=self.num_of_jobs,
                    num_of_moments=self.num_of_moments,
                )
            else:
                f1 = FlowSumSim.sum_2_Erlang_flows(
                    self.a[0],
                    self.a[1],
                    num_of_sim=self.num_of_jobs,
                    num_of_moments=self.num_of_moments,
                )

            self.flows_.append(f1)
            self.cvs.append(self.get_cv(f1))
            f = []
            f.append(f1)

            for j in range(len(self.a) - 2):
                f.append(self.a[j + 2])

            self.a = copy.deepcopy(f)

        self.a1_sum, self.a2_sum = zip(*self.flows_)

        self.result_flow = self.a[0]

    @staticmethod
    def sum_2_H2_flows(a1, a2, num_of_moments=4, num_of_sim=1000000):
        """
        суммирование двух потоков c коэффициентами вариации > 1
        Аппроксимация H2-распределением
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """

        y1_mus = H2Distribution.get_params(a1)
        arr1 = H2Distribution(y1_mus)
        y2_mus = H2Distribution.get_params(a2)
        arr2 = H2Distribution(y2_mus)
        arrivals = []
        time1 = arr1.generate()
        time2 = arr2.generate()
        ttek = 0

        for i in tqdm(range(num_of_sim)):
            if time1 < time2:
                arrivals.append(time1 - ttek)
                ttek = time1
                time1 = ttek + arr1.generate()
            else:
                arrivals.append(time2 - ttek)
                ttek = time2
                time2 = ttek + arr2.generate()

        f = []
        for i in range(num_of_moments):
            f.append(0)

        for k in range(num_of_moments):
            summ = 0
            for i in range(num_of_sim):
                summ += pow(arrivals[i], k + 1)
            f[k] = summ / num_of_sim

        return f

    @staticmethod
    def get_cv(a: list[complex]):
        """
        Calculating coefficient of variation for distribution.
        """
        D = a[1] - pow(a[0], 2)
        cv = math.sqrt(D) / a[0]
        return cv

    @staticmethod
    def sum_2_Pa_flows(a1, a2, num_of_moments=4, num_of_sim=1000000):
        """
        суммирование двух потоков
        Аппроксимация Pa-распределением
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """

        a_K = ParetoDistribution.get_params(a1)
        arr1 = ParetoDistribution(a_K)
        b_M = ParetoDistribution.get_params(a2)
        arr2 = ParetoDistribution(b_M)
        arrivals = []
        time1 = arr1.generate()
        time2 = arr2.generate()
        ttek = 0

        for i in tqdm(range(num_of_sim)):
            if time1 < time2:
                arrivals.append(time1 - ttek)
                ttek = time1
                time1 = ttek + arr1.generate()
            else:
                arrivals.append(time2 - ttek)
                ttek = time2
                time2 = ttek + arr2.generate()

        f = []
        for i in range(num_of_moments):
            f.append(0)

        for k in range(num_of_moments):
            summ = 0
            for i in range(num_of_sim):
                summ += pow(arrivals[i], k + 1)
            f[k] = summ / num_of_sim

        return f

    @staticmethod
    def sum_2_Erlang_flows(a1, a2, num_of_moments=4, num_of_sim=1000000):
        """
        Summing two Erlang flows with parameters a1 and a2.
        num_of_moments: number of moments to calculate.
        num_of_sim: number of simulations.
        """

        params1 = ErlangDistribution.get_params(a1)
        arr1 = ErlangDistribution(params1)
        params2 = ErlangDistribution.get_params(a2)
        arr2 = ErlangDistribution(params2)
        arrivals = []
        time1 = arr1.generate()
        time2 = arr2.generate()
        ttek = 0

        for i in tqdm(range(num_of_sim)):
            if time1 < time2:
                arrivals.append(time1 - ttek)
                ttek = time1
                time1 = ttek + arr1.generate()
            else:
                arrivals.append(time2 - ttek)
                ttek = time2
                time2 = ttek + arr2.generate()

        f = []
        for i in range(num_of_moments):
            f.append(0)

        for k in range(num_of_moments):
            summ = 0
            for i in range(num_of_sim):
                summ += pow(arrivals[i], k + 1)
            f[k] = summ / num_of_sim

        return f

    @staticmethod
    def sum_2_Gamma_flows(a1, a2, num_of_moments=4, num_of_sim=1000000):
        """
        суммирование двух потоков c коэффициентами вариации > 1
        Аппроксимация H2-распределением
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """

        params1 = GammaDistribution.get_params(a1)
        arr1 = GammaDistribution(params1)
        params2 = GammaDistribution.get_params(a2)
        arr2 = GammaDistribution(params2)
        arrivals = []
        time1 = arr1.generate()
        time2 = arr2.generate()
        ttek = 0

        for i in tqdm(range(num_of_sim)):
            if time1 < time2:
                arrivals.append(time1 - ttek)
                ttek = time1
                time1 = ttek + arr1.generate()
            else:
                arrivals.append(time2 - ttek)
                ttek = time2
                time2 = ttek + arr2.generate()

        f = []
        for i in range(num_of_moments):
            f.append(0)

        for k in range(num_of_moments):
            summ = 0
            for i in range(num_of_sim):
                summ += pow(arrivals[i], k + 1)
            f[k] = summ / num_of_sim

        return f

    @staticmethod
    def sum_n_flows(a, disr="Gamma", verbose=True):
        """
        суммирование n потоков
        a[i][j] - i - номер потока, j номер начального момента
        интервалов между соседникми заявками i потока
        """
        n = len(a)  # число суммируемых потоков

        if disr == "Gamma":
            distr_str = "Gamma"
        elif disr == "H":
            distr_str = "H2"
        elif disr == "E":
            distr_str = "Erlang"
        else:
            print("Sum of flows. ERROR: Unknown distribution type.")
            return 0

        for i in range(n - 1):
            if verbose:
                print(f"Summation of flows. Start sim {i + 1} from { n - 1}. Dist: {distr_str}")

            if disr == "Gamma":
                f1 = FlowSumSim.sum_2_Gamma_flows(a[0], a[1])
            elif disr == "H":
                f1 = FlowSumSim.sum_2_H2_flows(a[0], a[1])
            else:
                f1 = FlowSumSim.sum_2_Erlang_flows(a[0], a[1])

            f = []
            for j in range(len(a) - 2):
                f.append(a[j + 2])
            f.append(f1)
            a = copy.deepcopy(f)

        return a[0]
