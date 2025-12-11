"""
Input flows sum simulation
"""

import copy
import math

from tqdm import tqdm

from most_queue.constants import DEFAULT_NUM_JOBS
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
        self, a, distr="Gamma", verbose=True, num_of_moments=4, num_of_jobs=DEFAULT_NUM_JOBS
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
    def _sum_2_flows_generic(
        distribution_class, get_params_method, a1, a2, num_of_moments=4, num_of_sim=DEFAULT_NUM_JOBS
    ):
        """
        Generic method for summing two flows with any distribution.
        :param distribution_class: Distribution class (e.g., H2Distribution, GammaDistribution)
        :param get_params_method: Method to get parameters from moments (e.g., H2Distribution.get_params)
        :param a1: List of initial moments for first flow
        :param a2: List of initial moments for second flow
        :param num_of_moments: Number of moments to calculate
        :param num_of_sim: Number of simulations
        :return: List of moments for the summed flow
        """
        params1 = get_params_method(a1)
        arr1 = distribution_class(params1)
        params2 = get_params_method(a2)
        arr2 = distribution_class(params2)
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

        f = [0] * num_of_moments

        for k in range(num_of_moments):
            summ = 0
            for i in range(num_of_sim):
                summ += pow(arrivals[i], k + 1)
            f[k] = summ / num_of_sim

        return f

    @staticmethod
    def sum_2_H2_flows(a1, a2, num_of_moments=4, num_of_sim=DEFAULT_NUM_JOBS):
        """
        суммирование двух потоков c коэффициентами вариации > 1
        Аппроксимация H2-распределением
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """
        return FlowSumSim._sum_2_flows_generic(
            H2Distribution, H2Distribution.get_params, a1, a2, num_of_moments, num_of_sim
        )

    @staticmethod
    def get_cv(a: list[complex]):
        """
        Calculating coefficient of variation for distribution.
        """
        D = a[1] - pow(a[0], 2)
        cv = math.sqrt(D) / a[0]
        return cv

    @staticmethod
    def sum_2_Pa_flows(a1, a2, num_of_moments=4, num_of_sim=DEFAULT_NUM_JOBS):
        """
        суммирование двух потоков
        Аппроксимация Pa-распределением
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """
        return FlowSumSim._sum_2_flows_generic(
            ParetoDistribution, ParetoDistribution.get_params, a1, a2, num_of_moments, num_of_sim
        )

    @staticmethod
    def sum_2_Erlang_flows(a1, a2, num_of_moments=4, num_of_sim=DEFAULT_NUM_JOBS):
        """
        Summing two Erlang flows with parameters a1 and a2.
        num_of_moments: number of moments to calculate.
        num_of_sim: number of simulations.
        """
        return FlowSumSim._sum_2_flows_generic(
            ErlangDistribution, ErlangDistribution.get_params, a1, a2, num_of_moments, num_of_sim
        )

    @staticmethod
    def sum_2_Gamma_flows(a1, a2, num_of_moments=4, num_of_sim=DEFAULT_NUM_JOBS):
        """
        суммирование двух потоков c коэффициентами вариации > 1
        Аппроксимация H2-распределением
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """
        return FlowSumSim._sum_2_flows_generic(
            GammaDistribution, GammaDistribution.get_params, a1, a2, num_of_moments, num_of_sim
        )

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
