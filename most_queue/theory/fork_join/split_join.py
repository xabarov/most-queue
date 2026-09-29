"""
Numerical calculation of Fork-Join queuing systems
"""

import math

from most_queue.random.utils.params import ParetoParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.utils.max_dist import MaxDistribution, pareto_max_moments
from most_queue.theory.vacations.mg1_warm_calc import CalcParams, MG1WarmCalc


class SplitJoinCalc(BaseQueue):
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

    def __init__(self, n: int, calc_params: CalcParams | None = None):
        """
        :param n: number of servers
        :param calc_params: calculation parameters
        Notice:
            calc_params.approx_distr for the raw moments of service time
            can be 'gamma', 'h2', 'erlang' or 'pareto', default is 'gamma'.
            'pareto' is exact (no approximation): the sub-task service time
            is Pareto-distributed and set_servers() takes a ParetoParams
            directly instead of raw moments (see EPIC-022 / max_dist.pareto_max_moments).
        """

        super().__init__(n=n, calc_params=calc_params)

        self.l = None
        self.b = None

        self.b_max = None
        self.b_max_warm = None
        self.approximation = self.calc_params.approx_distr

        if self.approximation not in ["gamma", "h2", "erlang", "pareto"]:
            raise ValueError("Approximation must be one of 'gamma', 'h2', 'erlang' or 'pareto'.")

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        self.l = l

        self.is_sources_set = True

    def set_servers(self, b: list[float] | ParetoParams):  # pylint: disable=arguments-differ
        """
        Set servers
        :param b: list of raw moments of service time, or (when
            approximation == 'pareto') a ParetoParams(alpha, K).
        """
        if self.approximation == "pareto" and not isinstance(b, ParetoParams):
            raise TypeError("approximation='pareto' requires set_servers(ParetoParams(alpha, K))")
        if self.approximation != "pareto" and isinstance(b, ParetoParams):
            raise TypeError(f"ParetoParams requires approximation='pareto', got '{self.approximation}'")
        self.b = b

        self.is_servers_set = True

    def run(self) -> QueueResults:
        """
        Run calculations for Split-Join queueing systems
        """

        start = self._measure_time()
        v = self.get_v()
        utilization = self.get_utilization()

        result = QueueResults(v=v, utilization=utilization)
        self._set_duration(result, start)
        return result

    def get_v(self) -> list[float]:
        """
        Calculate sojourn time raw moments for Split-Join queueing systems

        :return: list[float] : raw moments of sojourn time distribution
        """

        # Calc Split-Join max of n channels service time distribution

        if not self.v is None:
            return self.v

        if self.approximation == "pareto":
            self.b_max = self._pareto_b_max()
        else:
            max_distr = MaxDistribution(b=self.b, n=self.n, approximation=self.approximation)
            self.b_max = max_distr.get_max_moments()

        # Further calculation as in a regular M/G/1 queueing system with
        # raw moments of the distribution maximum of the random variable
        mg1 = MG1Calc()
        mg1.set_sources(self.l)
        mg1.set_servers(self.b_max)
        self.v = mg1.get_v()
        return self.v

    def _pareto_b_max(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of the max of n iid Pareto sub-task service times
        (see most_queue.theory.utils.max_dist.pareto_max_moments). Requests
        as many moments as exist, capped at `num`: M/G/1's P-K formula needs
        at least a finite variance (alpha > 2); for alpha <= 2 the exact max
        tail/CDF (pareto_max_tail) is still well-defined and usable on its
        own, just not the full-queue sojourn time via P-K.
        """
        alpha = self.b.alpha
        if alpha <= 2:
            raise ValueError(
                f"Split-Join's M/G/1 (Pollaczek-Khinchine) step requires a finite service variance "
                f"(alpha > 2), got alpha={alpha}. For alpha <= 2, use "
                "most_queue.theory.utils.max_dist.pareto_max_tail directly for the exact tail of "
                "the sub-task maximum instead of the full-queue sojourn time."
            )
        max_k = min(num, math.floor(alpha - 1e-9))
        return pareto_max_moments(self.b, self.n, num=int(max_k))

    def get_v_delta(self, b_delta: list[float] | float) -> list[float]:
        """
        Calculate sojourn time raw moments for Split-Join queueing systems with delta
        :param b_delta:  If delta is a list, it should contain the moments of
        time delay caused by reception and restoration operations for each part.
        If delta is a float, delay is determistic and equal to delta.
        :return: list[float] : raw moments of sojourn time distribution
        """
        if self.approximation == "pareto":
            raise NotImplementedError(
                "get_v_delta() (warm-up/reception delay) is not implemented for approximation='pareto'"
            )

        max_distr = MaxDistribution(b=self.b, n=self.n, approximation=self.approximation)
        self.b_max = max_distr.get_max_moments()

        self.b_max_warm = max_distr.get_max_moments_delta(b_delta)

        mg1_approx = "gamma" if self.approximation == "erlang" else self.approximation
        calc_params = CalcParams(approx_distr=mg1_approx)
        mg1_warm = MG1WarmCalc(calc_params=calc_params)
        mg1_warm.set_sources(self.l)
        mg1_warm.set_servers(self.b_max, self.b_max_warm)
        return mg1_warm.get_v()

    def get_utilization(self):
        """
        Calculate the utilization factor for Split-Join queueing systems
        """

        if self.b_max is None:
            self.get_v()

        return self.l * self.b_max[0]
