"""
RDR-A (Recursive Dimensionality Reduction, aggregated variant) for
M/M/n queues with an arbitrary number m of preemptive-resume priority classes.

Method (Harchol-Balter, Osogami, Scheller-Wolf, Wierman, "Multi-Server Queueing
Systems with Multiple Priority Classes"): to analyse class k, all higher-priority
classes 1..k-1 are aggregated into a single higher-priority stream, and the pair
(aggregate, class k) is solved by the exact two-class RDR calculator
``MMnPR2ClsBusyApprox`` (which matches the higher-priority busy period by a Cox-2
distribution). Applying this for every k reduces the m-dimensionally-infinite
Markov chain to a sequence of two-class problems -- the "A" (aggregated)
approximation of RDR.

The highest class is exactly M/M/n (``MMnrCalc``). Each lower class yields its
mean sojourn/waiting time. The aggregate of the higher classes keeps the true
aggregate offered load: its effective service rate is chosen so that
sum_i(lambda_i / mu_i) is preserved (exact when all classes share one mu, as in
the paper's figures).
"""

import time

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.fifo.mgn_takahasi import TakahashiTakamiParams
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import MMnPR2ClsBusyApprox


class RDRAPriorityCalc(BaseQueue):
    """
    RDR-A calculator: M/M/n with m preemptive-resume priority classes.

    Classes are given in priority order, index 0 being the highest priority.
    Returns mean sojourn and waiting time for every class; the highest class
    additionally carries its full raw-moment vector (exact M/M/n).
    """

    def __init__(self, n: int, calc_params: TakahashiTakamiParams | None = None):
        """
        :param n: number of servers.
        :param calc_params: parameters forwarded to the two-class RDR solver.
        """
        super().__init__(n=n)
        self.n = n
        self.two_class_params = calc_params
        self.lambdas: list[float] = []
        self.mus: list[float] = []
        self.num_classes = 0

    def set_sources(self, class_arrival_rates: list[float]):  # pylint: disable=arguments-differ
        """
        :param class_arrival_rates: arrival rate of each class, highest priority first.
        """
        self.lambdas = list(class_arrival_rates)
        self.num_classes = len(self.lambdas)
        self.is_sources_set = True

    def set_servers(self, class_service_rates: list[float]):  # pylint: disable=arguments-differ
        """
        :param class_service_rates: service rate (mu) of each class, highest priority first.
        """
        self.mus = list(class_service_rates)
        self.is_servers_set = True

    def run(self) -> PriorityResults:
        """
        Run the RDR-A recursion over all classes.
        """
        self._check_if_servers_and_sources_set()
        if len(self.lambdas) != len(self.mus):
            raise ValueError("class_arrival_rates and class_service_rates must have equal length")

        start = time.process_time()

        v_moments: list[list[float]] = []
        w_moments: list[list[float]] = []

        # Highest class: exact M/M/n (finite-r approximation with a large r).
        high = MMnrCalc(n=self.n, r=300)
        high.set_sources(l=self.lambdas[0])
        high.set_servers(mu=self.mus[0])
        high_res = high.run()
        v_moments.append(list(high_res.v))
        w_moments.append(list(high_res.w))

        # Each lower class k sees classes 0..k-1 aggregated into one stream.
        for k in range(1, self.num_classes):
            lam_high = sum(self.lambdas[:k])
            # effective aggregate service rate preserving the offered load
            # rho_high = sum_i lambda_i / mu_i  (exact when all mu_i are equal)
            load_high = sum(self.lambdas[i] / self.mus[i] for i in range(k))
            mu_high = lam_high / load_high

            two = MMnPR2ClsBusyApprox(n=self.n, calc_params=self.two_class_params)
            two.set_sources(l_low=self.lambdas[k], l_high=lam_high)
            two.set_servers(mu_low=self.mus[k], mu_high=mu_high)
            two_res = two.run()

            v1 = float(two_res.v[1][0].real if hasattr(two_res.v[1][0], "real") else two_res.v[1][0])
            w1 = v1 - 1.0 / self.mus[k]
            v_moments.append([v1, 0, 0, 0])
            w_moments.append([w1, 0, 0, 0])

        utilization = sum(self.lambdas[i] / self.mus[i] for i in range(self.num_classes)) / self.n

        results = PriorityResults(v=v_moments, w=w_moments, p=[], utilization=utilization)
        results.duration = time.process_time() - start
        return results


class RDRAPriorityPH(BaseQueue):
    """
    RDR-A for M/PH/k with m preemptive-resume priority classes and per-class
    phase-type service (the paper's Fig 5b/6/10 setting).

    Classes are given highest priority first, each with a service given by its
    first three raw moments. To analyse class k the higher classes 0..k-1 are
    aggregated into a single high class whose service moments are the arrival-rate
    mixture of theirs, and the pair (aggregate-PH, class-k-PH) is solved exactly by
    the two-class ``MPhPhK2Class`` solver. Returns the per-class mean sojourn and
    waiting time (standard FCFS-resume discipline).
    """

    def __init__(self, n: int, truncation: int = 60):
        super().__init__(n=n)
        self.n = n
        self.truncation = truncation
        self.lambdas: list[float] = []
        self.moments: list[list[float]] = []
        self.num_classes = 0

    def set_sources(self, class_arrival_rates: list[float]):  # pylint: disable=arguments-differ
        """:param class_arrival_rates: arrival rate of each class, highest priority first."""
        self.lambdas = list(class_arrival_rates)
        self.num_classes = len(self.lambdas)
        self.is_sources_set = True

    def set_servers(self, class_service_moments: list[list[float]]):  # pylint: disable=arguments-differ
        """:param class_service_moments: [E[X], E[X^2], E[X^3]] of each class, highest priority first."""
        self.moments = [list(m) for m in class_service_moments]
        self.is_servers_set = True

    def _aggregate_moments(self, k: int) -> list[float]:
        """Arrival-rate mixture of the service moments of classes 0..k-1."""
        lam = self.lambdas[:k]
        total = sum(lam)
        return [sum(lam[i] / total * self.moments[i][r] for i in range(k)) for r in range(3)]

    def run(self) -> PriorityResults:
        """Run the RDR-A recursion, solving each 2-class step with MPhPhK2Class."""
        # local import to avoid a heavy import at module load
        from most_queue.theory.priority.preemptive.mph_ph_k_2class import MPhPhK2Class, PhaseType

        self._check_if_servers_and_sources_set()
        start = time.process_time()

        v_moments: list[list[float]] = []
        w_moments: list[list[float]] = []

        for k in range(self.num_classes):
            target = PhaseType.from_moments(self.moments[k])
            if k == 0:
                # highest class: single high class, no aggregate -> M/PH/n via the
                # same exact 2-class solver with a negligible low stream.
                two = MPhPhK2Class(n=self.n, truncation=self.truncation)
                two.set_sources(l_high=self.lambdas[0], l_low=1e-9)
                two.set_servers(target, target)
                res = two.run()
                v1 = float(res.v[0][0])
            else:
                lam_high = sum(self.lambdas[:k])
                aggregate = PhaseType.from_moments(self._aggregate_moments(k))
                two = MPhPhK2Class(n=self.n, truncation=self.truncation)
                two.set_sources(l_high=lam_high, l_low=self.lambdas[k])
                two.set_servers(aggregate, target)
                res = two.run()
                v1 = float(res.v[1][0])
            v_moments.append([v1, 0, 0, 0])
            w_moments.append([v1 - self.moments[k][0], 0, 0, 0])

        utilization = sum(self.lambdas[i] * self.moments[i][0] for i in range(self.num_classes)) / self.n
        results = PriorityResults(v=v_moments, w=w_moments, p=[], utilization=utilization)
        results.duration = time.process_time() - start
        return results
