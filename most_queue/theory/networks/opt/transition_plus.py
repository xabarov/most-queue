"""
Class for optimizing the transition matrix for an open network.
Version with additional features and optimizations.
}
"""
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from most_queue.theory.networks.opt.transition import (
    MaxLoadNodeResults,
    NetworkOptimizer,
    OptimizerDynamic,
)


@dataclass
class Candidates:
    """
    Results of finding candidates for optimization
    """
    optimized: bool
    candidates: list[MaxLoadNodeResults]


class Strategy(Enum):
    """
    Strategy for finding candidates for optimization
    """
    ALL = 1
    TOP_K = 2
    MIN_AND_MAX = 3
    RANDOM = 4
    TOP_ONE = 5


class NetworkOptimizerPlus(NetworkOptimizer):
    """
    Class for optimizing the transition matrix for an open network.
    Version with additional features and optimizations.
    """

    def _find_opt_candidates(self, loads: list[float],
                             intensities: list[float],
                             strategy: Strategy = Strategy.RANDOM,
                             top_k: int = 3) -> Candidates:
        """
        Find candidates for optimization based on the current loads and intensities.
        """

        candidates = []

        if self._check_if_optimized():
            return Candidates(optimized=True, candidates=candidates)

        for load_node, _load in enumerate(loads):
            # Find parent with max intesity*P[parent, max_load_node]
            parent = -1
            lam_r_max = -1
            for i in range(load_node+1):
                if i == 0:  # source
                    lam = self.arrival_rate
                else:
                    lam = intensities[i-1]
                lam_r = lam * self.R[i, load_node]
                if lam_r > lam_r_max:
                    # check if parent has >=2 childrens
                    childrens = np.sum(
                        [1 if self.R[i, k] > 0 else 0 for k in range(self.rows-1)])
                    if childrens >= 2:
                        parent = i
                        lam_r_max = lam_r

            if parent != -1:
                candidates.append(MaxLoadNodeResults(optimized=False, lam_r_max=lam_r_max,
                                                     parent=parent, node=load_node))
        # sort candidates by their loads
        candidates.sort(key=lambda x: loads[x.node], reverse=True)

        if strategy == Strategy.MIN_AND_MAX:
            if len(candidates) > 2:
                candidates = [candidates[0], candidates[-1]]
        elif strategy == Strategy.TOP_K:
            candidates = candidates[:top_k]
        elif strategy == Strategy.RANDOM:
            # choose candidate randomly
            candidates = random.sample(candidates, k=1)
        elif strategy == Strategy.TOP_ONE:
            candidates = [candidates[0]]
        else:
            # all candidates
            pass

        return Candidates(optimized=False, candidates=candidates)

    def run(self, tolerance=1e-6, max_steps=100,
            strategy: Strategy = Strategy.RANDOM,
            top_k: int = 3):
        """
        Run the optimization algorithm.
        """

        self._maximize_outs()
        self._optimize_loops()

        if self.verbose:
            self._print_header()

        net_res = self._get_network_calc()

        loads = net_res['loads']
        intencities = net_res['intensities']
        current_v1 = net_res['v'][0]
        self.dynamics.append(OptimizerDynamic(v1=current_v1, loads=loads))

        if self.verbose:
            self._print_state()

        old_v1 = 1e10
        step_num = 0

        while np.fabs(old_v1 - current_v1) > tolerance:

            if step_num > max_steps:
                break

            candidates_res = self._find_opt_candidates(
                loads, intencities, strategy=strategy, top_k=top_k)

            if candidates_res.optimized:
                break

            for candidate in candidates_res.candidates:

                z_res = self._find_balance(loads, candidate)

                parent = candidate.parent
                max_load_node = candidate.node

                self._balance(parent, max_load_node, z_res)

                old_v1 = current_v1

                net_res = self._get_network_calc()

                loads = net_res['loads']
                intencities = net_res['intensities']
                current_v1 = net_res['v'][0]

                self.dynamics.append(OptimizerDynamic(
                    v1=current_v1, loads=loads))

                if self.verbose:
                    self._print_state()
                step_num += 1

        if self.verbose:
            self._print_line()

        return self.R, current_v1
