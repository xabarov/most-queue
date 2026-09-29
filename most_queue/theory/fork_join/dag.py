"""
Fork-Join over a series-parallel task DAG (EPIC-030), generalizing SplitJoinCalc
(most_queue.theory.fork_join.split_join, EPIC-022) from one flat fork->n->join
level with i.i.d. branches to an arbitrary nesting of "series" (sequential,
sum of durations) and "parallel" (fork-join, max of durations) composition,
with branches that need not be identically distributed.

A DAG node is one of:
    ("leaf", family, spec)             -- a single sub-task
    ("series", [node, node, ...])      -- sequential: sum of children's durations
    ("parallel", [node, node, ...])    -- fork-join: max of children's durations
    ("parallel", [node, ...], k)       -- (n,k)-fork-join (EPIC-031): done once any
                                           k of the n children finish; k defaults to
                                           len(children) (plain fork-join, EPIC-030)

`family` is "pareto" (spec = ParetoParams, exact) or "gamma"/"h2"/"erlang"
(spec = raw moments, fitted -- same convention as
most_queue.theory.utils.max_dist.branch_tail).

Composition is exact at the leaf level (Pareto) or numerically exact given
the assumed family (gamma/h2/erlang, via quadrature) -- same standard as
EPIC-022. A "parallel" node whose children are themselves composite
subtrees needs one extra step: the subtree's already-computed raw moments
are fitted to `fit_family` (default "gamma") to obtain a tail function for
the max integral -- the same fit-based approximation SplitJoinCalc already
applies to its own non-Pareto i.i.d. case, not a new source of inexactness
(see docs/research/fork-join-dag-heterogeneous-2026.md sec. "Scope decision").

The whole DAG's root moments are the per-job service-time moments of a
Split-Join queue (blocking: a new job does not enter until the current one
finishes, EPIC-022's semantics) -- wrapped in the exact M/G/1
(Pollaczek-Khinchine) formula (MG1Calc), no new queueing-level method.
"""

from typing import Any

from most_queue.random.distributions import ParetoDistribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.max_dist import heterogeneous_kth_order_moments

DAGNode = tuple  # ("leaf", family, spec) | ("series", list[DAGNode]) | ("parallel", list[DAGNode])


def _leaf_moments(family: str, spec: Any, num: int) -> list[float]:
    if family == "pareto":
        return list(ParetoDistribution.calc_theory_moments(spec, num))
    return list(spec[:num])


def dag_moments(node: DAGNode, num: int = 4, fit_family: str = "gamma") -> list[float]:
    """Raw moments of the total duration of the (sub-)DAG rooted at `node`."""
    kind = node[0]

    if kind == "leaf":
        _, family, spec = node
        return _leaf_moments(family, spec, num)

    if kind == "series":
        _, children = node
        if not children:
            raise ValueError("series node must have at least one child")
        acc = dag_moments(children[0], num, fit_family)
        for child in children[1:]:
            acc = list(conv_moments(acc, dag_moments(child, num, fit_family), num))
        return acc

    if kind == "parallel":
        children = node[1]
        k = node[2] if len(node) > 2 else len(children)
        if not children:
            raise ValueError("parallel node must have at least one child")
        if not 1 <= k <= len(children):
            raise ValueError(f"k must satisfy 1 <= k <= len(children), got k={k}, len(children)={len(children)}")
        branches = []
        for child in children:
            if child[0] == "leaf":
                branches.append((child[1], child[2]))
            else:
                branches.append((fit_family, dag_moments(child, num, fit_family)))
        return heterogeneous_kth_order_moments(branches, k, num)

    raise ValueError(f"unknown DAG node kind {kind!r}; expected 'leaf', 'series' or 'parallel'")


class ForkJoinDAGCalc(BaseQueue):
    """
    Split-Join (blocking) queue whose per-job service time is the total
    duration of a series-parallel task DAG (see module docstring).
    """

    def __init__(self, dag: DAGNode, fit_family: str = "gamma", calc_params: CalcParams | None = None):
        """
        :param dag: DAG node spec, see module docstring.
        :param fit_family: family used to re-fit a composite subtree's raw
            moments before feeding it into a `parallel` node's max
            integration (only matters for a `parallel` node with non-leaf
            children); "gamma", "h2" or "erlang".
        """
        super().__init__(n=1, calc_params=calc_params)
        self.dag = dag
        self.fit_family = fit_family
        self.l = None
        self.b_dag: list[float] | None = None
        self.is_servers_set = True  # the DAG spec (given at construction) *is* the "servers"

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Not used -- the DAG spec passed to __init__ already fully specifies service times."""
        raise NotImplementedError("ForkJoinDAGCalc takes its service-time spec as the `dag` constructor argument")

    def get_dag_moments(self, num: int = 4) -> list[float]:
        """Raw moments of one job's total DAG completion time."""
        if self.b_dag is None:
            self.b_dag = dag_moments(self.dag, num, self.fit_family)
        return self.b_dag

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """Solve the wrapping M/G/1 (Pollaczek-Khinchine) queue over the DAG's completion time."""
        start = self._measure_time()
        with self._validate_state():
            b_dag = self.get_dag_moments(num_of_moments)
            mg1 = MG1Calc()
            mg1.set_sources(self.l)
            mg1.set_servers(b_dag)
            result = mg1.run(num_of_moments)
        self._set_duration(result, start)
        return result
