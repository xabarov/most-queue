"""
Unit tests for the series-parallel Fork-Join task DAG calculator
(most_queue.theory.fork_join.dag, EPIC-030).
"""

import numpy as np
import pytest

from most_queue.random.utils.params import ParetoParams
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fork_join.dag import ForkJoinDAGCalc, dag_moments
from most_queue.theory.fork_join.split_join import SplitJoinCalc


def test_flat_iid_pareto_dag_reduces_to_split_join():
    """A one-level "parallel" of n identical Pareto leaves must match SplitJoinCalc(pareto)."""
    params = ParetoParams(alpha=5.0, K=1.0)
    dag = ("parallel", [("leaf", "pareto", params)] * 3)

    calc = ForkJoinDAGCalc(dag)
    calc.set_sources(0.1)
    res = calc.run()

    sj = SplitJoinCalc(n=3, calc_params=CalcParams(approx_distr="pareto"))
    sj.set_sources(0.1)
    sj.set_servers(params)
    res_sj = sj.run()

    assert np.allclose(res.v[:2], res_sj.v[:2], rtol=1e-4)


def test_two_stage_series_of_parallel_matches_monte_carlo():
    """series(parallel(leaf,leaf), parallel(leaf,leaf)) -- all leaves, no fit step needed."""
    p1 = ParetoParams(alpha=6.0, K=1.0)
    p2 = ParetoParams(alpha=7.0, K=1.2)
    p3 = ParetoParams(alpha=6.0, K=0.8)
    gamma_moments = [2.0, 5.0, 14.0]  # mean=2, var=1

    dag = (
        "series",
        [
            ("parallel", [("leaf", "pareto", p1), ("leaf", "gamma", gamma_moments)]),
            ("parallel", [("leaf", "pareto", p2), ("leaf", "pareto", p3)]),
        ],
    )
    theory = dag_moments(dag, num=3)

    rng = np.random.default_rng(1)
    n = 2_000_000
    x1 = (rng.pareto(p1.alpha, n) + 1) * p1.K
    x2 = rng.gamma(4.0, 0.5, n)  # mean=2, var=1
    stage1 = np.maximum(x1, x2)

    x3 = (rng.pareto(p2.alpha, n) + 1) * p2.K
    x4 = (rng.pareto(p3.alpha, n) + 1) * p3.K
    stage2 = np.maximum(x3, x4)

    total = stage1 + stage2
    mc = [np.mean(total**k) for k in (1, 2, 3)]

    assert np.allclose(theory, mc, rtol=0.01)


def test_composite_subtree_inside_parallel_matches_monte_carlo():
    """parallel(series(leaf,leaf), leaf) -- exercises the fit-based composite-node step."""
    p1 = ParetoParams(alpha=6.0, K=1.0)
    p2 = ParetoParams(alpha=7.0, K=0.9)
    gamma_moments = [1.5, 3.0, 8.0]  # mean=1.5, var=0.75

    dag = (
        "parallel",
        [
            ("series", [("leaf", "pareto", p1), ("leaf", "gamma", gamma_moments)]),
            ("leaf", "pareto", p2),
        ],
    )
    theory = dag_moments(dag, num=2, fit_family="gamma")

    rng = np.random.default_rng(2)
    n = 2_000_000
    x1 = (rng.pareto(p1.alpha, n) + 1) * p1.K
    x2 = rng.gamma(3.0, 0.5, n)  # mean=1.5, var=0.75
    branch_a = x1 + x2
    x3 = (rng.pareto(p2.alpha, n) + 1) * p2.K
    mx = np.maximum(branch_a, x3)
    mc = [np.mean(mx**k) for k in (1, 2)]

    # only mean/variance checked -- the moment-fit re-approximation error
    # (expected, documented) concentrates in the higher moments
    assert np.allclose(theory, mc, rtol=0.02)


def test_nk_parallel_node_matches_monte_carlo():
    """("parallel", children, k) -- (n,k)-fork-join (EPIC-031) inside a DAG node."""
    p1 = ParetoParams(alpha=6.0, K=1.0)
    p2 = ParetoParams(alpha=7.0, K=1.2)
    p3 = ParetoParams(alpha=6.0, K=0.8)

    dag = ("parallel", [("leaf", "pareto", p1), ("leaf", "pareto", p2), ("leaf", "pareto", p3)], 2)
    theory = dag_moments(dag, num=2)

    rng = np.random.default_rng(4)
    n = 2_000_000
    x1 = (rng.pareto(p1.alpha, n) + 1) * p1.K
    x2 = (rng.pareto(p2.alpha, n) + 1) * p2.K
    x3 = (rng.pareto(p3.alpha, n) + 1) * p3.K
    stacked = np.sort(np.vstack([x1, x2, x3]), axis=0)
    mc = [np.mean(stacked[1] ** m) for m in (1, 2)]  # k=2 -> 2nd smallest

    assert np.allclose(theory, mc, rtol=0.01)


def test_nk_parallel_node_default_k_unchanged():
    """Omitting k must give the exact same result as EPIC-030's plain fork-join."""
    p1 = ParetoParams(alpha=5.0, K=1.0)
    children = [("leaf", "pareto", p1), ("leaf", "gamma", [1.0, 2.5, 8.0])]
    without_k = dag_moments(("parallel", children), num=3)
    with_k_equal_n = dag_moments(("parallel", children, len(children)), num=3)
    assert np.allclose(without_k, with_k_equal_n, rtol=1e-9)


def test_run_gives_sane_queue_results():
    p1 = ParetoParams(alpha=5.0, K=1.0)
    dag = ("parallel", [("leaf", "pareto", p1), ("leaf", "gamma", [1.0, 2.5, 8.0])])
    calc = ForkJoinDAGCalc(dag)
    calc.set_sources(0.2)
    res = calc.run()
    assert res.v[0] > 0
    assert res.w[0] >= 0
    assert 0 < res.utilization < 1


def test_set_servers_rejected():
    dag = ("leaf", "gamma", [1.0, 2.0, 6.0])
    calc = ForkJoinDAGCalc(dag)
    with pytest.raises(NotImplementedError):
        calc.set_servers()


def test_unknown_node_kind_rejected():
    with pytest.raises(ValueError):
        dag_moments(("bogus", []), num=2)


if __name__ == "__main__":
    test_flat_iid_pareto_dag_reduces_to_split_join()
    test_two_stage_series_of_parallel_matches_monte_carlo()
    test_composite_subtree_inside_parallel_matches_monte_carlo()
    test_nk_parallel_node_matches_monte_carlo()
    test_nk_parallel_node_default_k_unchanged()
    test_run_gives_sane_queue_results()
    test_set_servers_rejected()
    test_unknown_node_kind_rejected()
    print("all fork-join DAG tests passed")
