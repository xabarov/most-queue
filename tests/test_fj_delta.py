"""
Test for ForkJoin queue with delta.
"""
import numpy as np

from most_queue.general_utils.tables import times_print
from most_queue.rand_distribution import Erlang_dist, H2_dist
from most_queue.sim.fj_delta_sim import ForkJoinSimDelta
from most_queue.theory import fj_calc, mg1_warm_calc


def test_fj_delta():
    """
    Test for ForkJoin queue with delta.
    """
    n = 3
    l = 1.0
    b1 = 0.35
    coev = 1.2
    b1_delta = 0.1
    b_params = H2_dist.get_params_by_mean_and_coev(b1, coev)

    delta_params = H2_dist.get_params_by_mean_and_coev(b1_delta, coev)
    b_delta = H2_dist.calc_theory_moments(*delta_params)
    b = H2_dist.calc_theory_moments(*b_params, 4)

    qs = ForkJoinSimDelta(n, n, b_delta, True)

    qs.set_sources(l, 'M')
    qs.set_servers(b_params, 'H')
    qs.run(100000)
    v_im = qs.v

    b_max_warm = fj_calc.getMaxMomentsDelta(n, b, 4, b_delta)
    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_warm_calc.get_v(l, b_max, b_max_warm)

    print("\n")
    print("-" * 60)
    print(f"{'Split-Join QS with service start delay':^60s}")
    print("-" * 60)
    print(f"Coefficient of variation of service time: {coev}")
    print(f"Average delay before service start: {b1_delta:.3f}")
    print(f"Coefficient of variation of delay: {coev:.3f}")
    print(f"Utilization coefficient: {ro:.3f}")

    times_print(v_im, v_ch, is_w=False)

    assert len(v_im) == len(v_ch)
    assert np.allclose(np.array(v_im[:1]), np.array(v_ch[:1]), rtol=1e-1)

    coev = 0.53
    b1 = 0.5

    b1_delta = 0.1
    delta_params = Erlang_dist.get_params_by_mean_and_coev(b1_delta, coev)
    b_delta = Erlang_dist.calc_theory_moments(*delta_params)

    b_params = Erlang_dist.get_params_by_mean_and_coev(b1, coev)
    b = Erlang_dist.calc_theory_moments(*b_params, 4)

    qs = ForkJoinSimDelta(n, n, b_delta, True)
    qs.set_sources(l, 'M')
    qs.set_servers(b_params, 'E')
    qs.run(100000)
    v_im = qs.v

    b_max_warm = fj_calc.getMaxMomentsDelta(n, b, 4, b_delta)
    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_warm_calc.get_v(l, b_max, b_max_warm)

    print("\n\nCoefficient of variation of service time: ", coev)
    print(f"Load coefficient: {ro:.3f}")
    print(
        f"Average waiting time for service start: {b1_delta:.3f}"
    )
    print(f"Coefficient of variation of waiting time: {coev}")

    times_print(v_im, v_ch, is_w=False)

    assert len(v_im) == len(v_ch)
    assert np.allclose(np.array(v_im[:1]), np.array(v_ch[:1]), rtol=1e-1)


if __name__ == "__main__":

    test_fj_delta()
