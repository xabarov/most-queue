import numpy as np

from most_queue.theory import fj_calc
from most_queue.theory import mg1_warm_calc
from most_queue.general_utils.tables import times_print
from most_queue.sim.fj_delta_sim import ForkJoinSimDelta
from most_queue.rand_distribution import H2_dist, Erlang_dist


def test_fj_delta():
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
    print("{:^60s}".format('СМО Split-Join c задержкой начала обслуживания'))
    print("-" * 60)
    print("Коэфф вариации времени обслуживания: ", coev)
    print(
        "Среднее время задежки начала обслуживания: {:4.3f}".format(b1_delta))
    print("Коэфф вариации времени задержки: {:4.3f}".format(coev))
    print("Коэффициент загрузки: {:4.3f}".format(ro))

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

    print("\n\nКоэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print(
        "Среднее время задежки начала обслуживания: {:4.3f}".format(b1_delta))
    print("Коэфф вариации времени задержки: {:4.3f}".format(coev))

    times_print(v_im, v_ch, is_w=False)

    assert len(v_im) == len(v_ch)
    assert np.allclose(np.array(v_im[:1]), np.array(v_ch[:1]), rtol=1e-1)


if __name__ == "__main__":

    test_fj_delta()
