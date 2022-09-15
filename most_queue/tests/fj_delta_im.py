from most_queue.theory import fj_calc
from most_queue.theory import mg1_warm_calc
from most_queue.sim.fj_delta_im import SmoFJDelta
from most_queue.sim import rand_destribution as rd


def test():
    n = 3
    l = 1.0
    b1 = 0.35
    coev = 1.2
    b1_delta = 0.1
    b_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)

    delta_params = rd.H2_dist.get_params_by_mean_and_coev(b1_delta, coev)
    b_delta = rd.H2_dist.calc_theory_moments(*delta_params)
    b = rd.H2_dist.calc_theory_moments(*b_params, 4)

    smo = SmoFJDelta(n, n, b_delta, True)
    smo.set_sources(l, 'M')
    smo.set_servers(b_params, 'H')
    smo.run(100000)
    v_im = smo.v

    b_max_warm = fj_calc.getMaxMomentsDelta(n, b, 4, b_delta)
    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_warm_calc.get_v(l, b_max, b_max_warm)

    print("\n")
    print("-" * 60)
    print("{:^60s}".format('СМО Split-Join c задержкой начала обслуживания'))
    print("-" * 60)
    print("Коэфф вариации времени обслуживания: ", coev)
    print("Среднее время задежки начала обслуживания: {:4.3f}".format(b1_delta))
    print("Коэфф вариации времени задержки: {:4.3f}".format(coev))
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)

    coev = 0.53
    b1 = 0.5

    b1_delta = 0.1
    delta_params = rd.Erlang_dist.get_params_by_mean_and_coev(b1_delta, coev)
    b_delta = rd.Erlang_dist.calc_theory_moments(*delta_params)

    b_params = rd.Erlang_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.Erlang_dist.calc_theory_moments(*b_params, 4)

    smo = SmoFJDelta(n, n, b_delta, True)
    smo.set_sources(l, 'M')
    smo.set_servers(b_params, 'E')
    smo.run(100000)
    v_im = smo.v

    b_max_warm = fj_calc.getMaxMomentsDelta(n, b, 4, b_delta)
    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_warm_calc.get_v(l, b_max, b_max_warm)

    print("\n\nКоэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Среднее время задежки начала обслуживания: {:4.3f}".format(b1_delta))
    print("Коэфф вариации времени задержки: {:4.3f}".format(coev))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)


if __name__ == "__main__":
    test()
