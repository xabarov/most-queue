from most_queue.theory.mg1_warm_calc import *


def test():
    l = 1
    b1 = 0.8
    b1_warm = 0.9
    coev = 1.3
    b_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.H2_dist.calc_theory_moments(*b_params, 4)

    b_warm_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b_warm = rd.H2_dist.calc_theory_moments(*b_params, 4)

    smo = smo_im.SmoIm(1)
    smo.set_servers(b_params, "H")
    smo.set_warm(b_warm_params, "H")
    smo.set_sources(l, "M")
    smo.run(100000)

    v_ch = get_v(l, b, b_warm)
    v_im = smo.v

    print("\nЗначения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))


if __name__ == "__main__":
    test()
