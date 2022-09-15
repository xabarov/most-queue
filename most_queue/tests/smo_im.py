from most_queue.sim.smo_im import SmoIm
from most_queue.theory import mmnr_calc
from most_queue.theory import m_d_n_calc


def test():
    n = 3
    l = 1.0
    r = 30
    ro = 0.8
    mu = l / (ro * n)
    smo = SmoIm(n, buffer=r)

    smo.set_sources(l, 'M')
    smo.set_servers(mu, 'M')

    smo.run(1000000)

    w = mmnr_calc.M_M_n_formula.get_w(l, mu, n, r)

    w_im = smo.w

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w[j], w_im[j]))
    print("\n\nДанные ИМ::\n")
    print(smo)

    smo = SmoIm(n)

    smo.set_sources(l, 'M')
    smo.set_servers(1.0 / mu, 'D')

    smo.run(1000000)

    mdn = m_d_n_calc.M_D_n(l, 1 / mu, n)
    p_ch = mdn.calc_p()
    p_im = smo.get_p()

    print("-" * 36)
    print("{0:^36s}".format("Вероятности состояний СМО M/D/{0:d}".format(n)))
    print("-" * 36)
    print("{0:^4s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 36)
    for i in range(11):
        print("{0:^4d}|{1:^15.5g}|{2:^15.5g}".format(i, p_ch[i], p_im[i]))
    print("-" * 36)


if __name__ == "__main__":
    test()
