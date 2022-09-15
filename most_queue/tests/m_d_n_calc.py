from most_queue.sim import smo_im
from most_queue.theory.m_d_n_calc import M_D_n


def test():
    l = 1.0  # интенсивность входного потока
    ro = 0.8  # коэффициент загрузки
    n = 2  # количество каналов обслуживания
    num_of_jobs = 800000  # количество заявок для ИМ

    b = ro * n / l
    mdn = M_D_n(l, b, n)
    p_ch = mdn.calc_p()

    smo = smo_im.SmoIm(n)
    smo.set_sources(l, "M")
    smo.set_servers(b, "D")
    smo.run(num_of_jobs)
    v_im = smo.v
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
