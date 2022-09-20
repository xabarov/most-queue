from most_queue.sim.smo_im import SmoIm
from most_queue.theory import mmnr_calc
from most_queue.theory import m_d_n_calc


def test():
    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    r = 30  # длина очереди
    ro = 0.8  # коэфф загрузки
    mu = l / (ro * n)  # интенсивность обслуживания

    # создвем экземпляр класса ИМ
    smo = SmoIm(n, buffer=r)

    # задаем вх поток - параметры и тип распределения.
    smo.set_sources(l, 'M')
    # задаем распределение обслуживания - параметры и тип распределения.
    smo.set_servers(mu, 'M')

    # запуск ИМ - передаем кол-во заявок для обслуживания
    smo.run(1000000)
    # получаем начальные моменты времени ожидания. Также можно получить время пребывания .v,
    # вероятности состояний .get_p(), периоды непрерывной занятости .pppz
    w_im = smo.w

    # для сравнения расчитаем теже начальные моменты численно
    w = mmnr_calc.M_M_n_formula.get_w(l, mu, n, r)

    # вывод результатов

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w[j], w_im[j]))
    print("\n\nДанные ИМ::\n")
    print(smo)

    # тоже для детерминированного обслуживания

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
