from most_queue.sim.smo_im_prty import SmoImPrty
from most_queue.theory import prty_calc
from most_queue.sim import rand_destribution as rd


def test():
    n = 5 # число каналов
    k = 3 # число классов заявок
    l = [0.2, 0.3, 0.4]  # интенсивности пребытия заявок по классам
    lsum = sum(l)
    num_of_jobs = 1000000  # количество заявок для ИМ

    # Зададим параметры обслуживания по начальным моментам.
    # Зададим средние времена обслуживания по классам
    b1 = [0.45 * n, 0.9 * n, 1.35 * n]

    # Коэфф вариации времени обслуживания пусть будет одинаковый для всех классов
    coev = 0.577

    # вторые начальные моменты
    b2 = [0] * k #
    for i in range(k):
        b2[i] = (b1[i] ** 2) * (1 + coev ** 2)

    b_sr = sum(b1) / k

    # получим коэфф загрузки
    ro = lsum * b_sr / n

    # теперь по заданным двум нач моментам подберем параметры аппроксимирующего Гамма-распределения
    # и добавим в список параметров params
    params = []
    for i in range(k):
        params.append(rd.Gamma.get_mu_alpha([b1[i], b2[i]]))

    b = []
    for j in range(k):
        b.append(rd.Gamma.calc_theory_moments(params[j][0], params[j][1], 4))

    print("\nСравнение данных ИМ и результатов расчета методом инвариантов отношения (Р) \n"
          "времени пребывания в многоканальной СМО с приоритетами")
    print("Число каналов: " + str(n) + "\nЧисло классов: " + str(k) + "\nКоэффициент загрузки: {0:<1.2f}".format(ro) +
          "\nКоэффициент вариации времени обслуживания: " + str(coev) + "\n")
    print("Абсолютный приоритет")

    # при создании ИМ передаем число каналов, число классов и тип приоритета.
    # PR - абсолютный с дообслуживанием заявок
    smo = SmoImPrty(n, k, "PR")

    # для задания источников заявок и каналов обслуживания нужно задать набор словарей с полями
    # type - тип распределения,
    # params - его параметры.
    # Число таких словарей в списках sources и servers_params соответствует числу классов

    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    smo.set_sources(sources)
    smo.set_servers(servers_params)

    # запуск ИМ
    smo.run(num_of_jobs)

    # получение начальных моментов времени пребывания

    v_im = smo.v

    # расчет их же методом инвариантов отношения (для сравнения)
    v_teor = prty_calc.get_v_prty_invar(l, b, n, 'PR')

    # вывод результатов
    print("-" * 60)
    print("{0:^11s}|{1:^47s}|".format('', 'Номер начального момента'))
    print("{0:^10s}| ".format('№ кл'), end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(3):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 60)

    for i in range(k):
        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("ИМ"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_im[i][j]), end="")
        print("")
        print("{:^5s}".format(str(i + 1)) + "|" + "-" * 54)

        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("Р"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_teor[i][j]), end="")
        print("")
        print("-" * 60)

    print("\n")
    print("Относительный приоритет")

    # Тоже самое для относительного приоритета (NP)
    smo = SmoImPrty(n, k, "NP")
    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    smo.set_sources(sources)
    smo.set_servers(servers_params)

    smo.run(num_of_jobs)

    v_im = smo.v

    v_teor = prty_calc.get_v_prty_invar(l, b, n, 'NP')

    print("-" * 60)
    print("{0:^11s}|{1:^47s}|".format('', 'Номер начального момента'))
    print("{0:^10s}| ".format('№ кл'), end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(3):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 60)

    for i in range(k):
        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("ИМ"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_im[i][j]), end="")
        print("")
        print("{:^5s}".format(str(i + 1)) + "|" + "-" * 54)

        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("Р"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_teor[i][j]), end="")
        print("")
        print("-" * 60)


if __name__ == "__main__":
    test()
