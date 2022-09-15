from most_queue.sim.smo_im_prty import SmoImPrty
from most_queue.theory import prty_calc
from most_queue.sim import rand_destribution as rd


def test():
    n = 5
    k = 3
    l = [0.2, 0.3, 0.4]
    lsum = sum(l)
    num_of_jobs = 1000000
    b1 = [0.45 * n, 0.9 * n, 1.35 * n]
    b2 = [0] * k
    coev = 0.577

    for i in range(k):
        b2[i] = (b1[i] ** 2) * (1 + coev ** 2)
    b_sr = sum(b1) / k
    ro = lsum * b_sr / n
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

    smo = SmoImPrty(n, k, "PR")
    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    smo.set_sources(sources)
    smo.set_servers(servers_params)

    smo.run(num_of_jobs)

    # v_im = smo.v
    # calc1pr = prty_calc.calc_pr1(l, b)
    # v_ch = calc1pr['v']
    #
    # #w_ch = prty_calc.climov_w_pr_calc(l, b)
    # # w_ch_1 = prty_calc.get_w1_pr(l, b)
    # # for j in range(3):
    # #     print("{:^15.3g}|".format(w_ch_1[j]), end="")
    # # print("\n")
    #
    # for i in range(k):
    #     print(" " * 5 + "|", end="")
    #     print("{:^5s}|".format("ИМ"), end="")
    #     for j in range(3):
    #         print("{:^15.3g}|".format(v_im[i][j]), end="")
    #     print("")
    #     print("{:^5s}".format(str(i + 1)) + "|" + "-" * 54)
    #
    #     print(" " * 5 + "|", end="")
    #     print("{:^5s}|".format("Р"), end="")
    #     for j in range(3):
    #         print("{:^15.3g}|".format(v_ch[i][j]), end="")
    #     print("")
    #     print("-" * 60)

    v_im = smo.v

    v_teor = prty_calc.get_v_prty_invar(l, b, n, 'PR')

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
