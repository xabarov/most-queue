import math
from most_queue.sim import smo_im
from most_queue.sim import rand_destribution as rd
import time
from most_queue.theory.mgn_tt import MGnCalc


def test():
    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.8  # коэфф загрузки
    b1 = n * ro / l  # ср время обслуживания
    num_of_jobs = 800000  # число обсл заявок ИМ
    b_coev = [0.42, 1.5]  # коэфф вариации времени обсл

    for k in range(len(b_coev)):

        #  расчет начальных моментов времени обслуживания по заданному среднему и коэфф вариации
        b = [0.0] * 3
        alpha = 1 / (b_coev[k] ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev[k], 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        tt_start = time.process_time()
        #  запуск метода Такахаси-Таками
        tt = MGnCalc(n, l, b)
        tt.run()
        # получение результатов численных расчетов
        p_tt = tt.get_p()
        v_tt = tt.get_v()
        tt_time = time.process_time() - tt_start
        # также можно узнать сколько итераций потребовалось
        num_of_iter = tt.num_of_iter_

        # запуск ИМ для верификации результатов
        im_start = time.process_time()
        smo = smo_im.SmoIm(n)
        smo.set_sources(l, 'M')
        gamma_params = rd.Gamma.get_mu_alpha([b[0], b[1]])
        smo.set_servers(gamma_params, 'Gamma')
        smo.run(num_of_jobs)
        p = smo.get_p()
        v_im = smo.v
        im_time = time.process_time() - im_start

        # вывод результатов

        print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.\n"
              "ИМ - M/Gamma/{0:^2d}\nТакахаси-Таками - M/H2/{0:^2d}"
              "с комплексными параметрами\n"
              "Коэффициент загрузки: {1:^1.2f}\nКоэффициент вариации времени обслуживания: {2:^1.2f}\n".format(n, ro,
                                                                                                               b_coev[
                                                                                                                   k]))
        print("Количество итераций алгоритма Такахаси-Таками: {0:^4d}".format(num_of_iter))
        print("Время работы алгоритма Такахаси-Таками: {0:^5.3f} c".format(tt_time))
        print("Время работы ИМ: {0:^5.3f} c".format(im_time))
        print("{0:^25s}".format("Первые 10 вероятностей состояний СМО"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
        print("-" * 32)
        for i in range(11):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], p[i]))

        print("\n")
        print("{0:^25s}".format("Начальные моменты времени пребывания в СМО"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
        print("-" * 32)
        for i in range(3):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i + 1, v_tt[i], v_im[i]))


if __name__ == "__main__":
    test()
