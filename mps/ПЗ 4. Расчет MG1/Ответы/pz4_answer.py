import smo_im
import rand_destribution as rd
import numpy as np
import matplotlib.pyplot as plt


def polyachek(l, coev, b1):
    """
    Формула Полячека-Хинчина для среднего времени ожидания в СМО M/G/1
    l - интенсивность вх. потока
    coev - коэффициент вариации времени обслуживания
    b1 - среднее время обслуживания
    """
    return l * b1 * b1 * (1 + coev ** 2) / (2 * (1 - l * b1))


def print_head(head):
    print("-" * len(head))
    print(head)
    print("-" * len(head))


if __name__ == "__main__":
    n = 1
    l = 1.0

    jobs_count = 100000
    ro_fix = 0.7
    coev_fix = 1.2

    coevs = np.linspace(0.3, 3, 15)
    b1 = ro_fix / l
    w_polyachek = []
    w_im = []
    errors = []

    for i in range(len(coevs)):
        smo = smo_im.SmoIm(n)
        smo.set_sources(l, 'M')
        if coevs[i] < 1:
            params = rd.Erlang_dist.get_params_by_mean_and_coev(b1, coevs[i])
            smo.set_servers(params, 'E')
        else:
            params = rd.H2_dist.get_params_by_mean_and_coev(b1, coevs[i], is_clx=False)
            smo.set_servers(params, 'H')

        smo.run(jobs_count)
        w_im.append(smo.w[0])
        w_polyachek.append(polyachek(l, coevs[i], b1))
        errors.append(100 * (w_im[i] - w_polyachek[i]) / w_polyachek[i])

    print("\nCреднее время ожидания в СМО от "
          "коэффициента вариации времени обслуживания")


    head = "{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("coev", "ИМ", "Теор", "Err, %")
    print_head(head)

    for i in range(len(coevs)):
        print("{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}".format(coevs[i], w_im[i], w_polyachek[i], errors[i]))

    fig, ax = plt.subplots()

    ax.plot(coevs, w_im, label="ИМ")
    ax.plot(coevs, w_polyachek, label="Числ")

    ax.set_xlabel('coev')
    ax.set_ylabel('w1')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()

    ax.plot(coevs, errors, label="относ ошибка ИМ")
    ax.set_xlabel('coev')
    ax.set_ylabel('err, %')
    plt.legend()
    plt.show()

    roes = np.linspace(0.1, 0.95, 15)
    w_polyachek = []
    w_im = []
    errors = []

    for i in range(len(roes)):
        smo = smo_im.SmoIm(n)
        smo.set_sources(l, 'M')
        b1 = roes[i] / l
        if coev_fix < 1:
            params = rd.Erlang_dist.get_params_by_mean_and_coev(b1, coev_fix)
            smo.set_servers(params, 'E')
        else:
            params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev_fix, is_clx=False)
            smo.set_servers(params, 'H')

        smo.run(jobs_count)
        w_im.append(smo.w[0])
        w_polyachek.append(polyachek(l, coev_fix, b1))
        errors.append(100 * (w_im[i] - w_polyachek[i]) / w_polyachek[i])

    print("\nСреднее время ожидания в СМО от коэффициента загрузки системы")
    head = "{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("ro", "ИМ", "Теор", "Err, %")
    print_head(head)

    for i in range(len(roes)):
        print("{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}".format(roes[i], w_im[i], w_polyachek[i], errors[i]))

    fig, ax = plt.subplots()

    ax.plot(roes, w_im, label="ИМ")
    ax.plot(roes, w_polyachek, label="Числ")

    ax.set_xlabel('ro')
    ax.set_ylabel('w1')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()

    ax.plot(roes, errors, label="относ ошибка ИМ")
    ax.set_xlabel('ro')
    ax.set_ylabel('err, %')
    plt.legend()
    plt.show()
