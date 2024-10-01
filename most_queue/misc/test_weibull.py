import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from most_queue.theory.utils.weibull import Weibull

matplotlib.use('TkAgg')


def test():
    """
    Тестирование аппроксимации Вейбулла
    """
    mean = 1.0
    coevs = [1.0, 2.1, 3.2]

    fig, ax = plt.subplots()

    for coev in coevs:
        k, T = Weibull.get_params_by_mean_and_coev(mean, coev)

        print("Параметры распределения Вейбулла при коэфф вариации {0:1.3f}: k = {1:1.3f} T = {2:1.3f}".format(coev, k,
                                                                                                               T))

        t = np.linspace(0, 3 * coevs[len(coevs) - 1], 100)
        dfr = Weibull.get_tail([k, T], t)

        ax.plot(t, dfr, label="$\\nu$  = {0:1.1f}".format(coev))

    ax.set_xlabel('t')
    ax.set_ylabel('ДФР')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
