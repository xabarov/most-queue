import rand_destribution as rd
import numpy as np
import matplotlib.pyplot as plt
import gi_m_n_calc
from matplotlib.ticker import MaxNLocator

def print_head(head):
    print("-" * len(head))
    print(head)
    print("-" * len(head))


def get_cost(v_cost, n_cost, v1, n):
    return v_cost * v1 + n_cost * n


if __name__ == "__main__":

    a1 = 1
    coev = 2.7

    ns = [x for x in range(1, 10)]

    ro_fix = 0.7
    mu = 1 / (ro_fix)
    v_cost = 1
    n_cost = 0.7

    total_costs = []
    for n in ns:
        v, alpha = rd.Gamma.get_mu_alpha_by_mean_and_coev(a1, coev)
        a = rd.Gamma.calc_theory_moments(v, alpha)
        v_ch = gi_m_n_calc.get_v(a, mu, n)

        v1 = v_ch[0]
        total_costs.append(get_cost(v_cost, n_cost, v1, n))

    print("Минимальное значение стоимости: {0:1.3f} при n = {1:d}".format(min(total_costs), np.argmin(total_costs)+1))
    fig, ax = plt.subplots()
    ax.plot(ns, total_costs)
    ax.set_xlabel('n')
    ax.set_ylabel('Стоимость')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Распределение стоимости обслуживания системы")
    plt.show()