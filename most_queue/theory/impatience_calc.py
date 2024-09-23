import math

"""
Расчет М/M/1 с экспоненциальным нетерпением
"""


def get_p0(l, mu, gamma, tol=1e-12):
    """
    Вероятность нулевого состояния системы
    """
    summ = 0
    elem_old = l
    elem_new = 0

    i = 1
    while math.fabs(elem_new - elem_old) > tol:
        chisl = math.pow(l, i)
        znam = mu
        j = 1
        while j < i:
            znam *= (mu + j * gamma)
            j += 1

        elem_old = elem_new
        elem_new = chisl / znam
        summ += elem_new

        i += 1

    return 1.0 / (1.0 + summ)


def get_p(l, mu, gamma, tol=1e-12, max_num=100000):
    """
    Вероятности состояний системы
    """
    p0 = get_p0(l, mu, gamma, tol)
    ps = []
    ps.append(p0)

    for i in range(1, max_num):
        chisl = math.pow(l, i)
        znam = mu
        j = 1
        while j < i:
            znam *= (mu + j * gamma)
            j += 1

        pi = p0 * chisl / znam
        ps.append(pi)
        if pi < tol:
            break

    return ps


def get_N(l, mu, gamma, tol=1e-12):
    """
    Среднее число заявок в системе
    """
    ps = get_p(l, mu, gamma, tol)
    N = 0
    for i, p in enumerate(ps):
        N += i * p

    return N


def get_Q(l, mu, gamma, tol=1e-12):
    """
    Среднее число заявок в очереди
    """
    ps = get_p(l, mu, gamma, tol)
    Q = 0
    for i, p in enumerate(ps):
        if i != 0:
            Q += (i - 1) * p

    return Q


def get_w1(l, mu, gamma, tol=1e-12):
    """
    Среднее время ожидания
    """
    # print(mu * (1.0 - get_p0(l, mu, gamma)) + gamma * get_Q(l, mu, gamma))
    return get_Q(l, mu, gamma, tol) / l


def get_v1(l, mu, gamma, tol=1e-12):
    """
    Среднее время пребывания
    """
    return get_N(l, mu, gamma, tol) / l


if __name__ == '__main__':
    l = 0.7
    mu = 0.9
    gamma = 0.2

    print(get_w1(l, mu, gamma))
