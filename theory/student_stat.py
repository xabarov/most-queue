import math


def get_ty(gamma, n):
    """
    :param gamma: доверительная вероятность
    :param n: объем выбоки
    :return: Табличный параметр ty
    """

    if n < 5:
        print('Student calc error. Param "n" must be greater then 5!')
        return 0

    gammas = [0.95, 0.99, 0.999]

    if not gammas.__contains__(gamma):
        print('Student calc error. Param "gamma" must take one of the following values: 0.95, 0.99, 0.999!')
        return 0

    gamma_095 = [2.78, 2.57, 2.45, 2.37, 2.31,
                 2.26, 2.23, 2.20, 2.18, 2.16,
                 2.15, 2.13, 2.12, 2.11, 2.10,
                 2.093, 2.064, 2.045, 2.032, 2.023,
                 2.016, 2.009, 2.001, 1.996, 1.001,
                 1.987, 1.984, 1.980, 1.960]
    gamma_099 = [4.60, 4.03, 3.71, 3.50, 3.36,
                 3.25, 3.17, 3.11, 3.06, 3.01,
                 2.98, 2.95, 2.92, 2.90, 2.88,
                 2.861, 2.797, 2.756, 2.720, 2.708,
                 2.692, 2.679, 2.662, 2.649, 2.640,
                 2.633, 2.627, 2.617, 2.576]
    gamma_0999 = [8.61, 6.86, 5.96, 5.41, 5.04,
                  4.78, 4.59, 4.44, 4.32, 4.22,
                  4.14, 4.07, 4.02, 3.97, 3.92,
                  3.883, 3.745, 3.659, 3.600, 3.558,
                  3.527, 3.502, 3.464, 3.439, 3.418,
                  3.403, 3.392, 3.374, 3.291]

    if n >= 5 and n < 20:
        if gamma == 0.95:
            return gamma_095[n - 5]
        if gamma == 0.99:
            return gamma_099[n - 5]
        if gamma == 0.999:
            return gamma_0999[n - 5]

    elif n >= 20 and n < 23:
        if gamma == 0.95:
            return gamma_095[16]
        if gamma == 0.99:
            return gamma_099[16]
        if gamma == 0.999:
            return gamma_0999[16]

    elif n >= 23 and n < 28:
        if gamma == 0.95:
            return gamma_095[17]
        if gamma == 0.99:
            return gamma_099[17]
        if gamma == 0.999:
            return gamma_0999[17]

    elif n >= 28 and n < 33:
        if gamma == 0.95:
            return gamma_095[18]
        if gamma == 0.99:
            return gamma_099[18]
        if gamma == 0.999:
            return gamma_0999[18]

    elif n >= 33 and n < 38:
        if gamma == 0.95:
            return gamma_095[19]
        if gamma == 0.99:
            return gamma_099[19]
        if gamma == 0.999:
            return gamma_0999[19]

    elif n >= 38 and n < 43:
        if gamma == 0.95:
            return gamma_095[20]
        if gamma == 0.99:
            return gamma_099[20]
        if gamma == 0.999:
            return gamma_0999[20]

    elif n >= 43 and n < 48:
        if gamma == 0.95:
            return gamma_095[21]
        if gamma == 0.99:
            return gamma_099[21]
        if gamma == 0.999:
            return gamma_0999[21]

    elif n >= 48 and n < 56:
        if gamma == 0.95:
            return gamma_095[22]
        if gamma == 0.99:
            return gamma_099[22]
        if gamma == 0.999:
            return gamma_0999[22]

    elif n >= 56 and n < 66:
        if gamma == 0.95:
            return gamma_095[23]
        if gamma == 0.99:
            return gamma_099[23]
        if gamma == 0.999:
            return gamma_0999[23]

    elif n >= 66 and n < 76:
        if gamma == 0.95:
            return gamma_095[23]
        if gamma == 0.99:
            return gamma_099[23]
        if gamma == 0.999:
            return gamma_0999[23]

    elif n >= 76 and n < 86:
        if gamma == 0.95:
            return gamma_095[24]
        if gamma == 0.99:
            return gamma_099[24]
        if gamma == 0.999:
            return gamma_0999[24]

    elif n >= 86 and n < 96:
        if gamma == 0.95:
            return gamma_095[25]
        if gamma == 0.99:
            return gamma_099[25]
        if gamma == 0.999:
            return gamma_0999[25]

    elif n >= 96 and n < 111:
        if gamma == 0.95:
            return gamma_095[26]
        if gamma == 0.99:
            return gamma_099[26]
        if gamma == 0.999:
            return gamma_0999[26]

    elif n >= 111 and n <= 120:
        if gamma == 0.95:
            return gamma_095[27]
        if gamma == 0.99:
            return gamma_099[27]
        if gamma == 0.999:
            return gamma_0999[27]

    else:
        if gamma == 0.95:
            return gamma_095[28]
        if gamma == 0.99:
            return gamma_099[28]
        if gamma == 0.999:
            return gamma_0999[28]


def get_conf_intervals(gamma, n, mean, std):
    """
    :param gamma: доверительная вероятность
    :param n: объем выбоки
    :param mean: выборочное среднее
    :param std: "исправленноеэ СКО
    :return: доверительные итнервалы
    """
    ty = get_ty(gamma, n)
    fract = ty * std / math.sqrt(n)

    return mean - fract, mean + fract


if __name__ == '__main__':
    # Пример из учебника Гмурман. Стр. 217
    n = 16
    x = 20.2
    s = 0.8
    gamma = 0.95

    left, right = get_conf_intervals(gamma, n, x, s)

    print("{0:5.3f} < a < {1:5.3f}".format(left, right))
