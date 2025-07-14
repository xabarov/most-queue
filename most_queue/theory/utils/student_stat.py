"""
Student statistical functions
"""

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

    ty_data = {
        0.95: [
            2.78,
            2.57,
            2.45,
            2.37,
            2.31,
            2.26,
            2.23,
            2.20,
            2.18,
            2.16,
            2.15,
            2.13,
            2.12,
            2.11,
            2.10,
            2.093,
            2.064,
            2.045,
            2.032,
            2.023,
            2.016,
            2.009,
            2.001,
            1.996,
            1.001,
            1.987,
            1.984,
            1.980,
            1.960,
        ],
        0.99: [
            4.60,
            4.03,
            3.71,
            3.50,
            3.36,
            3.25,
            3.17,
            3.11,
            3.06,
            3.01,
            2.98,
            2.95,
            2.92,
            2.90,
            2.88,
            2.861,
            2.797,
            2.756,
            2.720,
            2.708,
            2.692,
            2.679,
            2.662,
            2.649,
            2.640,
            2.633,
            2.627,
            2.617,
            2.576,
        ],
        0.999: [
            8.61,
            6.86,
            5.96,
            5.41,
            5.04,
            4.78,
            4.59,
            4.44,
            4.32,
            4.22,
            4.14,
            4.07,
            4.02,
            3.97,
            3.92,
            3.883,
            3.745,
            3.659,
            3.600,
            3.558,
            3.527,
            3.502,
            3.464,
            3.439,
            3.418,
            3.403,
            3.392,
            3.374,
            3.291,
        ],
    }

    if gamma not in ty_data:
        print(
            'Student calc error.'
            'Param "gamma" must take one of the following values:'
            '0.95, 0.99, 0.999!'
        )
        return 0

    # Define ranges and their corresponding indices
    n_ranges = [
        (5, 20, lambda n_val: n_val - 5),  # For 5 <= n < 20, index is n - 5
        (20, 23, 16),
        (23, 28, 17),
        (28, 33, 18),
        (33, 38, 19),
        (38, 43, 20),
        (43, 48, 21),
        (48, 56, 22),
        (56, 76, 23),
        (76, 86, 24),
        (86, 96, 25),
        (96, 111, 26),
        (111, 121, 27),  # Changed upper bound to 121 to include 120
    ]

    for lower, upper, index_val in n_ranges:
        if lower <= n < upper:
            index = index_val if isinstance(index_val, int) else index_val(n)
            return ty_data[gamma][index]

    # If n is 121 or greater, return the last value
    return ty_data[gamma][28]


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


if __name__ == "__main__":
    # Пример из учебника Гмурман. Стр. 217
    N = 16
    X = 20.2
    S = 0.8
    GAMMA_TEST = 0.95

    left, right = get_conf_intervals(GAMMA_TEST, N, X, S)

    print(f"{left:5.3f} < a < {right:5.3f}")
