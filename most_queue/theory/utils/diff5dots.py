"""
Numerical differentiation of a function using the five-point formula
"""


def diff5dots(f: list[float], h: float):
    """
    Numerical differentiation of a function using the five-point formula
    :param f: list of function values at points x_0, x_1, ..., x_n
    :param h: step size
    :return: array of three values - first, second, and third derivatives
    """
    res = [0] * 3

    count = len(f)

    if count == 3:
        res[0] = (1.0 / (2.0 * h)) * (-f[2] + 4.0 * f[1] - 3.0 * f[0])
        res[1] = (1.0 / (h * h)) * (f[2] - 2.0 * f[1] + f[0])
        res[2] = 0
    elif count == 4:
        res[0] = (1.0 / (6.0 * h)) * (2.0 * f[3] - 9.0 * f[2] + 18.0 * f[1] - 11.0 * f[0])
        res[1] = (1.0 / (h * h)) * (-f[3] + 4.0 * f[2] - 5.0 * f[1] + 2.0 * f[0])
        res[2] = (1.0 / (h * h * h)) * (f[3] - 3.0 * f[2] + 3.0 * f[1] - f[0])
    elif count == 5:
        res[0] = (1.0 / (12.0 * h)) * (-3.0 * f[4] + 16.0 * f[3] - 36.0 * f[2] + 48.0 * f[1] - 25.0 * f[0])
        res[1] = (1.0 / (12.0 * h * h)) * (11.0 * f[4] - 56.0 * f[3] + 114.0 * f[2] - 104.0 * f[1] + 35.0 * f[0])
        res[2] = (1.0 / (2.0 * h * h * h)) * (-3.0 * f[4] + 14.0 * f[3] - 24.0 * f[2] + 18.0 * f[1] - 5.0 * f[0])
    else:
        for i in range(3):
            res[i] = 0

    return res
