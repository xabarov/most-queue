"""
Functions for calculating convolutions of distributions.
"""


def get_moments(a: list[float], b: list[float], num: int = 3) -> list[float]:
    """
    Calculates the moments of the convolution of two distributions.
    :param a: Moments of the first distribution.
    :param b: Moments of the second distribution.
    :param num: Number of moments to calculate.
    :return: List of moments.
    """
    num = min(num, 3)
    res = [0] * num
    res[0] = a[0] + b[0]
    res[1] = a[1] + b[1] + 2 * a[0] * b[0]
    res[2] = a[2] + b[2] + 3 * a[1] * b[0] + 3 * a[0] * b[1]
    return res


def get_self_conv_moments(b: list[float], n_times, num: int = 3) -> list[float]:
    """
    Calculates the moments of the self-convolution of a distribution.
    :param b: Moments of the distribution.
    :param n_times: Number of times to self-convolve.
    :param num: Number of moments to calculate.
    :return: List of moments.
    """
    num = min(num, 3)
    res = [0] * num
    for i in range(n_times):
        res = get_moments(res, b, num)
    return res


def get_moments_minus(a: list[float], b: list[float], num: int = 3) -> list[float]:
    """
    b = conv(a, x)
    Calculates the moments of x from the moments of a and b.
    :param a: Moments of the first distribution.
    :param b: Moments of the second distribution.
    :param num: Number of moments to calculate.
    :return: List of moments.
    """
    num = min(num, 3)
    res = [0] * num
    res[0] = a[0] - b[0]
    res[1] = a[1] - b[1] - 2 * a[0] * b[0]
    res[2] = a[2] - b[2] - 3 * a[1] * b[0] - 3 * a[0] * b[1]
    return res
