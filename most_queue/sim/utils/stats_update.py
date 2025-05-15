import math


def refresh_moments_stat(moments, new_a, count):
    """
    Updating statistics of the moments
    moments: list[moment: float]
    new_a: float, new value
    count: how many events of calcs
    """

    for i in range(3):
        moments[i] = moments[i] * (1.0 - (1.0 / count)) + \
            math.pow(new_a, i + 1) / count

    return moments
