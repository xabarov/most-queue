"""
Utilities to update statistics
"""


def refresh_moments_stat(moments, new_a, count):
    """
    Updating statistics of the moments
    moments: list[moment: float]
    new_a: float, new value
    count: how many events of calcs
    """
    # Optimize: use power accumulation instead of math.pow for each iteration
    power = new_a  # new_a^1
    inv_count = 1.0 / count
    one_minus_inv_count = 1.0 - inv_count

    for i in range(len(moments)):
        moments[i] = moments[i] * one_minus_inv_count + power * inv_count
        power *= new_a  # Accumulate: new_a^(i+2) for next iteration

    return moments
