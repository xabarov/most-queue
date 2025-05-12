import math

def busy_calc(l:float, b:list[float], num:int=5):
    """
    Calculation of initial moments of continuous busy period for M/G/1 queue
    By default, the first five are calculated.
    :param l: - intensity of input stream
    :param b: [j], j=1..num, initial moments of service time
    :param num: number of initial moments to calculate
    :return: list of initial moments of continuous busy period for M/G/1 queue
    """
    num = min(num, len(b))
    busy_moments = []
    ro = l * b[0]
    busy_moments.append(b[0] / (1 - ro))
    z = 1 + l * busy_moments[0]
    if num > 1:
        busy_moments.append(b[1] / math.pow(1 - ro, 3))
    if num > 2:
        busy_moments.append(b[2] / math.pow(1 - ro, 4) + 3 *
                   l * b[1] * b[1] / math.pow(1 - ro, 5))
    if num > 3:
        chisl = b[3] * math.pow(z, 4) + 6 * b[2] * l * busy_moments[1] * z * z + b[1] * (
                3 * math.pow(l * busy_moments[1], 2) + 4 * l * busy_moments[2] * z)
        busy_moments.append(chisl / (1 - ro))
    if num > 4:
        chisl = b[4] * math.pow(z, 5) + 10 * b[3] * l * busy_moments[1] * math.pow(z, 3) + \
                b[2] * (15 * math.pow(l * busy_moments[1], 2) * z + 10 * l * busy_moments[2 * z * z]) + b[1] * (
            5 * l * busy_moments[3] * z + 10 * l * l * busy_moments[1] * busy_moments[2])
        busy_moments.append(chisl / (1 - ro))

    return busy_moments


def busy_calc_warm_up(l:float, f:list[float], busy_moments: list[float], num: int=5):
    """
    Calculate the initial moments of continuous busy period for M/G/1 queue with warm-up
    By default, the first three are calculated.
    :param l: - input flow intensity
    :param f: - initial service time moments
    :param busy_moments: - initial moments of busy period
    :param num: - number of moments to calculate
    :return: busy_moments_warm_up
    
    """
    num = min(num, len(f))

    busy_moments_warm_up = []
    z = 1 + l * busy_moments[0]
    busy_moments_warm_up.append(f[0] * z)
    if num > 1:
        busy_moments_warm_up.append(f[0] * l * busy_moments[1] + f[1] * z * z)
    if num > 2:
        busy_moments_warm_up.append(f[0] * l * busy_moments[2] + 3 * f[1] *
                           l * busy_moments[1] * z + f[2] * math.pow(z, 3))
    if num > 3:
        busy_moments_warm_up.append(f[0] * l * busy_moments[3] + f[1] * (3 * math.pow(l * busy_moments[1], 2) + 4 * l * busy_moments[2] * z)
                           + 6 * f[2] * l * busy_moments[1] * z * z + f[3] * math.pow(z, 4))
    if num > 4:
        busy_moments_warm_up.append(f[0] * l * busy_moments[4] + f[1] * (5 * l * busy_moments[3] * z + 10 * l * l * busy_moments[1] * busy_moments[2]) +
                           f[2] * (15 * math.pow(l * busy_moments[1], 2) * z + 10 * f[3] * l * busy_moments[1] * math.pow(z, 3) + f[
                               4] * math.pow(z, 5)))

    return busy_moments_warm_up
