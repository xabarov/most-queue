def get_moments(a, b, num=3):
    num = min(num, 3)
    res = [0] * num
    res[0] = a[0] + b[0]
    res[1] = a[1] + b[1] + 2 * a[0] * b[0]
    res[2] = a[2] + b[2] + 3 * a[1] * b[0] + 3 * a[0] * b[1]
    return res


def get_self_concolution(b, n_times, num=3):
    num = min(num, 3)
    res = [0] * num
    for i in range(n_times):
        res = get_moments(res, b, num)
    return res

def get_moments_minus(a, b, num=3):
    num = min(num, 3)
    res = [0] * num
    res[0] = a[0] - b[0]
    res[1] = a[1] - b[1] - 2 * a[0] * b[0]
    res[2] = a[2] - b[2] - 3 * a[1] * b[0] - 3 * a[0] * b[1]
    return res