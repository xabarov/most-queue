import math


def get_q_Gamma(l, mu, alpha, num=100):
    q = [0.0] * num
    q[0] = math.pow(mu / (mu + l), alpha)
    for j in range(1, num):
        q[j] = q[j - 1] * l * (alpha + j - 1) / ((l + mu) * j)

    return q
