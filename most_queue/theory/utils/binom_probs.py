from scipy import special


def calc_binom_probs(size, y1):
    p = y1
    q = 1.0 - y1
    probs = []
    for i in range(size):
        probs.append(special.comb(size - 1, size - 1 - i) * pow(p, size - 1 - i) * pow(q, i))
    return probs


if __name__ == '__main__':
    print(calc_binom_probs(4, 0.4))
