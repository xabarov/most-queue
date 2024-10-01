from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.rand_distribution import Gamma


def get_v(l, b, b_warm):
    tv = b_warm[0] / (1 - l * b[0])
    p0_star = 1 / (1 + l * tv)

    b_param = Gamma.get_mu_alpha(b)
    b_warm_param = Gamma.get_mu_alpha(b_warm)

    h = 0.0001
    steps = 5

    v_pls = []

    for c in range(1, steps):
        s = h * c
        chisl = p0_star * ((1 - s / l) * Gamma.get_pls(*b_warm_param, s) - Gamma.get_pls(*b_param, s))
        znam = 1 - s / l - Gamma.get_pls(*b_param, s)
        v_pls.append(chisl / znam)

    v = diff5dots(v_pls, h)
    v[0] = -v[0]
    v[2] = -v[2]

    return v


