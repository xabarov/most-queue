from most_queue.sim import rand_destribution as rd
from diff5dots import diff5dots
from most_queue.sim import smo_im


def get_v(l, b, b_warm):
    tv = b_warm[0] / (1 - l * b[0])
    p0_star = 1 / (1 + l * tv)

    b_param = rd.Gamma.get_mu_alpha(b)
    b_warm_param = rd.Gamma.get_mu_alpha(b_warm)

    h = 0.0001
    steps = 5

    v_pls = []

    for c in range(1, steps):
        s = h * c
        chisl = p0_star * ((1 - s / l) * rd.Gamma.get_pls(*b_warm_param, s) - rd.Gamma.get_pls(*b_param, s))
        znam = 1 - s / l - rd.Gamma.get_pls(*b_param, s)
        v_pls.append(chisl / znam)

    v = diff5dots(v_pls, h)
    v[0] = -v[0]
    v[2] = -v[2]

    return v


if __name__ == '__main__':
    from most_queue.utils.tables import times_print, probs_print

    l = 1
    b1 = 0.8
    b1_warm = 0.9
    coev = 1.3
    b_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.H2_dist.calc_theory_moments(*b_params, 4)

    b_warm_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b_warm = rd.H2_dist.calc_theory_moments(*b_params, 4)

    smo = smo_im.SmoIm(1)
    smo.set_servers(b_params, "H")
    smo.set_warm(b_warm_params, "H")
    smo.set_sources(l, "M")
    smo.run(100000)

    v_ch = get_v(l, b, b_warm)
    v_im = smo.v

    times_print(v_im, v_ch, False)
