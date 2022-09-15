from most_queue.theory.fj_calc import *


def test():
    n = [x for x in range(2, 15)]
    mu = 1.0
    b = [1 / mu, 2 / pow(mu, 2), 6 / pow(mu, 3)]

    ro = 0.8

    l = ro / b[0]
    num_of_jobs = 1000000
    v_im = []
    v_inv = []
    v_varma = []
    v_varki = []
    v_nelson = []

    str_f = "{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}|{4:^15s}"
    str_f_v = "{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}|{4:^15.3f}"

    print(str_f.format("ИМ", "Инвар", "Varki", "Varma", "Nelson"))
    i = 0
    for nn in n:
        smo = SmoFJ(nn, nn, False)

        smo.set_sources(l, 'M')
        smo.set_servers(mu, 'M')
        smo.run(num_of_jobs)
        v = smo.v
        v1_inv = get_v1_fj_invar(l, mu, nn, 100)
        v_im.append(v[0])
        v_inv.append(v1_inv)
        v_varki.append(get_v1_fj_varki_merchant(l, mu, nn))
        v_varma.append(get_v1_fj_varma(l, mu, nn))
        v_nelson.append(get_v1_fj_nelson_tantawi(l, mu, nn))
        print(str_f_v.format(v_im[i], v_inv[i], v_varki[i], v_varma[i], v_nelson[i]))
        i += 1

    # v1

    fig, ax = plt.subplots()

    tek_style = 0

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    ax.plot(n, v_im, label="ИМ", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_inv, label="Инвар", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_varki, label="Varki", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_varma, label="Varma", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_nelson, label="Nelson", linestyle=linestyles[tek_style])

    ax.set_ylabel("v1")
    ax.set_xlabel("n")

    plt.legend()
    plt.savefig("v1_invar_from_n_with_ro = {0:^4.2f}.png".format(ro), dpi=300)

    # errors
    v_inv_err = []
    v_nelson_err = []
    v_varma_err = []
    v_varki_err = []

    for i in range(len(v_inv)):
        v_inv_err.append(100 * (v_inv[i] - v_im[i]) / v_im[i])
        v_varma_err.append(100 * (v_varma[i] - v_im[i]) / v_im[i])
        v_varki_err.append(100 * (v_varki[i] - v_im[i]) / v_im[i])
        v_nelson_err.append(100 * (v_nelson[i] - v_im[i]) / v_im[i])

    fig, ax = plt.subplots()
    tek_style = 0
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    ax.plot(n, v_inv_err, label="Инвар", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_varki_err, label="Varki", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_varma_err, label="Varma", linestyle=linestyles[tek_style])
    tek_style += 1
    if tek_style == 4:
        tek_style = 0
    ax.plot(n, v_nelson_err, label="Nelson", linestyle=linestyles[tek_style])

    ax.set_ylabel("error, %")
    ax.set_xlabel("n")

    plt.legend()
    plt.savefig("error_invar_from_n_with_ro = {0:^4.2f}.png".format(ro), dpi=300)


if __name__ == "__main__":
    test()
