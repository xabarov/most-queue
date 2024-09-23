from theory import mmnr_calc
from theory import mgn_tt
from theory import mg1_calc

import matplotlib.pyplot as plt
from sim import rand_destribution as rd
import math
import time

from theory.convolution_sum_calc import get_moments, get_self_concolution


def get_lambda(min, max):
    l = np.random.randn()
    while l < min or l > max: \
            l = np.random.randn()
    return l


def get_1ambda_max(b, n, ro_max=0.8):
    b1_max = getMaxMoments(n, b, len(b))[0]
    return ro_max / b1_max


def calc_error_percentage(real_val, est_val):
    max_val = max(real_val, est_val)
    return 100 * math.fabs(real_val - est_val) / max_val


def getMaxMoments(n, b, num=None):
    """
    Расчет начальных моментов максимума СВ.
    :param n: число одинаково распределенных СВ
    :param b: начальные моменты СВ
    :param num: число выходных нач. моментов максимума СВ, по умолчанию - на один меньше числа начальных моментов b
    :return: начальные моменты максимума СВ.
    """
    if num:
        num = min(len(b), num)
    else:
        num = len(b)

    f = [0] * num
    variance = b[1] - b[0] * b[0]
    coev = math.sqrt(variance) / b[0]
    a_big = [1.37793470540E-1, 7.29454549503E-1, 1.808342901740E0,
             3.401433697855E0, 5.552496140064E0, 8.330152746764E0,
             1.1843785837900E1, 1.6279257831378E1, 2.1996585811981E1,
             2.9920697012274E1]
    g = [3.08441115765E-1, 4.01119929155E-1, 2.18068287612E-1,
         6.20874560987E-2, 9.50151697518E-3, 7.53008388588E-4,
         2.82592334960E-5, 4.24931398496E-7, 1.83956482398E-9,
         9.91182721961E-13]

    if len(b) >= 3:

        if coev < 1:
            params = rd.Erlang_dist.get_params(b)

            for j in range(10):
                p = g[j] * dfr_Erl_Mult(params, a_big[j], n) * math.exp(a_big[j])
                f[0] += p
                for i in range(1, num):
                    p = p * a_big[j]
                    f[i] += p
        else:
            params = rd.H2_dist.get_params(b)

            for j in range(10):
                p = g[j] * dfr_H2_Mult(params, a_big[j], n) * math.exp(a_big[j])
                f[0] += p
                for i in range(1, num):
                    p = p * a_big[j]
                    f[i] += p
    else:
        params = rd.Gamma.get_mu_alpha(b)

        for j in range(10):
            p = g[j] * dfr_Gamma_Mult(params, a_big[j], n) * math.exp(a_big[j])
            f[0] += p
            for i in range(1, num):
                p = p * a_big[j]
                f[i] += p

    for i in range(num - 1):
        f[i + 1] *= (i + 2)
    return f


def getMaxMomentsDelta(n, b, num=3, delta=0):
    """
        Расчет начальных моментов максимума СВ с учетом задержки delta.
        :param n: число одинаково распределенных СВ
        :param b: начальные моменты СВ
        :param num: число нач. моментов СВ
        :return: начальные моменты максимума СВ.
        """
    f = [0] * num
    a_big = [1.37793470540E-1, 7.29454549503E-1, 1.808342901740E0,
             3.401433697855E0, 5.552496140064E0, 8.330152746764E0,
             1.1843785837900E1, 1.6279257831378E1, 2.1996585811981E1,
             2.9920697012274E1]
    g = [3.08441115765E-1, 4.01119929155E-1, 2.18068287612E-1,
         6.20874560987E-2, 9.50151697518E-3, 7.53008388588E-4,
         2.82592334960E-5, 4.24931398496E-7, 1.83956482398E-9,
         9.91182721961E-13]

    if delta:
        params = rd.Gamma.get_mu_alpha(b)

        for j in range(10):
            p = g[j] * dfr_Gamma_Mult(params, a_big[j], n, delta) * math.exp(a_big[j])
            f[0] += p
            for i in range(1, num):
                p = p * a_big[j]
                f[i] += p

        for i in range(num - 1):
            f[i + 1] *= (i + 2)

    return f


# вспомогательные функции:

def dfr_H2_Mult(params, t, n, delta=None):
    res = 1.0
    if not delta:
        for i in range(n):
            res *= rd.H2_dist.get_cdf(params, t)
    else:
        if not isinstance(delta, list):
            for i in range(n):
                res *= rd.H2_dist.get_cdf(params, t - i * delta)
    return 1.0 - res


def dfr_Erl_Mult(params, t, n, delta=None):
    res = 1.0
    if not delta:
        for i in range(n):
            res *= rd.Erlang_dist.get_cdf(params, t)
    else:
        if not isinstance(delta, list):
            for i in range(n):
                res *= rd.Erlang_dist.get_cdf(params, t - i * delta)
    return 1.0 - res


def dfr_Gamma_Mult(params, t, n, delta=None):
    res = 1.0
    if not delta:
        for i in range(n):
            res *= rd.Gamma.get_cdf(*params, t)
    else:
        if not isinstance(delta, list):
            for i in range(n):
                res *= rd.Gamma.get_cdf(*params, t - i * delta)
        else:
            b = rd.Gamma.calc_theory_moments(*params)

            for i in range(n):
                b_delta = get_self_concolution(delta, i)
                b_summ = get_moments(b, b_delta)
                params_summ = rd.Gamma.get_mu_alpha(b_summ)
                res *= rd.Gamma.get_cdf(*params_summ, t)

    return 1.0 - res


def get_v1_fj2(l, mu):
    """
    Среднее время пребывания для СМО FJ с n=2
    :param ro: коэффициент загрузки
    :param mu: интенсивность обслуживания
    :param l: интенсивность вх потока
    """

    ro = l / mu
    H2 = 1.5
    v1_m = 1 / (mu - l)

    return (H2 - ro / 8) * v1_m


def get_v1_fj_varma(l, mu, n):
    ro = l / mu
    Vn = get_V(n)
    Hn = get_Hn(n)
    v1 = (Hn + (Vn - Hn) * ro) / (mu - l)
    return v1


def get_v1_fj_nelson_tantawi(l, mu, n):
    ro = l / mu
    Hn = get_Hn(n)
    v1 = (Hn / 1.5) + (4.0 / 11) * (1.0 - Hn / 1.5) * ro
    v1 *= (12 - ro) / (8 * (mu - l))
    return v1


def get_v1_fj_nelson_nk(l, mu, n, k):
    ro = l / mu
    res = 0
    coeff = (12 - ro) / (88 * mu * (1 - ro))

    if k == 1:
        res += n / (mu * (1 - ro))
        summ = 0

        for i in range(2, n + 1):
            summ += get_W(n, 1, i) * (11 * get_Hn(i) + 4 * ro * (get_Hn(2) - get_Hn(i))) / get_Hn(2)

        res += coeff * summ
    else:
        summ = 0
        for i in range(k, n + 1):
            summ += get_W(n, k, i) * (11 * get_Hn(i) + 4 * ro * (get_Hn(2) - get_Hn(i))) / get_Hn(2)
        res = coeff * summ

    return res


def get_V(n):
    Vn = 0
    for i in range(1, n + 1):
        elem = combination(n, i) * pow(-1, i - 1)
        summ2 = 0
        for j in range(1, i + 1):
            summ2 += combination(i, j) * factorial(j - 1) / pow(i, j + 1)
        elem *= summ2
        Vn += elem
    return Vn


def get_v1_varma_nk(l, mu, n, k):
    summ = 0
    ro = l / mu

    for i in range(k, n + 1):
        summ += get_W(n, k, i) * (get_Hn(i) + (get_V(i) - get_Hn(i) * ro)) / (l - mu)

    return summ


def get_A(n, k, i):
    if i == k:
        return 1
    else:
        summ = 0
        for j in range(1, i - k + 1):
            summ += combination(n - i + j, j) * get_A(n, k, i - j)

        return (-1) * summ


def get_W(n, k, i):
    summ = 0
    for j in range(k, i + 1):
        summ += combination(n, j) * get_A(n.j.i)

    return summ


def get_v1_fj_varki_merchant(l, mu, n):
    ro = l / mu
    Hn = get_Hn(n)
    summ = 0
    for i in range(1, n + 1):
        summ += 1 / (i - ro)

    summ2 = 0
    for i in range(1, n + 1):
        summ2 += 1 / (i * (i - ro))

    a = ro / (2 * (1 - ro))

    v1 = (1 / mu) * (Hn + a * (summ + (1 - 2 * ro) * summ2))

    return v1


def get_Hn(n):
    summ = 0
    for i in range(1, n + 1):
        summ += 1 / i
    return summ


def combination(n, i):
    return factorial(n) / (factorial(i) * factorial(n - i))


def factorial(n):
    if n == 0:
        return 1
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    return fact


def get_v1_fj_invar(l, mu, n, r=100):
    mu2 = mu / 2
    mu_n = mu / n

    v1_fj2 = get_v1_fj2(l, mu)
    v1_mmn = mmnr_calc.MMnr_calc.get_v(l, mu_n, n, r=r)[0]
    v1_mm2 = mmnr_calc.MMnr_calc.get_v(l, mu2, 2, r=r)[0]

    return v1_fj2 * v1_mmn / v1_mm2


def get_v1_fj_invar_sj(l, mu, n):
    mu2 = mu / 2
    mu_n = mu / n

    v1_fj2 = get_v1_fj2(l, mu)
    b1_sjn = getMaxMoments(n, [1 / mu_n, 2 / pow(mu_n, 2), 6 / pow(mu_n, 3)])
    b1_sj2 = getMaxMoments(n, [1 / mu2, 2 / pow(mu2, 2), 6 / pow(mu2, 3)])

    w_sjn = mg1_calc.get_w(l, b1_sjn)
    w_sj2 = mg1_calc.get_w(l, b1_sj2)

    return v1_fj2 * w_sjn[0] / w_sj2[0]


def get_v_fj_max(l, b, n, num_of_moments=3):
    # get v of individual branch:
    v_branch = mg1_calc.get_v(l, b, num_of_moments)
    v_max = getMaxMoments(n, v_branch, num_of_moments - 1)

    return v_max


def get_v_fj_invar_tt(l, b, n):
    v = []

    b1_2 = 2 * b[0]
    coev = math.sqrt(b[1] - b[0] * b[0]) / b[0]
    b2 = [0.0] * 3
    alpha = 1 / (coev ** 2)
    b2[0] = b1_2
    b2[1] = math.pow(b2[0], 2) * (math.pow(coev, 2) + 1)
    b2[2] = b2[1] * b2[0] * (1.0 + 2 / alpha)

    v1_fj2 = get_v1_fj2(l, 1.0 / b[0])

    tt2 = mgn_tt.MGnCalc(n, l, b2)
    tt2.run()
    w_tt2 = tt2.get_w()

    bn1 = b[0] * n
    bn = [0.0] * 3
    alpha = 1 / (coev ** 2)
    bn[0] = bn1
    bn[1] = math.pow(bn[0], 2) * (math.pow(coev, 2) + 1)
    bn[2] = bn[1] * bn[0] * (1.0 + 2 / alpha)

    ttn = mgn_tt.MGnCalc(n, l, bn)
    ttn.run()
    w_ttn = tt2.get_w()

    for i in range(3):
        v.append(v1_fj2 * w_ttn[i] / w_tt2[i])

    return v


def get_data(n_min=2, n_max=15, coev_min=0.2, coev_max=3, num_of_coevs=20,
             num_of_jobs=500000, moment=0):
    """
    Расчет зависимостей точности и времени моделирования от n и coev
    """

    coevs = np.linspace(coev_min, coev_max, num_of_coevs)
    ns = [x for x in range(n_min, n_max + 1)]

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    # зависимость от coev

    v1_im = []
    im_times = []
    v1_fj = []
    fj_times = []
    fj_errors = []

    # inicialize ml_errors:
    for nn in range(len(ns)):
        fj_errors.append([])
        fj_times.append([])
        for j in range(len(coevs)):
            fj_errors[nn].append(0)
            fj_times[nn].append(0)

    for nn in range(len(ns)):
        v1_im.append([])
        im_times.append([])
        v1_fj.append([])

        for j in range(len(coevs)):
            b = [0, 0, 0]
            b[0] = 1
            alpha = 1 / (coevs[j] ** 2)
            b[1] = math.pow(b[0], 2) * (math.pow(coevs[j], 2) + 1)
            b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
            l = get_1ambda_max(b, ns[nn])

            t_im_start = time.process_time()
            qs = ForkJoinSim(ns[nn], ns[nn], False)
            params = rd.Gamma.get_mu_alpha(b)
            qs.set_sources(l, 'M')
            qs.set_servers(params, 'Gamma')
            qs.run(num_of_jobs)
            v = qs.v
            v1_im[nn].append(qs.v[moment])
            im_times[nn].append(time.process_time() - t_im_start)

            t_fj_start = time.process_time()
            v_fj = get_v_fj_invar_tt(l, b, ns[nn])[moment]
            fj_times[nn][j] = time.process_time() - t_fj_start
            fj_errors[nn][j] = calc_error_percentage(v1_im[nn][j], v_fj)

        # make plots for error

        fig, ax = plt.subplots()

        tek_style = 0

        ax.plot(coevs, fj_errors[nn], label="Инвар", linestyle=linestyles[tek_style])
        tek_style += 1

        ax.set_ylabel("error, %")
        ax.set_xlabel("c")

        plt.legend()
        plt.savefig("invar_error_from_coev_with_n = " + str(ns[nn]) + ".png")

        # make plots for time

        fig, ax = plt.subplots()

        tek_style = 0

        ax.plot(coevs, fj_times[nn], label="Инвар", linestyle=linestyles[tek_style])
        tek_style += 1
        if tek_style == 4:
            tek_style = 0

        ax.plot(coevs, im_times[nn], label="ИМ")

        ax.set_ylabel("t, с")
        ax.set_xlabel("c")

        plt.legend()
        plt.savefig("time_from_coev_with_n = " + str(ns[nn]) + ".png")

        # make txt

        file = open("error_from_coev_with_n = " + str(ns[nn]) + ".txt", 'w')
        head = "coev;e;"
        head += "t_im;t_fj\n"

        file.write(head)
        size = len(coevs)
        for i in range(size):
            data_str = str(coevs[i]) + ";"
            data_str += str(fj_errors[nn][i]) + ";"

            data_str += str(im_times[nn][i]) + ";"

            data_str += str(fj_times[nn][i]) + "\n"

            file.write(data_str)

        file.close()

    # зависимость от coev

    v1_im = []
    im_times = []
    v1_fj = []
    fj_times = []
    fj_errors = []

    # inicialize ml_errors:
    for j in range(len(coevs)):
        fj_errors.append([])
        fj_times.append([])
        for nn in range(len(ns)):
            fj_errors[j].append(0)
            fj_times[j].append(0)

    for j in range(len(coevs)):
        v1_im.append([])
        im_times.append([])
        v1_fj.append([])

        for nn in range(len(ns)):
            b = [0, 0, 0]
            b[0] = 1
            alpha = 1 / (coevs[j] ** 2)
            b[1] = math.pow(b[0], 2) * (math.pow(coevs[j], 2) + 1)
            b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
            l = get_1ambda_max(b, ns[nn])

            t_im_start = time.process_time()
            qs = ForkJoinSim(ns[nn], ns[nn], False)
            params = rd.Gamma.get_mu_alpha(b)
            qs.set_sources(l, 'M')
            qs.set_servers(params, 'Gamma')
            qs.run(num_of_jobs)
            v = qs.v
            v1_im[nn].append(qs.v[moment])
            im_times[nn].append(time.process_time() - t_im_start)

            t_fj_start = time.process_time()
            v_fj = get_v_fj_invar_tt(l, b, ns[nn])[moment]
            fj_times[nn][j] = time.process_time() - t_fj_start
            fj_errors[nn][j] = calc_error_percentage(v1_im[nn][j], v_fj)

        # make plots for error

        fig, ax = plt.subplots()

        tek_style = 0

        ax.plot(ns, fj_errors[j], label="Инвар", linestyle=linestyles[tek_style])
        tek_style += 1

        ax.set_ylabel("error, %")
        ax.set_xlabel("c")

        plt.legend()
        plt.savefig("invar_error_from_n_with_coev = " + str(coevs[j]) + ".png")

        # make plots for time

        fig, ax = plt.subplots()

        tek_style = 0

        ax.plot(ns, fj_times[j], label="Инвар", linestyle=linestyles[tek_style])
        tek_style += 1
        if tek_style == 4:
            tek_style = 0

        ax.plot(ns, im_times[j], label="ИМ")

        ax.set_ylabel("t, с")
        ax.set_xlabel("c")

        plt.legend()
        plt.savefig("time_from_n_with_coev = " + str(coevs[j]) + ".png")

        # make txt

        file = open("error_from_n_with_coev = " + str(coevs[j]) + ".txt", 'w')
        head = "n;e;"
        head += "t_im;t_fj\n"

        file.write(head)
        size = len(ns)
        for i in range(size):
            data_str = str(ns[i]) + ";"
            data_str += str(fj_errors[j][i]) + ";"

            data_str += str(im_times[j][i]) + ";"

            data_str += str(fj_times[j][i]) + "\n"

            file.write(data_str)

        file.close()

