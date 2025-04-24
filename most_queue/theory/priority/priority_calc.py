import math

from most_queue.rand_distribution import GammaDistribution
from most_queue.theory.fifo.mg1_calc import MG1Calculation
from most_queue.theory.fifo.mgn_tt import MGnCalc
from most_queue.theory.utils.diff5dots import diff5dots


def get_w1_pr(l, b):
    """
    Расчет среднего времени ожидания в M/G/1 с абсолютным приоритетом
    :param l: [] - список интенсивностей для каждого класса заявок
    :param b: [k][j] - список начальных моментов для каждого класса
    """
    k = len(l)
    w1 = [0.0] * k
    R = [0.0] * k
    ro = [0.0] * k

    for j in range(k):
        ro[j] = l[j] * b[j][0]
        s = 0
        for i in range(j + 1):
            s += ro[i]
        R[j] = s
    for j in range(k):
        s = 0
        for i in range(j + 1):
            s += l[i] * b[i][1]
        if j == 0:
            w1[j] = s / (2 * (1 - R[j]))
        else:
            w1[j] = s / (2 * (1 - R[j]) * (1 - R[j - 1]))

        if j != 0:
            w1[j] += b[j][0] * R[j - 1] / (1 - R[j - 1])
    return w1


def get_w1_np(l, b):
    """
    Расчет среднего времени ожидания в M/G/1 с отнссительным приоритетом
    :param l: [k] список интенсивностей входящего потока для каждого класса k
    :param b: [k][j] список, k - класс заявки, j - номер начального момента времени обслуживания
    """
    k = len(l)
    w1 = [0.0] * k
    R = [0, 0] * k
    ro = [0.0] * k

    for j in range(k):
        ro[j] = l[j] * b[j][0]
        s = 0
        for i in range(j + 1):
            s += ro[i]
        R[j] = s
    for j in range(k):
        s = 0
        for i in range(k):
            s += l[i] * b[i][1]
        if j == 0:
            w1[j] = s / (2 * (1 - R[j]))
        else:
            w1[j] = s / (2 * (1 - R[j]) * (1 - R[j - 1]))
    return w1


def busy_calc(l, b, num=5):
    """
    Вычисляет начальные моменты периода непрерывной занятости для СМО M/G/1
    По умолчанию - первые три.
    :param l: - интенсивность вх. потока
    :param b: [j], j=1..num, начальные моменты времени обслуживания
    """
    num = min(num, len(b))
    pnz = []
    ro = l * b[0]
    pnz.append(b[0] / (1 - ro))
    z = 1 + l * pnz[0]
    if num > 1:
        pnz.append(b[1] / math.pow(1 - ro, 3))
    if num > 2:
        pnz.append(b[2] / math.pow(1 - ro, 4) + 3 *
                   l * b[1] * b[1] / math.pow(1 - ro, 5))
    if num > 3:
        chisl = b[3] * math.pow(z, 4) + 6 * b[2] * l * pnz[1] * z * z + b[1] * (
                3 * math.pow(l * pnz[1], 2) + 4 * l * pnz[2] * z)
        pnz.append(chisl / (1 - ro))
    if num > 4:
        chisl = b[4] * math.pow(z, 5) + 10 * b[3] * l * pnz[1] * math.pow(z, 3) + \
                b[2] * (15 * math.pow(l * pnz[1], 2) * z + 10 * l * pnz[2 * z * z]) + b[1] * (
            5 * l * pnz[3] * z + 10 * l * l * pnz[1] * pnz[2])
        pnz.append(chisl / (1 - ro))

    return pnz


def climov_w_pr_calc(l, b):
    # только два первых момента
    k_num = len(l)
    ro_k_j = []
    ro_k = []
    w = []
    for k in range(k_num):
        ro_k.append([])
        ro_k_j.append([0.0] * 3)
        for j in range(3):
            for i in range(k + 1):
                ro_k_j[k][j] += l[i] * b[i][j]
        ro_k[k] = 1 - ro_k_j[k][0]
    for k in range(k_num):
        w.append([0.0] * 2)
        if k != 0:
            w[k][0] = ro_k_j[k][1] / (2 * ro_k[k - 1] * ro_k[k])
            w[k][1] = ro_k_j[k][2] / (3 * ro_k[k]) + (ro_k_j[k][2]) / (
                2 * math.pow(ro_k[k], 2)) + \
                (ro_k_j[k][1]) / (2 * ro_k[k])

        else:
            w[k][0] = ro_k_j[k][1] / (2 * ro_k[k])
            w[k][1] = ro_k_j[k][2] / (3 * math.pow(ro_k[k - 1], 3) * ro_k[k]) + (ro_k_j[k][2] * ro_k_j[k - 1][1]) / (
                2 * math.pow(ro_k[k - 1], 2) * math.pow(ro_k[k], 2)) + \
                (ro_k_j[k][1] * ro_k_j[k - 1][1]) / \
                (2 * math.pow(ro_k[k - 1], 3) * ro_k[k])

    return w


def busy_calc_warm_up(l, f, pnz, num=5):
    """
    Вычисляет начальные моменты периода непрерывной занятости с разогревом для СМО M/G/1
    По умолчанию - первые три.
    l: - интенсивность вх. потока
    b: [j], j=1..num, начальные моменты времени обслуживания
    f: [j], нач. моменты распределения времени разогрева
    """
    num = min(num, len(f))

    pnz_warm_up = []
    z = 1 + l * pnz[0]
    pnz_warm_up.append(f[0] * z)
    if num > 1:
        pnz_warm_up.append(f[0] * l * pnz[1] + f[1] * z * z)
    if num > 2:
        pnz_warm_up.append(f[0] * l * pnz[2] + 3 * f[1] *
                           l * pnz[1] * z + f[2] * math.pow(z, 3))
    if num > 3:
        pnz_warm_up.append(f[0] * l * pnz[3] + f[1] * (3 * math.pow(l * pnz[1], 2) + 4 * l * pnz[2] * z)
                           + 6 * f[2] * l * pnz[1] * z * z + f[3] * math.pow(z, 4))
    if num > 4:
        pnz_warm_up.append(f[0] * l * pnz[4] + f[1] * (5 * l * pnz[3] * z + 10 * l * l * pnz[1] * pnz[2]) +
                           f[2] * (15 * math.pow(l * pnz[1], 2) * z + 10 * f[3] * l * pnz[1] * math.pow(z, 3) + f[
                               4] * math.pow(z, 5)))

    return pnz_warm_up


def calc_pr1(l, b, num=3):
    """
        Расчет нач. моментов времени пребывания, ожидания начала обслуживания без и с учетом прерываний,
        активного времени, ПНЗ в M/G/1 с абсолютным приоритетом
        :param l: [k] список интенсивностей входящего потока для каждого класса k
        :param b: [k][j] список, k - класс заявки, j - номер начального момента времени обслуживания
        res['v'][k][j] - нач моменты времени пребывания
        res['w'][k][j] - нач моменты времени ожидания до начала обслуживания
        res['h'][k][j] - нач моменты активного времени
        res['pnz'][k][j] - нач моменты ПНЗ
        res['w_with_pr'][k][j] - нач моменты времени ожидания с учетом прерываний
    """
    num_of_cl = len(l)
    L = []
    for i in range(num_of_cl):
        summ = 0
        for j in range(i + 1):
            summ += l[j]
        L.append(summ)

    pi_j_i = []
    pi_j_i.append([])
    pi_j = []
    w = []
    v = []
    h = []

    pi_j.append(busy_calc(l[0], b[0]))

    # Формула Полячека - Хинчина. Заявки первого
    # класса не прерываются
    w.append([0.0] * num)

    for i in range(num):
        summ = b[0][i + 1] / (i + 2)
        for s in range(1, i + 1):
            summ += b[0][i + 1 - s] * w[0][s] * \
                math.factorial(i + 1) / (math.factorial(s)
                                         * math.factorial(i + 2 - s))
        w[0][i] = summ * l[0] / (1 - l[0] * b[0][0])

    v.append([0.0] * num)
    v[0][0] = w[0][0] + b[0][0]
    v[0][1] = w[0][1] + 2 * w[0][0] * b[0][0] + b[0][1]
    if num > 2:
        v[0][2] = w[0][2] + 3 * w[0][1] * \
            b[0][0] + 3 * w[0][0] * b[0][1] + b[0][2]

    h.append(b[0])

    for j in range(1, num_of_cl):
        pi_j.append([0.0] * (num + 1))
        h.append(busy_calc_warm_up(L[j - 1], b[j], pi_j[j - 1]))

        pi_j_i.append([])
        for k in range(j + 1):
            pi_j_i[j].append([])

        pi_j_i[j][j] = busy_calc(l[j], h[j])

        for i in range(j):
            if j == 1:
                pi_j_i[j][i] = busy_calc_warm_up(l[j], pi_j[0], pi_j_i[j][j])
            else:
                pi_j_i[j][i] = busy_calc_warm_up(
                    l[j], pi_j_i[j - 1][i], pi_j_i[j][j])

        for moment in range(num + 1):
            summ = 0
            for i in range(j + 1):
                summ += l[i] * pi_j_i[j][i][moment]
            pi_j[j][moment] = summ / L[j]

        w.append([0.0] * (num + 1))
        w[j][0] = 1
        v.append([0.0] * num)

        c = (1.0 - l[j] * h[j][0]) / (1.0 + L[j - 1] * pi_j[j - 1][0])
        for i in range(1, num + 1):
            summ = 0
            for m in range(i):
                summ += w[j][m] * h[j][i - m] * \
                    math.factorial(i) / (math.factorial(m) *
                                         math.factorial(i + 1 - m))

            w[j][i] = (c * L[j] * pi_j[j - 1][i] / (i + 1) +
                       l[j] * summ) / (1.0 - l[j] * h[j][0])
        w[j] = w[j][1:]
        v[j][0] = w[j][0] + h[j][0]
        v[j][1] = w[j][1] + 2 * w[j][0] * h[j][0] + h[j][1]
        if num > 2:
            v[j][2] = w[j][2] + 3 * w[j][1] * \
                h[j][0] + 3 * w[j][0] * h[j][1] + h[j][2]

    res = {}
    res['v'] = v
    res['w'] = w
    res['h'] = h
    w_with_pr = []
    for j in range(num_of_cl):
        w_with_pr.append([0.0] * 3)
        w_with_pr[j][0] = v[j][0] - b[j][0]
        w_with_pr[j][1] = v[j][1] - 2 * w_with_pr[j][0] * b[j][0] - b[j][1]
        if num > 2:
            w_with_pr[j][2] = v[j][2] - 3 * w_with_pr[j][1] * \
                b[j][0] - 3 * w_with_pr[j][0] * b[j][1] - b[j][2]
    res['w_with_pr'] = w_with_pr
    res['pnz'] = pi_j

    return res


def plsH2(param, s):
    y1 = param[0]
    y2 = 1.0 - y1
    mu1 = param[1]
    mu2 = param[2]

    return y1 * mu1 / (s + mu1) + y2 * mu2 / (s + mu2)


def get_w_mg1_bp(l, b):
    """
    Расчет нач. моментов времен ожидания для одноканальной СМО, классы без приоритетов
    """
    l_sum = 0
    for l_i in l:
        l_sum += l_i

    b_sr = []
    num_of_moment = len(b[0])
    num_of_classes = len(b)
    for i in range(num_of_moment):
        b_sr.append(0)
        for k in range(num_of_classes):
            b_sr[i] += b[k][i]
        b_sr[i] /= num_of_classes

    mg1_num = MG1Calculation(l_sum, b_sr)
    w_k = mg1_num.get_w()

    w = []
    for k in range(num_of_classes):
        w.append(w_k)

    return w


def get_v_mg1_bp(l, b):
    w = get_w_mg1_bp(l, b)
    num_of_classes = len(b)
    num_of_moment = len(b[0]) - 1

    v = []

    for k in range(num_of_classes):
        v.append([])
        v[k].append(w[k][0] + b[k][0])
        if num_of_moment > 1:
            v[k].append(w[k][1] + 2 * w[k][0] * b[k][0] + b[k][1])
        if num_of_moment > 2:
            v[k].append(w[k][2] + 3 * w[k][1] * b[k][0] +
                        3 * w[k][0] * b[k][1] + b[k][2])

    return v


def get_v_np(l, b, num=3):
    k = len(l)
    v = []
    w = get_w_np(l, b, num)

    for i in range(k):
        v.append([])
        v[i].append(w[i][0] + b[i][0])
        if num > 1:
            v[i].append(w[i][1] + 2 * w[i][0] * b[i][0] + b[i][1])
        if num > 2:
            v[i].append(w[i][2] + 3 * w[i][1] * b[i][0] +
                        3 * w[i][0] * b[i][1] + b[i][2])

    return v


def get_w_np(l, b, num=3):
    """
    Расчет начальных моментов времени ожидания в M/G/1 с относительным приоритетом
    :param l: [] - список интенсивностей для каждого класса заявок
    :param b: [k][j] - список начальных моментов для каждого класса
    """
    # a - lower pr
    # j - the same
    # e - higher pr

    num_of_cl = len(l)
    w = []
    ro = 0
    L = []
    for i in range(num_of_cl):
        ro += l[i] * b[i][0]
        summ = 0
        for s in range(i + 1):
            summ += l[s]
        L.append(summ)

    for j in range(num_of_cl):
        w.append([])
        w[j] = [0.0] * (len(b[j]) - 1)

        la = 0
        for i in range(j):
            la += l[i]

        lb = 0
        for i in range(j + 1, num_of_cl):
            lb += l[i]

        b_i = b[j]
        num_of_mom = len(b_i)

        b_a = [0.0] * num_of_mom
        for m in range(num_of_mom):
            if j == 0:
                b_a[m] = 0
            else:
                summ = 0
                for i in range(j):
                    summ += l[i] * b[i][m]
                b_a[m] = summ / la

        b_b = [0.0] * num_of_mom
        for m in range(num_of_mom):
            if j == num_of_cl - 1:
                b_b[m] = 0
            else:
                summ = 0
                for i in range(j + 1, num_of_cl):
                    summ += l[i] * b[i][m]
                b_b[m] = summ / lb

        h = 0.0001
        steps = 5

        if j != num_of_cl - 1:
            b_b_param = GammaDistribution.get_params(b_b)
        else:
            b_b_param = 0

        b_k_param = GammaDistribution.get_params(b[j])

        nu_a_PNZ = busy_calc(la, b_a)

        if j != 0:
            nu_a_param = GammaDistribution.get_params(nu_a_PNZ)
        else:
            nu_a_param = 0

        w_pls = []

        for c in range(1, steps):
            s = h * c

            if j != 0:
                nu_a = GammaDistribution.get_pls(nu_a_param, s)
                summ = s + la - la * nu_a
            else:
                summ = s

            chisl = (1 - ro) * summ

            if j != len(l) - 1:
                chisl += lb * (1 - GammaDistribution.get_pls(b_b_param, summ))

            znam = l[j] * \
                GammaDistribution.get_pls(b_k_param, summ) - l[j] + s

            w_pls.append(chisl / znam)

            w[j] = diff5dots(w_pls, h)
            w[j][0] = -w[j][0]
            if len(b[j]) > 2:
                w[j][2] = -w[j][2]

    return w


def get_w_prty_invar(l, b, n, type='NP', N=150, num=3):
    """
    Аппроксимация нач моментов времени ожидания для многоканальной СМО с приоритетами
    на основе инвариантов отношения M*|G*|n = M*|G*|1 * (M|G|n / M|G|1)
    :param l: список интенсивностей вх потока l[k], k - номер класса
    :param b: b[k][j] - нач. моменты времени обслуживания, j - номер момента
    :param n: число каналов
    :param type: тип приоритета - "PR", "NP"
    :param N: число ярусов для метода Такахаси-Таками, также число вероятностей,
    вычисляемых для M/G/1
    :return: w[k][j] - нач моменты времени ожидания по всем классам
    """
    w = []
    k_num = len(l)
    j_num = len(b[0])

    for k in range(k_num):
        w.append([0.0] * num)

    # M*/G*/1 PRTY calculation:
    b1 = [] * k_num
    for k in range(k_num):
        b1.append([0.0] * j_num)

    for k in range(k_num):
        for j in range(j_num):
            b1[k][j] = b[k][j] / math.pow(n, j + 1)

    if type == 'NP':
        w1_prty = get_w_np(l, b1, num=num)
    elif type == 'PR':
        pr_prty_calc = calc_pr1(l, b1, num=num)
        w1_prty = pr_prty_calc['w_with_pr']
    else:
        return 'Wrong PRTY type. Should be "PR" or "NP"'

    # M/G/1 calculation:

    b_sr = [0.0] * j_num
    l_sum = 0
    for j in range(j_num):
        for k in range(k_num):
            b_sr[j] += b1[k][j]
        b_sr[j] /= k_num
    for k in range(k_num):
        l_sum += l[k]

    mg1_num = MG1Calculation(l_sum, b_sr)
    p1 = mg1_num.get_p(N)
    q1 = 0
    for i in range(1, N):
        q1 += (i - 1) * p1[i]

    # M/G/n calculation:

    b_sr = [0.0] * j_num
    l_sum = 0
    for j in range(j_num):
        for k in range(k_num):
            b_sr[j] += b[k][j]
        b_sr[j] /= k_num
    for k in range(k_num):
        l_sum += l[k]

    tt_n = MGnCalc(n, l_sum, b_sr)
    tt_n.run()
    p_n = tt_n.get_p()
    qn = 0
    for i in range(n + 1, N):
        qn += (i - n) * p_n[i]

    for k in range(k_num):
        for j in range(num):
            w[k][j] = w1_prty[k][j] * qn / q1

    return w


def get_v_prty_invar(l, b, n, type='NP', num=3):
    w = get_w_prty_invar(l, b, n, type, num=num)
    v = []
    k = len(l)
    for i in range(k):
        v.append([0.0] * num)
        v[i][0] = w[i][0] + b[i][0]
        if num > 1:
            v[i][1] = w[i][1] + 2 * w[i][0] * b[i][0] + b[i][1]
        if num > 2:
            v[i][2] = w[i][2] + 3 * w[i][1] * \
                b[i][0] + 3 * w[i][0] * b[i][1] + b[i][2]
    return v


def get_w_MxGn_no_pr(l, b, n):
    l_sum = sum(l)
    b_sr = []
    num_of_mom = len(b[0])
    for j in range(num_of_mom):
        b_sr.append(0)
        for k in range(len(b)):
            b_sr[j] += b[k][j]
        b_sr[j] /= num_of_mom

    tt = MGnCalc(n, l_sum, b_sr)
    tt.run()
    w = []
    w_k = tt.get_w()
    for i in range(len(b)):
        w.append(w_k)
    return w


def get_v_MxGn_no_pr(l, b, n):
    num_of_class = len(b)
    w = get_w_MxGn_no_pr(l, b, n)
    v = []
    for i in range(num_of_class):
        v.append([])
        v[i].append(b[i][0] + w[i][0])
        v[i].append(b[i][1] + 2 * b[i][0] * w[i][0] + w[i][1])
        v[i].append(b[i][2] + 3 * b[i][1] * w[i][0] +
                    3 * b[i][0] * w[i][1] + w[i][2])

    return v
