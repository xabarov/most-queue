import math

import numpy as np

from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.priority_calc import get_v_prty_invar
from most_queue.rand_distribution import Gamma


def balance_equation(L, R):
    rows = R.shape[0]
    cols = R.shape[1]

    # определяем нулевые строки и столбцы

    null_numbers = []

    for i in range(cols):
        null_counts = 0
        for j in range(rows):
            if math.fabs(R[j, i] < 1e-6):
                null_counts += 1
        if null_counts == rows:
            null_numbers.append(i)

    # создаем новую матрицу без нулевых строк и столбцов

    rows_mod = rows - len(null_numbers)
    cols_mod = rows_mod

    R_mod = np.zeros((rows_mod, cols_mod))
    row_tek = 0
    col_tek = 0

    for i in range(rows):
        skip = False
        for n in null_numbers:
            if n + 1 == i:
                skip = True
        if skip:
            continue
        for j in range(cols):
            skip = False
            for n in null_numbers:
                if n == j:
                    skip = True
            if skip:
                continue
            R_mod[row_tek, col_tek] = R[i, j]
            col_tek += 1
        row_tek += 1
        col_tek = 0
    R = R_mod
    b = np.zeros((cols_mod - 1, 1))
    for i in range(cols_mod - 1):
        b[i, 0] = np.dot(L, R[0, i])
    Q = np.zeros((cols_mod - 1, cols_mod - 1))
    for i in range(cols_mod - 1):
        for j in range(cols_mod - 1):
            Q[i, j] = R[i + 1, j]
    A = np.zeros((cols_mod - 1, cols_mod - 1))
    for i in range(cols_mod - 1):
        for j in range(cols_mod - 1):
            if i == j:
                A[i, j] = 1.0 - Q[j, i]
            else:
                A[i, j] = -Q[j, i]

    intensities = np.dot(np.linalg.inv(A), b)
    l = [0.0] * (cols - 1)
    for i in range(cols_mod - 1):
        l[i] = intensities[i, 0]
    l_out = [0.0] * (cols - 1)
    int_col = 0
    for i in range(cols - 1):
        skip = False
        for n in null_numbers:
            if n == i:
                l_out[i] = 0
                skip = True
        if skip:
            continue

        l_out[i] = l[int_col]
        int_col += 1

    return l_out


def network_prty_calc(R, b, n, L, prty, nodes_prty):
    """
    Расчет СеМО
    :param R[k] - матрицы передачи, k - номер класса
    :param b[k][node][j] - нач моменты времени обслуживания
    node - номер узла сети
    k - номер класса
    j - номер нач. момента
    :param n[node] - количество каналов в узлах сети
    :param L[k] - вх. интенсивность для k-го класса
    :param prty[node] - вид приоритета в узле. 'PR', 'NP'
    :param nodes_prty [node][0,2,1] - перестановки приритетов для каждого узла
    :return: {'v':[], 'v_node':[], 'loads':[]}
    v[k][j] - нач. моменты времени пребывания в Сети
    v_node[node][k][j] - нач моменты вр пребывания в узле
    loads[node] - коэффициенты загрузки Сети
    """
    res = {}

    k_num = len(L)
    nodes = R[0].shape[0] - 1
    res['loads'] = [0.0] * nodes
    res['v'] = []
    intensities = []

    for k in range(k_num):
        intensities.append(balance_equation(L[k], R[k]))

    b_order = []
    l_order = []

    for m in range(nodes):
        b_order.append([])
        l_order.append([])
        for k in range(k_num):
            order = nodes_prty[m][k]
            b_order[m].append(b[order][m])
            l_order[m].append(intensities[order][m])

    res['v_node'] = []
    for i in range(nodes):
        l_sum = 0
        for k in range(k_num):
            l_sum += l_order[i][k]

        b_sr = [0.0] * 4

        for j in range(4):
            for k in range(k_num):
                b_sr[j] += b_order[i][k][j]
            b_sr[j] /= k_num

        res['loads'][i] = l_sum * b_sr[0] / n[i]
        res['v_node'].append(get_v_prty_invar(l_order[i], b_order[i], n[i], prty[i]))
        for k in range(k_num):
            res['v_node'][i][nodes_prty[i][k]] = res['v_node'][i][k]

    h = 0.0001
    s = [0.0] * 4
    for i in range(4):
        s[i] = h * (i + 1)

    for k in range(k_num):
        I = np.zeros((nodes, nodes))
        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    I[i, j] = 1
        N = np.zeros((nodes, nodes))
        P = np.zeros((1, nodes))
        for i in range(nodes):
            P[0, i] = R[k][0, i]
        T = np.zeros((nodes, 1))
        for i in range(nodes):
            T[i, 0] = R[k][i + 1, nodes]
        Q = np.zeros((nodes, nodes))

        for i in range(nodes):
            for j in range(nodes):
                Q[i, j] = R[k][i + 1, j]

        gamma_mu_alpha = []
        for i in range(nodes):
            gamma_mu_alpha.append(Gamma.get_mu_alpha([res['v_node'][i][k][0], res['v_node'][i][k][1]]))

        g_PLS = []
        for i in range(4):
            for j in range(nodes):
                N[j, j] = Gamma.get_pls(*gamma_mu_alpha[j], s[i])
            G = np.dot(N, Q)
            FF = I - G
            F = np.linalg.inv(FF)
            F = np.dot(P, np.dot(F, np.dot(N, T)))
            g_PLS.append(F[0, 0])

        res['v'].append([])
        res['v'][k] = diff5dots(g_PLS, h)
        res['v'][k][0] = -res['v'][k][0]
        res['v'][k][2] = -res['v'][k][2]

    return res
