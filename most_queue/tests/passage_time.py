from most_queue.theory.passage_time import *
from most_queue.theory import prty_calc
from most_queue.sim import rand_destribution as rd


def test():
    print("\nСМО типа M/H2/1. Для одноканальной СМО passage time является ПНЗ. \n"
          "Сравним значения начальных моментов, полученных методом Ньютса \n"
          "и стандартным методом из библиотеки prty_calc\n")

    l = 1.0
    b_coev = 0.8
    b1 = 0.8
    b = [0.0] * 3
    alpha = 1 / (b_coev ** 2)
    b[0] = b1
    b[1] = pow(b[0], 2) * (pow(b_coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    h2_params_clx = rd.H2_dist.get_params_clx(b)

    y1 = h2_params_clx[0]
    y2 = 1.0 - y1
    mu1 = h2_params_clx[1]
    mu2 = h2_params_clx[2]

    A = []
    A.append(np.array([[l * y1, l * y2]], dtype="complex_"))
    A.append(np.array([[l, 0], [0, l]], dtype="complex_"))
    A.append(np.array([[l, 0], [0, l]], dtype="complex_"))

    B = []
    B.append(np.array([[0]], dtype="complex_"))
    B.append(np.array([[mu1], [mu2]], dtype="complex_"))
    B.append(np.array([[mu1 * y1, mu1 * y2], [mu2 * y1, mu2 * y2]], dtype="complex_"))

    C = []
    C.append(np.array([[0]], dtype="complex_"))
    C.append(np.array([[0, 0], [0, 0]], dtype="complex_"))
    C.append(np.array([[0, 0], [0, 0]], dtype="complex_"))

    D = []
    D.append(np.array([[l]], dtype="complex_"))
    D.append(np.array([[l + mu1, 0], [0, l + mu2]], dtype="complex_"))
    D.append(np.array([[l + mu1, 0], [0, l + mu2]], dtype="complex_"))

    pass_time = passage_time_calc(A, B, C, D)

    pass_time.calc()

    # print("\nЗначения матриц Z:\n")
    #
    # z_num = len(pass_time.Z)
    # for i in range(z_num):
    #
    #     print("Z{0:^1d}".format(i+1))
    #     for r in range(3):
    #         print("r = {0:^1d}".format(r + 1))
    #         rows = pass_time.Z[i][r].shape[0]
    #         cols = pass_time.Z[i][r].shape[1]
    #         for j in range(rows):
    #             for t in range(cols):
    #                 if t==cols-1:
    #                     if math.isclose(pass_time.Z[i][r][j, t].imag, 0):
    #                         print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t].real))
    #                     else:
    #                         print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]))
    #                 else:
    #                     if math.isclose(pass_time.Z[i][r][j, t].imag, 0):
    #                         print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t].real), end='')
    #                     else:
    #                         print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]), end='')
    #
    #
    # print("\nЗначения матриц G:\n")
    #
    # g_num = len(pass_time.G)
    # for i in range(g_num):
    #
    #     print("G{0:^1d}".format(i + 1))
    #
    #     rows = pass_time.G[i].shape[0]
    #     cols = pass_time.G[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 if math.isclose(pass_time.G[i][j, t].imag, 0):
    #                     print("{0:^5.3g}  ".format(pass_time.G[i][j, t].real))
    #                 else:
    #                     print("{0:^5.3g}  ".format(pass_time.G[i][j, t]))
    #             else:
    #                 if math.isclose(pass_time.G[i][j, t].imag, 0):
    #                     print("{0:^5.3g}  ".format(pass_time.G[i][j, t].real), end=" ")
    #                 else:
    #                     print("{0:^5.3g}  ".format(pass_time.G[i][j, t]), end=" ")

    neuts_moments = []
    for r in range(3):
        neuts_moments.append(
            pass_time.Z[1][r][0, 0] * pass_time.G[2][0, 0] + pass_time.Z[1][r][1, 0] * pass_time.G[2][0, 1])

    pnz = prty_calc.ppnz_calc(l, b, 3)

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Ньютс", "ПНЗ"))
    print("-" * 45)
    for j in range(3):
        if math.isclose(neuts_moments[j].imag, 0):
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, neuts_moments[j].real, pnz[j]))
        else:
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, neuts_moments[j], pnz[j]))

    print("\nСМО типа M/C2/1. Для одноканальной СМО passage time является ПНЗ. \nСравним значения "
          "начальных моментов, полученных методом Ньютса \nи стандартным методом из библиотеки prty_calc\n")

    l = 1.0
    b = [0.8, 2, 15]
    cox_param = rd.Cox_dist.get_params(b)

    y1_cox = cox_param[0]
    mu1_cox = cox_param[1]
    mu2_cox = cox_param[2]

    t1 = mu1_cox * (1.0 - y1_cox)
    t12 = mu1_cox * y1_cox
    t2 = mu2_cox

    A = []
    A.append(np.array([[l, 0]]))
    A.append(np.array([[l, 0], [0, l]]))
    A.append(np.array([[l, 0], [0, l]]))

    B = []
    B.append(np.array([[0]]))
    B.append(np.array([[t1], [t2]]))
    B.append(np.array([[t1, 0], [t2, 0]]))

    C = []
    C.append(np.array([[0]]))
    C.append(np.array([[0, t12], [0, 0]]))
    C.append(np.array([[0, t12], [0, 0]]))

    D = []
    D.append(np.array([[l]]))
    D.append(np.array([[l + t1 + t12, 0], [0, l + t2]]))
    D.append(np.array([[l + t1 + t12, 0], [0, l + t2]]))

    pass_time = passage_time_calc(A, B, C, D)

    pass_time.calc()

    # print("\nЗначения матриц Z:\n")
    #
    # z_num = len(pass_time.Z)
    # for i in range(z_num):
    #
    #     print("Z{0:^1d}".format(i + 1))
    #     for r in range(3):
    #         print("r = {0:^1d}".format(r + 1))
    #         rows = pass_time.Z[i][r].shape[0]
    #         cols = pass_time.Z[i][r].shape[1]
    #         for j in range(rows):
    #             for t in range(cols):
    #                 if t == cols - 1:
    #                     print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]))
    #                 else:
    #                     print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]), end=" ")
    #
    # print("\nЗначения матриц G:\n")
    #
    # g_num = len(pass_time.G)
    # for i in range(g_num):
    #
    #     print("G{0:^1d}".format(i + 1))
    #
    #     rows = pass_time.G[i].shape[0]
    #     cols = pass_time.G[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 print("{0:^5.3g}  ".format(pass_time.G[i][j, t]))
    #             else:
    #                 print("{0:^5.3g}  ".format(pass_time.G[i][j, t]), end=" ")

    neuts_moments = []
    for r in range(3):
        neuts_moments.append(pass_time.Z[1][r][0, 0])

    pnz = prty_calc.ppnz_calc(l, b, 3)

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Ньютс", "ПНЗ"))
    print("-" * 45)
    for j in range(3):
        if math.isclose(neuts_moments[j].imag, 0):
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, neuts_moments[j].real, pnz[j]))
        else:
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, neuts_moments[j], pnz[j]))

    print("\nСМО типа M/M/3. Для СМО passage time яруса 3 является ПНЗ.\nСравним значения "
          "начальных моментов, полученных методом Ньютса \nи стандартным методом из библиотеки"
          " prty_calc с интенсивностью 3mu\n")

    l = 1.0
    n = 3
    ro = 0.9
    mu = l / (ro * n)

    A = []
    A.append(np.array([[l]], dtype="complex_"))
    A.append(np.array([[l]], dtype="complex_"))
    A.append(np.array([[l]], dtype="complex_"))
    A.append(np.array([[l]], dtype="complex_"))

    B = []
    B.append(np.array([[0]], dtype="complex_"))
    B.append(np.array([[mu]], dtype="complex_"))
    B.append(np.array([[2 * mu]], dtype="complex_"))
    B.append(np.array([[3 * mu]], dtype="complex_"))

    C = []
    C.append(np.array([[0]], dtype="complex_"))
    C.append(np.array([[0]], dtype="complex_"))
    C.append(np.array([[0]], dtype="complex_"))
    C.append(np.array([[0]], dtype="complex_"))

    D = []
    D.append(np.array([[l]], dtype="complex_"))
    D.append(np.array([[l + mu]], dtype="complex_"))
    D.append(np.array([[l + 2 * mu]], dtype="complex_"))
    D.append(np.array([[l + 3 * mu]], dtype="complex_"))

    pass_time = passage_time_calc(A, B, C, D)

    pass_time.calc()

    # print("\nЗначения матриц Z:\n")
    #
    # z_num = len(pass_time.Z)
    # for i in range(z_num):
    #
    #     print("Z{0:^1d}".format(i + 1))
    #     for r in range(3):
    #         print("r = {0:^1d}".format(r + 1))
    #         rows = pass_time.Z[i][r].shape[0]
    #         cols = pass_time.Z[i][r].shape[1]
    #         for j in range(rows):
    #             for t in range(cols):
    #                 if t == cols - 1:
    #                     print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]))
    #                 else:
    #                     print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]), end=" ")
    #
    # print("\nЗначения матриц G:\n")
    #
    # g_num = len(pass_time.G)
    # for i in range(g_num):
    #
    #     print("G{0:^1d}".format(i + 1))
    #
    #     rows = pass_time.G[i].shape[0]
    #     cols = pass_time.G[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 print("{0:^5.3g}  ".format(pass_time.G[i][j, t]))
    #             else:
    #                 print("{0:^5.3g}  ".format(pass_time.G[i][j, t]), end=" ")

    neuts_moments = []
    for r in range(3):
        neuts_moments.append(pass_time.Z[3][r][0, 0])

    b_mom = [0, 0, 0]

    for j in range(3):
        b_mom[j] = math.factorial(j + 1) / pow(3 * mu, j + 1)

    pnz = prty_calc.ppnz_calc(l, b_mom, 3)

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Ньютс", "ПНЗ"))
    print("-" * 45)
    for j in range(3):
        if math.isclose(neuts_moments[j].imag, 0):
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, neuts_moments[j].real, pnz[j]))
        else:
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, neuts_moments[j], pnz[j]))

    # mu_M = 1.5  # интенсивность обслуживания заявок 2-го класса
    # mu_H = 1.5  # интенсивность обслуживания заявок 1-го класса
    # l_M = 1.2  # интенсивность вх потока заявок 2-го класса
    # l_H = 1.2  # интенсивность вх потока заявок 1-го класса
    # n = 2
    #
    # mu_sr = (mu_H + mu_M)/2
    # ro = (l_M+l_H)/(mu_sr*n)
    #
    # print("ro = {0:^4.2g}".format(ro))
    #
    # b_mom = [0, 0, 0]
    #
    # for j in range(3):
    #     b_mom[j] = math.factorial(j + 1) / pow(2 * mu_H, j + 1)
    #
    #
    # pnz = prty_calc.ppnz_calc(l_H, b_mom, 3)
    #
    # param_cox = rd.Cox_dist.get_params(pnz)
    #
    # print("coev = {0:^8.3g}".format(math.sqrt(pnz[1]-pnz[0]**2)/pnz[0]))
    #
    # y1_cox = param_cox[0]
    # mu1_cox = param_cox[1]
    # mu2_cox = param_cox[2]
    #
    # t1 = mu1_cox*(1.0-y1_cox)
    # t12 = mu1_cox*y1_cox
    # t2 = mu2_cox
    #
    # A = []
    # A.append(np.array([[l_H, l_M]]))
    # A.append(np.array([[l_H, 0, l_M, 0], [0, 0, l_H, l_M]]))
    # A.append(np.array([[l_M, 0, 0, 0], [0, l_M, 0, 0], [l_H, 0, l_M, 0], [0, 0, l_H, l_M]]))
    # A.append(np.array([[l_M, 0, 0, 0], [0, l_M, 0, 0], [l_H, 0, l_M, 0], [0, 0, l_H, l_M]]))
    #
    # B = []
    # B.append(np.array([[0]]))
    # B.append(np.array([[mu_H], [mu_M]]))
    # B.append(np.array([[t1, 0], [t2, 0], [mu_M, mu_H], [0, 2*mu_M]]))
    # B.append(np.array([[0, 0, t1, 0], [0, 0, t2, 0], [0, 0, mu_M, mu_H], [0, 0, 0, 2 * mu_M]]))
    #
    # C = []
    # C.append(np.array([[0]]))
    # C.append(np.array([[0, 0], [0, 0]]))
    # C.append(np.array([[0, t12, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    # C.append(np.array([[0, t12, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    #
    #
    # D = []
    # for i in range(len(C)):
    #     d_rows = C[i].shape[0]
    #     D.append(np.array(np.zeros((d_rows, d_rows))))
    #
    #     for row in range(d_rows):
    #         a_sum = 0
    #         a_cols = A[i].shape[1]
    #         for j in range(a_cols):
    #             a_sum += A[i][row, j]
    #         b_sum = 0
    #         b_cols = B[i].shape[1]
    #         for j in range(b_cols):
    #             b_sum += B[i][row, j]
    #         c_sum = 0
    #         c_cols = C[i].shape[1]
    #         for j in range(c_cols):
    #             c_sum += C[i][row, j]
    #         D[i][row, row] = a_sum + b_sum + c_sum
    #
    #
    # pass_time = passage_time_calc(A, B, C, D)
    #
    # pass_time.calc()
    # busy_periods = []
    # coev = []
    # for j in range(6):
    #     busy_periods.append([0, 0, 0])
    #     coev.append(0)
    # for r in range(3):
    #     busy_periods[0][r] = pass_time.Z[2][r][3, 1]
    #     busy_periods[1][r] = pass_time.Z[2][r][3, 0]
    #     busy_periods[2][r] = pass_time.Z[2][r][2, 1]
    #     busy_periods[3][r] = pass_time.Z[2][r][2, 0]
    #     busy_periods[4][r] = pass_time.Z[2][r][0, 1]
    #     busy_periods[5][r] = pass_time.Z[2][r][0, 0]
    #
    # print("\nПериоды НЗ:\n")
    # for j in range(6):
    #     coev[j] = busy_periods[j][1] - busy_periods[j][0]**2
    #     coev[j] = math.sqrt(coev[j])/busy_periods[j][0]
    #     for r in range(3):
    #         print("{0:^8.3g}".format(busy_periods[j][r]), end=" ")
    #     print("coev = {0:^4.3g}".format(coev[j]))
    #
    # pp = []
    # pp.append(pass_time.G[2][3, 1])
    # pp.append(pass_time.G[2][3, 0])
    # pp.append(pass_time.G[2][2, 1])
    # pp.append(pass_time.G[2][2, 0])
    # pp.append(pass_time.G[2][0, 1])
    # pp.append(pass_time.G[2][0, 0])
    #
    # # pp - список из шести вероятностей p2mm, p2mh, phmm, phmh, p2hm, p2hh
    # print("\nВероятности:\n")
    # print("p2mm = {0:^4.3g}".format(pp[0]))
    # print("p2mh = {0:^4.3g}".format(pp[1]))
    # print("phmm = {0:^4.3g}".format(pp[2]))
    # print("phmh = {0:^4.3g}".format(pp[3]))
    # print("p2hm = {0:^4.3g}".format(pp[4]))
    # print("p2hh = {0:^4.3g}".format(pp[5]))

    # print("\nЗначения матриц Z:\n")
    #
    # z_num = len(pass_time.Z)
    # for i in range(z_num):
    #
    #     print("Z{0:^1d}".format(i+1))
    #     for r in range(3):
    #         print("r = {0:^1d}".format(r + 1))
    #         rows = pass_time.Z[i][r].shape[0]
    #         cols = pass_time.Z[i][r].shape[1]
    #         for j in range(rows):
    #             for t in range(cols):
    #                 if t==cols-1:
    #                     print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]))
    #                 else:
    #                     print("{0:^5.3g}  ".format(pass_time.Z[i][r][j, t]), end=" ")
    #
    #
    #
    # print("\nЗначения матриц:\n")
    #
    # f_num = len(pass_time.F)
    # for i in range(f_num):
    #
    #     print("F{0:^1d}".format(i))
    #
    #     rows = pass_time.F[i].shape[0]
    #     cols = pass_time.F[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 print("{0:^5.3g}  ".format(pass_time.F[i][j, t]))
    #             else:
    #                 print("{0:^5.3g}  ".format(pass_time.F[i][j, t]), end=" ")
    #
    #     print("B{0:^1d}".format(i))
    #
    #     rows = pass_time.B[i].shape[0]
    #     cols = pass_time.B[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 print("{0:^5.3g}  ".format(pass_time.B[i][j, t]))
    #             else:
    #                 print("{0:^5.3g}  ".format(pass_time.B[i][j, t]), end=" ")
    #
    #     print("L{0:^1d}".format(i))
    #
    #     rows = pass_time.L[i].shape[0]
    #     cols = pass_time.L[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 print("{0:^5.3g}  ".format(pass_time.L[i][j, t]))
    #             else:
    #                 print("{0:^5.3g}  ".format(pass_time.L[i][j, t]), end=" ")
    #
    # print("\nЗначения матриц G:\n")
    #
    # g_num = len(pass_time.G)
    # for i in range(g_num):
    #
    #     print("G{0:^1d}".format(i))
    #
    #     rows = pass_time.G[i].shape[0]
    #     cols = pass_time.G[i].shape[1]
    #     for j in range(rows):
    #         for t in range(cols):
    #             if t == cols - 1:
    #                 print("{0:^5.3g}  ".format(pass_time.G[i][j, t]))
    #             else:
    #                 print("{0:^5.3g}  ".format(pass_time.G[i][j, t]), end=" ")


if __name__ == "__main__":
    test()
