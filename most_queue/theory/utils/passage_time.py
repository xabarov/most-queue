import copy
import math

import numpy as np


class passage_time_calc:
    def __init__(self, A, B, C, D, is_clx=True, is_verbose=False, l_tilda=None):
        self.A_input = A
        self.B_input = B
        self.C_input = C
        self.D_input = D
        if not l_tilda:
            self.l_tilda = len(D) - 1  # номер яруса, с которого все матрицы одинаковые
        else:
            self.l_tilda = l_tilda
        self.e = 1e-9
        self.F = []
        self.Fr = []
        self.B = []
        self.Br = []
        self.L = []
        self.Lr = []
        self.G = []
        self.is_clx = is_clx
        n_l_max = D[self.l_tilda].shape[0]
        if is_clx:
            self.G_l_tilda = np.zeros((n_l_max, n_l_max), dtype='complex64')
        else:
            self.G_l_tilda = np.zeros((n_l_max, n_l_max))
        self.Gr_tilda = []
        self.Gr = []
        self.Z = []
        self.is_verbose = is_verbose

    def calc(self):
        self.calc_FBL_matrices()
        self.G_tilda_calc()
        self.Gr_tilda_calc()
        self.G_calc()
        self.Gr_calc()
        self.Z_calc()

    def calc_FBL_matrices(self):
        """
        Расчитыыает матрицы F, B, L, Fr, Br, Lr
        """
        for i in range(self.l_tilda + 1):

            # получаем gamma параметры - интенсивности выхода из яруса
            gammas = []
            d_rows = self.D_input[i].shape[0]
            if d_rows == 1:
                gammas.append(self.D_input[i][0])
            else:
                for j in range(d_rows):
                    gammas.append(self.D_input[i][j, j])

            # формируем матрицу F

            self.F.append(copy.deepcopy(self.A_input[i]))
            rows = self.F[i].shape[0]
            cols = self.F[i].shape[1]
            for j in range(rows):
                for k in range(cols):
                    if math.isclose(gammas[j].real, 0):
                        self.F[i][j, k] = 0
                    else:
                        self.F[i][j, k] = self.F[i][j, k] / gammas[j]

            # формируем матрицу B

            self.B.append(copy.deepcopy(self.B_input[i]))
            rows = self.B[i].shape[0]
            cols = self.B[i].shape[1]
            for j in range(rows):
                for k in range(cols):
                    if math.isclose(gammas[j].real, 0):
                        self.B[i][j, k] = 0
                    else:
                        self.B[i][j, k] = self.B[i][j, k] / gammas[j]

            # формируем матрицу L

            self.L.append(copy.deepcopy(self.C_input[i]))
            rows = self.L[i].shape[0]
            cols = self.L[i].shape[1]
            for j in range(rows):
                for k in range(cols):
                    if math.isclose(gammas[j].real, 0):
                        self.L[i][j, k] = 0
                    else:
                        self.L[i][j, k] = self.L[i][j, k] / gammas[j]

            # формируем матрицы с индексом r. r - номер начального момента
            r_num = 3
            left_mrx = []
            self.Fr.append([])
            self.Br.append([])
            self.Lr.append([])

            for r in range(r_num):

                # формируем левую часть для матриц с индексом r.
                left_mrx.append(copy.deepcopy(self.D_input[i]))
                rows = self.D_input[i].shape[0]
                if rows == 1:
                    if math.isclose(left_mrx[r][0].real, 0):
                        left_mrx[r][0] = 0
                    else:
                        left_mrx[r][0] = math.factorial(r + 1) / pow(left_mrx[r][0], r + 1)

                else:
                    for j in range(rows):
                        if math.isclose(left_mrx[r][j, j].real, 0):
                            left_mrx[r][j, j] = 0
                        else:
                            left_mrx[r][j, j] = math.factorial(r + 1) / pow(left_mrx[r][j, j], r + 1)

                self.Fr[i].append(np.dot(left_mrx[r], self.F[i]))
                self.Br[i].append(np.dot(left_mrx[r], self.B[i]))
                self.Lr[i].append(np.dot(left_mrx[r], self.L[i]))

    def G_tilda_calc(self):

        """
        Вычисление матрицы G = G_l_tilda
        """
        b_rows = self.B[self.l_tilda].shape[0]
        b_cols = self.B[self.l_tilda].shape[1]
        if self.is_clx:
            self.G_l_tilda = np.eye(b_rows, b_cols,
                                    dtype='complex64')  # присвоить первоначальное значение G = B не работает !!!
        else:
            self.G_l_tilda = np.eye(b_rows, b_cols)
        max_elem = self.G_l_tilda.max()
        max_elem_pr = 0

        n_iter = 0

        while (abs(max_elem - max_elem_pr) > self.e):
            self.G_l_tilda = self.B[self.l_tilda] + np.dot(self.L[self.l_tilda], self.G_l_tilda) + \
                             np.dot(self.F[self.l_tilda], np.dot(self.G_l_tilda, self.G_l_tilda))
            max_elem_pr = max_elem
            max_elem = self.G_l_tilda.max()
            n_iter += 1

        if self.is_verbose:
            print("Количество итераций вычисления матрицы G_l_tilda = {0:d}".format(n_iter))

    def norm_mrx(self, mrx, is_max=True):
        rows = mrx.shape[0]
        cols = mrx.shape[1]
        if is_max:
            max_el = 0
            for r in range(rows):
                summ = 0
                for j in range(cols):
                    summ += math.fabs(mrx[r, j].real)
                if summ > max_el:
                    max_el = summ
            return max_el
        else:
            ave = 0
            for r in range(rows):
                for j in range(cols):
                    ave += math.fabs(mrx[r, j].real)
            return ave / (rows * cols)

    def Gr_tilda_calc(self):

        B1 = self.Br[self.l_tilda][0]
        L1 = self.Lr[self.l_tilda][0]
        L = self.L[self.l_tilda]
        F = self.F[self.l_tilda]
        G = self.G_l_tilda
        F1 = self.Fr[self.l_tilda][0]

        G1 = B1 + np.dot(L1, G) + np.dot(F1, np.dot(G, G))

        # if self.is_clx:
        #     G1 = np.array(np.eye(B1.shape[0], B1.shape[1]), dtype='complex64') # присвоить первоначальное значение G = B не работает !!!
        # else:
        #     G1 = np.array(np.eye(B1.shape[0], B1.shape[1]))

        # вычисляем первый момент
        max_elem = self.norm_mrx(G1)
        max_elem_pr = 0
        n_iter = 0

        while (abs(max_elem - max_elem_pr) > self.e):
            G1 = B1 + np.dot(L1, G) + np.dot(L, G1) + np.dot(F1, np.dot(G, G)) + np.dot(F, np.dot(G1, G)) \
                 + np.dot(F, np.dot(G, G1))
            max_elem_pr = max_elem
            max_elem = self.norm_mrx(G1)
            n_iter += 1

        if self.is_verbose:
            print("Количество итераций вычисления матрицы G_r1_tilda = {0:d}".format(n_iter))

        self.Gr_tilda.append(G1)

        # вычисляем второй момент
        B2 = self.Br[self.l_tilda][1]
        L2 = self.Lr[self.l_tilda][1]
        F2 = self.Fr[self.l_tilda][1]

        G2 = B2 + np.dot(L2, G) + 2 * np.dot(L1, G1) + np.dot(F2, np.dot(G, G)) \
             + 2 * np.dot(F1, (np.dot(G1, G) + np.dot(G, G1)))

        # if self.is_clx:
        #     G2 = np.array(np.eye(B2.shape[0], B2.shape[1]),
        #                    dtype='complex64')  # присвоить первоначальное значение G = B не работает !!!
        # else:
        #     G2 = np.array(np.eye(B2.shape[0], B2.shape[1]))
        #
        max_elem = self.norm_mrx(G2)
        max_elem_pr = 0
        n_iter = 0

        while (abs(max_elem - max_elem_pr) > self.e):
            G2 = B2 + np.dot(L2, G) + 2 * np.dot(L1, G1) + np.dot(L, G2) \
                 + np.dot(F2, np.dot(G, G)) + \
                 2 * np.dot(F1, (np.dot(G1, G) + np.dot(G, G1))) \
                 + np.dot(F, (np.dot(G2, G) + 2 * np.dot(G1, G1) + np.dot(G, G2)))

            max_elem_pr = max_elem
            max_elem = self.norm_mrx(G2)
            n_iter += 1

        if self.is_verbose:
            print("Количество итераций вычисления матрицы G_r2_tilda = {0:d}".format(n_iter))

        self.Gr_tilda.append(G2)

        # вычисляем 3 момент
        B3 = self.Br[self.l_tilda][2]
        L3 = self.Lr[self.l_tilda][2]
        F3 = self.Fr[self.l_tilda][2]

        G3 = B3 + np.dot(L3, G) + 3 * np.dot(L2, G1) + 3.0 * np.dot(L1, G2) + np.dot(F3, np.dot(G, G)) + \
             3.0 * np.dot(F2, (np.dot(G1, G) + np.dot(G, G1))) + \
             3.0 * np.dot(F1, (np.dot(G2, G1) + 2 * np.dot(G1, G1) + np.dot(G, G2)))

        max_elem = self.norm_mrx(G3)
        max_elem_pr = 0
        n_iter = 0

        while (abs(max_elem - max_elem_pr) > self.e):
            G3 = B3 + np.dot(L3, G) + 3 * np.dot(L2, G1) + 3 * np.dot(L1, G2) \
                 + np.dot(L, G3) + np.dot(F3, np.dot(G, G)) \
                 + 3 * np.dot(F2, (np.dot(G1, G) + np.dot(G, G1))) + \
                 3 * np.dot(F1, (np.dot(G2, G) + 2 * np.dot(G1, G1) + np.dot(G, G2))) \
                 + np.dot(F, (np.dot(G3, G) + 3 * np.dot(G2, G1) + 3 * np.dot(G1, G2) + np.dot(G, G3)))

            max_elem_pr = max_elem
            max_elem = self.norm_mrx(G3)
            n_iter += 1

        if self.is_verbose:
            print("Количество итераций вычисления матрицы G_r3_tilda = {0:d}".format(n_iter))

        self.Gr_tilda.append(G3)

    def G_calc(self):
        """
            Вычисление G для неповторяющихся матриц, l < l_tilda
        """
        for i in range(self.l_tilda):
            self.G.append([])
        self.G.append(self.G_l_tilda)

        for i in range(self.l_tilda):
            # расчет начинаем с конца
            B = copy.deepcopy(self.B[self.l_tilda - i - 1])
            L = self.L[self.l_tilda - i - 1]
            F = self.F[self.l_tilda - i - 1]
            G_plus_one = self.G[self.l_tilda - i]

            G = B

            # b_rows = B.shape[0]
            # b_cols = B.shape[1]
            # G = (np.array(np.eye(b_rows, b_cols)) - L - F*G_plus_one).I()*B

            max_elem = self.norm_mrx(G)
            max_elem_pr = 0
            n_iter = 0
            while (abs(max_elem - max_elem_pr) > self.e):
                G = B + np.dot(L, G) + np.dot(F, np.dot(G_plus_one, G))
                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G)
                n_iter += 1

            if self.is_verbose:
                print("Количество итераций вычисления матрицы G{0:d} = {1:d}".format(self.l_tilda - i, n_iter))

            self.G[self.l_tilda - i - 1] = G

    def Gr_calc(self):
        """
            Вычисление Gr для неповторяющихся матриц, l < l_tilda
        """
        for i in range(self.l_tilda):
            self.Gr.append([])
        self.Gr.append(self.Gr_tilda)

        for i in range(self.l_tilda):

            # вычисляем первый момент
            B1 = self.Br[self.l_tilda - i - 1][0]
            L1 = self.Lr[self.l_tilda - i - 1][0]
            L = self.L[self.l_tilda - i - 1]
            F1 = self.Fr[self.l_tilda - i - 1][0]
            F = self.F[self.l_tilda - i - 1]
            G1plus_one = self.Gr[self.l_tilda - i][0]
            Gplus_one = self.G[self.l_tilda - i]
            G = self.G[self.l_tilda - i - 1]

            G1 = B1 + np.dot(L1, G) + np.dot(F1, np.dot(Gplus_one, G)) + np.dot(F, np.dot(G1plus_one, G))
            # if i!=self.l_tilda-1:
            #     if self.is_clx:
            #         G1 = np.array(np.eye(B1.shape[0], B1.shape[1]),
            #                        dtype='complex64')  # присвоить первоначальное значение G = B не работает !!!
            #     else:
            #         G1 = np.array(np.eye(B1.shape[0], B1.shape[1]))

            max_elem = self.norm_mrx(G1)
            max_elem_pr = 0
            n_iter = 0
            while (abs(max_elem - max_elem_pr) > self.e):
                G1 = B1 + np.dot(L1, G) + np.dot(L, G1) + np.dot(F1, np.dot(Gplus_one, G)) \
                     + np.dot(F, np.dot(G1plus_one, G)) + np.dot(F, np.dot(Gplus_one, G1))
                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G1)
                n_iter += 1
            self.Gr[self.l_tilda - i - 1].append(G1)

            if self.is_verbose:
                print("Количество итераций вычисления матрицы Gr{0:d}[0] = {1:d}".format(self.l_tilda - i, n_iter))

            # вычисляем 2 момент
            B2 = self.Br[self.l_tilda - i - 1][1]
            L2 = self.Lr[self.l_tilda - i - 1][1]
            F2 = self.Fr[self.l_tilda - i - 1][1]
            G2plus_one = self.Gr[self.l_tilda - i][1]

            G2 = B2 + np.dot(L2, G) + 2 * np.dot(L1, G1) + np.dot(F2, np.dot(Gplus_one, G)) + \
                 2 * np.dot(F1, (np.dot(G1plus_one, G) + np.dot(Gplus_one, G1)))

            max_elem = self.norm_mrx(G2)
            max_elem_pr = 0
            n_iter = 0
            while (abs(max_elem - max_elem_pr) > self.e):
                G2 = B2 + np.dot(L2, G) + 2.0 * np.dot(L1, G1) + np.dot(L, G2) \
                     + np.dot(F2, np.dot(Gplus_one, G)) + 2.0 * np.dot(F1,
                                                                       (np.dot(G1plus_one, G) + np.dot(Gplus_one, G1))) \
                     + np.dot(F, (np.dot(G2plus_one, G) + 2.0 * np.dot(G1plus_one, G1) + np.dot(Gplus_one, G2)))
                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G2)
                n_iter += 1
            self.Gr[self.l_tilda - i - 1].append(G2)
            if self.is_verbose:
                print("Количество итераций вычисления матрицы Gr{0:d}[1] = {1:d}".format(self.l_tilda - i, n_iter))

            # вычисляем 3 момент
            B3 = self.Br[self.l_tilda - i - 1][2]
            L3 = self.Lr[self.l_tilda - i - 1][2]
            F3 = self.Fr[self.l_tilda - i - 1][2]
            G3plus_one = self.Gr[self.l_tilda - i][2]

            G3 = B3 + np.dot(L3, G) + 3.0 * np.dot(L2, G1) + 3.0 * np.dot(L1, G2) \
                 + np.dot(F3, np.dot(Gplus_one, G)) \
                 + 3.0 * np.dot(F2, (np.dot(G1plus_one, G) + np.dot(Gplus_one, G1))) \
                 + 3.0 * np.dot(F1, (np.dot(G2plus_one, G) + 2.0 * np.dot(G1plus_one, G1) + np.dot(Gplus_one, G2)))

            max_elem = self.norm_mrx(G3)
            max_elem_pr = 0
            n_iter = 0
            while (abs(max_elem - max_elem_pr) > self.e):
                G3 = B3 + np.dot(L3, G) + 3.0 * np.dot(L2, G1) + 3.0 * np.dot(L1, G2) + np.dot(L, G3) \
                     + np.dot(F3, np.dot(Gplus_one, G)) \
                     + 3.0 * np.dot(F2, (np.dot(G1plus_one, G) + np.dot(Gplus_one, G1))) \
                     + 3.0 * np.dot(F1, (np.dot(G2plus_one, G) + 2.0 * np.dot(G1plus_one, G1) + np.dot(Gplus_one, G2))) \
                     + np.dot(F, (
                            np.dot(G3plus_one, G) + 3 * np.dot(G2plus_one, G1) + 3.0 * np.dot(G1plus_one, G2) + np.dot(
                        Gplus_one, G3)))

                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G3)
                n_iter += 1
            self.Gr[self.l_tilda - i - 1].append(G3)
            if self.is_verbose:
                print("Количество итераций вычисления матрицы Gr{0:d}[2] = {1:d}".format(self.l_tilda - i, n_iter))

    def Z_calc(self):
        for i in range(self.l_tilda + 1):
            rows = self.Gr[i][0].shape[0]
            cols = self.Gr[i][0].shape[1]
            self.Z.append([])
            for r in range(3):
                if self.is_clx:
                    self.Z[i].append(np.zeros((rows, cols), dtype='complex64'))
                else:
                    self.Z[i].append(np.zeros((rows, cols)))
                for j in range(rows):
                    for t in range(cols):
                        znam = self.G[i][j, t]
                        if math.isclose(znam.real, 0.0):
                            self.Z[i][r][j, t] = 0
                        else:
                            self.Z[i][r][j, t] = self.Gr[i][r][j, t] / znam

    def Z_gap_calc(self, l_start, l_end):
        G_gap = self.G_gap_calc(l_start, l_end)
        Gr_gap = self.Gr_gap_calc(l_start, l_end)
        rows = Gr_gap[0].shape[0]
        cols = Gr_gap[0].shape[1]
        Z = []
        for r in range(3):
            if self.is_clx:
                Z.append(np.zeros((rows, cols), dtype='complex64'))
            else:
                Z.append(np.zeros((rows, cols)))
            for j in range(rows):
                for t in range(cols):
                    znam = G_gap[j, t]
                    if math.isclose(znam.real, 0.0):
                        Z[r][j, t] = 0
                    else:
                        Z[r][j, t] = Gr_gap[r][j, t] / znam
        return Z

    def G_gap_calc(self, l_start, l_end):

        if l_start <= l_end:
            return -1

        # B section
        if l_start >= self.l_tilda:
            BG = copy.deepcopy(self.B[self.l_tilda])
            LG = np.dot(self.L[self.l_tilda], self.G[self.l_tilda])
            FGG = np.dot(self.F[self.l_tilda], np.dot(self.G[self.l_tilda], self.G[self.l_tilda]))
        elif l_start == self.l_tilda - 1:
            BG = copy.deepcopy(self.B[l_start])
            LG = np.dot(self.L[l_start], self.G[l_start])
            FGG = np.dot(self.F[l_start], np.dot(self.G[self.l_tilda], self.G[l_start]))
        else:
            BG = copy.deepcopy(self.B[l_start])
            LG = np.dot(self.L[l_start], self.G[l_start])
            FGG = np.dot(self.F[l_start], np.dot(self.G[l_start + 1], self.G[l_start]))

        for i in range(l_start - 1, l_end, -1):
            if i >= self.l_tilda:
                BG = np.dot(BG, self.G[self.l_tilda])
                LG = np.dot(LG, self.G[self.l_tilda])
                FGG = np.dot(FGG, self.G[self.l_tilda])
            else:
                BG = np.dot(BG, self.G[i])
                LG = np.dot(LG, self.G[i])
                FGG = np.dot(FGG, self.G[i])

        G_gap = BG + LG + FGG

        return G_gap

    def Gr_gap_calc(self, l_start, l_end):

        if l_start <= l_end:
            return -1

        Gr_gap = []
        for i in range(3):
            if l_end + 1 >= self.l_tilda:
                Gr_gap.append(copy.deepcopy(self.Gr[self.l_tilda][i]))
            else:
                Gr_gap.append(copy.deepcopy(self.Gr[l_end + 1][i]))

        if l_start <= l_end:
            return -1

        for i in range(l_end + 2, l_start + 1):

            # первый момент
            Ggap_minus_one = self.G_gap_calc(i - 1, l_end)

            # B1G-section

            if i >= self.l_tilda:
                B1G = np.dot(self.Br[self.l_tilda][0], Ggap_minus_one)
            else:
                B1G = np.dot(self.Br[i][0], Ggap_minus_one)

            # BG1 section
            if i >= self.l_tilda:
                BG1 = np.dot(self.B[self.l_tilda], Gr_gap[0])
            else:
                BG1 = np.dot(self.B[i], Gr_gap[0])

            # L1GG-section
            if i >= self.l_tilda:
                L1GG = np.dot(self.Lr[self.l_tilda][0], np.dot(self.G[self.l_tilda], Ggap_minus_one))
            else:
                L1GG = np.dot(self.Lr[i][0], np.dot(self.G[i], Ggap_minus_one))

            # LG1G section
            if i >= self.l_tilda:
                LG1G = np.dot(self.L[self.l_tilda], np.dot(self.Gr[self.l_tilda][0], Ggap_minus_one))
            else:
                LG1G = np.dot(self.L[i], np.dot(self.Gr[i][0], Ggap_minus_one))

            # LGG1 section
            if i >= self.l_tilda:
                LGG1 = np.dot(self.L[self.l_tilda], np.dot(self.G[self.l_tilda], Gr_gap[0]))
            else:
                LGG1 = np.dot(self.L[i], np.dot(self.G[i], Gr_gap[0]))

            # F1GGG-section
            if i >= self.l_tilda:
                F1GGG = np.dot(self.Fr[self.l_tilda][0],
                               np.dot(self.G[self.l_tilda], np.dot(self.G[self.l_tilda], Ggap_minus_one)))
            elif i == self.l_tilda - 1:
                F1GGG = np.dot(self.Fr[i][0], np.dot(self.G[self.l_tilda], np.dot(self.G[i], Ggap_minus_one)))
            else:
                F1GGG = np.dot(self.Fr[i][0], np.dot(self.G[i + 1], np.dot(self.G[i], Ggap_minus_one)))

            # FG1GG-section
            if i >= self.l_tilda:
                FG1GG = np.dot(self.F[self.l_tilda],
                               np.dot(self.Gr[self.l_tilda][0], np.dot(self.G[self.l_tilda], Ggap_minus_one)))
            elif i == self.l_tilda - 1:
                FG1GG = np.dot(self.F[i], np.dot(self.Gr[self.l_tilda][0], np.dot(self.G[i], Ggap_minus_one)))
            else:
                FG1GG = np.dot(self.F[i], np.dot(self.Gr[i + 1][0], np.dot(self.G[i], Ggap_minus_one)))

            # FGG1G-section
            if i >= self.l_tilda:
                FGG1G = np.dot(self.F[self.l_tilda],
                               np.dot(self.G[self.l_tilda], np.dot(self.Gr[self.l_tilda][0], Ggap_minus_one)))
            elif i == self.l_tilda - 1:
                FGG1G = np.dot(self.F[i], np.dot(self.G[self.l_tilda], np.dot(self.Gr[i][0], Ggap_minus_one)))
            else:
                FGG1G = np.dot(self.F[i], np.dot(self.G[i + 1], np.dot(self.Gr[i][0], Ggap_minus_one)))

            # FGGG1-section
            if i >= self.l_tilda:
                FGGG1 = np.dot(self.F[self.l_tilda],
                               np.dot(self.G[self.l_tilda], np.dot(self.G[self.l_tilda], Gr_gap[0])))
            elif i == self.l_tilda - 1:
                FGGG1 = np.dot(self.F[i], np.dot(self.G[self.l_tilda], np.dot(self.G[i], Gr_gap[0])))
            else:
                FGGG1 = np.dot(self.F[i], np.dot(self.G[i + 1], np.dot(self.G[i], Gr_gap[0])))

            Gr_gap[0] = B1G + BG1 + L1GG + LG1G + LGG1 + F1GGG + FG1GG + FGG1G + FGGG1
            Gr_gap[1] = Gr_gap[0]
            Gr_gap[2] = Gr_gap[0]
            # while (abs(max_elem - max_elem_pr) > self.e):
            #     Gr_gap[0] = B1G + BG1 + L1G + L*Gr_gap[0] + F1GG + FG1G + FG1*Gr_gap[0]
            #     max_elem_pr = max_elem
            #     max_elem = self.norm_mrx(Gr_gap[0])

        return Gr_gap


if __name__ == '__main__':

    from most_queue.rand_distribution import H2_dist, Cox_dist
    from most_queue.theory.priority_calc import ppnz_calc

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

    h2_params_clx = H2_dist.get_params_clx(b)

    y1 = h2_params_clx[0]
    y2 = 1.0 - y1
    mu1 = h2_params_clx[1]
    mu2 = h2_params_clx[2]

    A = []
    A.append(np.array([[l * y1, l * y2]], dtype='complex64'))
    A.append(np.array([[l, 0], [0, l]], dtype='complex64'))
    A.append(np.array([[l, 0], [0, l]], dtype='complex64'))

    B = []
    B.append(np.array([[0]], dtype='complex64'))
    B.append(np.array([[mu1], [mu2]], dtype='complex64'))
    B.append(np.array([[mu1 * y1, mu1 * y2], [mu2 * y1, mu2 * y2]], dtype='complex64'))

    C = []
    C.append(np.array([[0]], dtype='complex64'))
    C.append(np.array([[0, 0], [0, 0]], dtype='complex64'))
    C.append(np.array([[0, 0], [0, 0]], dtype='complex64'))

    D = []
    D.append(np.array([[l]], dtype='complex64'))
    D.append(np.array([[l + mu1, 0], [0, l + mu2]], dtype='complex64'))
    D.append(np.array([[l + mu1, 0], [0, l + mu2]], dtype='complex64'))

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

    pnz = ppnz_calc(l, b, 3)

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
    cox_param = Cox_dist.get_params(b)

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

    pnz = ppnz_calc(l, b, 3)

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
    A.append(np.array([[l]], dtype='complex64'))
    A.append(np.array([[l]], dtype='complex64'))
    A.append(np.array([[l]], dtype='complex64'))
    A.append(np.array([[l]], dtype='complex64'))

    B = []
    B.append(np.array([[0]], dtype='complex64'))
    B.append(np.array([[mu]], dtype='complex64'))
    B.append(np.array([[2 * mu]], dtype='complex64'))
    B.append(np.array([[3 * mu]], dtype='complex64'))

    C = []
    C.append(np.array([[0]], dtype='complex64'))
    C.append(np.array([[0]], dtype='complex64'))
    C.append(np.array([[0]], dtype='complex64'))
    C.append(np.array([[0]], dtype='complex64'))

    D = []
    D.append(np.array([[l]], dtype='complex64'))
    D.append(np.array([[l + mu]], dtype='complex64'))
    D.append(np.array([[l + 2 * mu]], dtype='complex64'))
    D.append(np.array([[l + 3 * mu]], dtype='complex64'))

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

    pnz = ppnz_calc(l, b_mom, 3)

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
    # pnz = ppnz_calc(l_H, b_mom, 3)
    #
    # param_cox = Cox_dist.get_params(pnz)
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
