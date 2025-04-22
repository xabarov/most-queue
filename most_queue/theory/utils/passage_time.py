import copy
import math

import numpy as np
import numpy.typing as npt


class PassageTimeCalculation:
    """
    Class for calculating the passage time of a Markov chain.
    """

    def __init__(self, A: list[npt.ArrayLike], B: list[npt.ArrayLike],
                 C: list[npt.ArrayLike], D: list[npt.ArrayLike],
                 is_clx=True, is_verbose=False, l_tilda=None):
        """
        Initialize the PassageTimeCalculation class.
        :param A: list of matrices A for each level
        :param B: list of matrices B for each level
        :param C: list of matrices C for each level
        :param D: list of matrices D for each level
        :param is_clx: if True, the chain is complex (has different levels)
        :param is_verbose: if True, print intermediate results
        :param l_tilda: index of the last level with different matrices, default is len(D) - 1
        """
        self.A_input = A
        self.B_input = B
        self.C_input = C
        self.D_input = D
        if not l_tilda:
            # number of levels with different matrices
            self.l_tilda = len(D) - 1
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
        """
        Calculation of F, B, L, Fr, Br, Lr matrices.
        """
        self.calc_FBL_matrices()
        self.G_tilda_calc()
        self.Gr_tilda_calc()
        self.G_calc()
        self.Gr_calc()
        self.Z_calc()
        
    def get_gammas(self, i):
        d_rows = self.D_input[i].shape[0]
        gammas = []
        if d_rows == 1:
            gammas.append(self.D_input[i][0])
        else:
            for j in range(d_rows):
                gammas.append(self.D_input[i][j, j])
        return gammas
    
    def process_matrix(self, matrix, gammas):
        rows, cols = matrix.shape
        for j in range(rows):
            for k in range(cols):
                if math.isclose(gammas[j].real, 0):
                    matrix[j, k] = 0
                else:
                    matrix[j, k] /= gammas[j]
        return matrix
    
    def process_matrix_with_factorial(self, matrix, r):
        rows, cols = matrix.shape
        for j in range(rows):
            if math.isclose(matrix[j, j].real, 0):
                matrix[j, j] = 0
            else:
                matrix[j, j] = math.factorial(r + 1) / pow(matrix[j, j], r + 1)
        return matrix

    def calc_FBL_matrices(self):
        """
        Calculation of F, B, L, Fr, Br, Lr
        """
        for i in range(self.l_tilda + 1):

            # получаем gamma параметры - интенсивности выхода из яруса
            gammas = self.get_gammas(i)

            # формируем матрицу F

            self.F.append(copy.deepcopy(self.A_input[i]))
            self.B.append(copy.deepcopy(self.B_input[i]))
            self.L.append(copy.deepcopy(self.C_input[i]))
            
            self.F[i] = self.process_matrix(self.F[i], gammas)
            self.B[i] = self.process_matrix(self.B[i], gammas)
            self.L[i] = self.process_matrix(self.L[i], gammas)

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
                    left_mrx[r][0] = self.process_matrix_with_factorial(left_mrx[r], r)[0, 0]
                else:
                    left_mrx[r] = self.process_matrix_with_factorial(left_mrx[r], r)

                self.Fr[i].append(np.dot(left_mrx[r], self.F[i]))
                self.Br[i].append(np.dot(left_mrx[r], self.B[i]))
                self.Lr[i].append(np.dot(left_mrx[r], self.L[i]))

    def G_tilda_calc(self):
        """
        calculation of the matrix G = G_l_tilda
        """
        b_rows = self.B[self.l_tilda].shape[0]
        b_cols = self.B[self.l_tilda].shape[1]
        if self.is_clx:
            self.G_l_tilda = np.eye(b_rows, b_cols,
                                    dtype='complex64')  # присвоить первоначальное значение G = B не работает !!!
        else:
            self.G_l_tilda = np.eye(b_rows, b_cols)
        max_elem_pr = np.inf
        n_iter = 0

        while abs(max_elem_pr - self.G_l_tilda.max()) > self.e:
            G_next = self.B[self.l_tilda] + np.dot(self.L[self.l_tilda], self.G_l_tilda) + \
                      np.dot(self.F[self.l_tilda], np.dot(self.G_l_tilda, self.G_l_tilda))
            max_elem_pr = self.G_l_tilda.max()
            self.G_l_tilda = G_next
            n_iter += 1

        if self.is_verbose:
            print(f"Number of iterations to calculate the matrix G_l_tilda = {n_iter}")

    def norm_mrx(self, mrx, is_max=True):
        """
        Calculate the norm of a matrix.
         :param mrx: input matrix
         :param is_max: if True, calculate max-norm, otherwise calculate Frobenius norm
         :return: norm of the matrix

        """
        rows = mrx.shape[0]
        cols = mrx.shape[1]
        if is_max:
            return max(sum(math.fabs(mrx[r, j].real) for j in range(cols)) for r in range(rows))
        
        return sum(math.fabs(mrx[r, j].real) for r in range(rows) for j in range(cols)) / (rows * cols)
    
    def Gr_tilda_calc(self):
        """
        Calculate the matrix Gr_tilda
         :return: None
        """

        B1 = self.Br[self.l_tilda][0]
        L1 = self.Lr[self.l_tilda][0]
        L = self.L[self.l_tilda]
        F = self.F[self.l_tilda]
        G = self.G_l_tilda
        F1 = self.Fr[self.l_tilda][0]
        
        GG = np.dot(G, G)
        L1G = np.dot(L1, G)

        G1 = B1 + L1G + np.dot(F1, GG)

        # вычисляем первый момент
        max_elem = self.norm_mrx(G1)
        max_elem_pr = 0
        n_iter = 0
        
        not_dep_on_G1 = copy.deepcopy(G1)

        while (abs(max_elem - max_elem_pr) > self.e):
            G1 = not_dep_on_G1 + np.dot(L, G1)  + np.dot(F, np.dot(G1, G)) \
                + np.dot(F, np.dot(G, G1))
            max_elem_pr = max_elem
            max_elem = self.norm_mrx(G1)
            n_iter += 1

        if self.is_verbose:
            print(
                f"Number of iterations for calculation matrix  G_r1_tilda = {n_iter}")

        self.Gr_tilda.append(G1)

        # вычисляем второй момент
        B2 = self.Br[self.l_tilda][1]
        L2 = self.Lr[self.l_tilda][1]
        F2 = self.Fr[self.l_tilda][1]
        
        L1G1 = np.dot(L1, G1)
        L2G = np.dot(L2, G)
        G1G = np.dot(G1, G)
        GG1 = np.dot(G, G1)

        G2 = B2 + L2G + 2 * L1G1 + np.dot(F2, GG) \
            + 2 * np.dot(F1, (G1G + GG1))

        max_elem = self.norm_mrx(G2)
        max_elem_pr = 0
        n_iter = 0
        
        not_dep_on_G2 = copy.deepcopy(G2)
        g1g1 = np.dot(G1, G1)
        while (abs(max_elem - max_elem_pr) > self.e):
            G2 = not_dep_on_G2 + np.dot(L, G2) + np.dot(F, (np.dot(G2, G) + 2 * g1g1 + np.dot(G, G2)))

            max_elem_pr = max_elem
            max_elem = self.norm_mrx(G2)
            n_iter += 1

        if self.is_verbose:
            print(
                f"Number of iterations for calculation matrix  G_r2_tilda = {n_iter}")

        self.Gr_tilda.append(G2)

        # вычисляем 3 момент
        B3 = self.Br[self.l_tilda][2]
        L3 = self.Lr[self.l_tilda][2]
        F3 = self.Fr[self.l_tilda][2]
        
        G1G2 = np.dot(G1, G2)
        G2G1 = np.dot(G2, G1)

        G3 = B3 + np.dot(L3, G) + 3 * np.dot(L2, G1) + 3.0 * np.dot(L1, G2) + np.dot(F3, GG) + \
            3.0 * np.dot(F2, (G1G + GG1)) + \
            3.0 * np.dot(F1, (np.dot(G2, G1) + 2 *
                         np.dot(G1, G1) + np.dot(G, G2)))

        max_elem = self.norm_mrx(G3)
        max_elem_pr = 0
        n_iter = 0
        
        not_dep_on_G3 = copy.deepcopy(G3)
        
        while (abs(max_elem - max_elem_pr) > self.e):
            G3 = not_dep_on_G3 + np.dot(L, G3)  + np.dot(F, (np.dot(G3, G) + 3 * G2G1 +
                             3 * G1G2 + np.dot(G, G3)))

            max_elem_pr = max_elem
            max_elem = self.norm_mrx(G3)
            n_iter += 1

        if self.is_verbose:
            print(
                f"Number of iterations for calculation matrix  G_r3_tilda = {n_iter}")
            
        self.Gr_tilda.append(G3)
        
    

    def G_calc(self):
        """
        Calculation of the matrix G for non-repeating matrices, l < l_tilda.
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

            G = copy.deepcopy(B)

            max_elem = self.norm_mrx(G)
            max_elem_pr = 0
            n_iter = 0
            while (abs(max_elem - max_elem_pr) > self.e):
                G = B + np.dot(L, G) + np.dot(F, np.dot(G_plus_one, G))
                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G)
                n_iter += 1

            if self.is_verbose:
                print(f"Number of iterations to calculate matrix  G{self.l_tilda - i} = {n_iter}")

            self.G[self.l_tilda - i - 1] = G

    def Gr_calc(self):
        """
        Calculation of Gr for non-repeating matrices, l < l_tilda
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
            
            L1G = np.dot(L1, G)
            FG_sum = np.dot(F1, np.dot(Gplus_one, G)
                                             ) + np.dot(F, np.dot(G1plus_one, G))

            G1 = B1 + L1G + FG_sum

            max_elem = self.norm_mrx(G1)
            max_elem_pr = 0
            n_iter = 0
            
            not_dep_on_G1 = copy.deepcopy(G1)
            
            while (abs(max_elem - max_elem_pr) > self.e):
                G1 = not_dep_on_G1 + np.dot(L, G1) + np.dot(F, np.dot(Gplus_one, G1))
                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G1)
                n_iter += 1
            self.Gr[self.l_tilda - i - 1].append(G1)

            if self.is_verbose:
                print(f"Number of iterations to calculate matrix  Gr{self.l_tilda - i}[0] = {n_iter}")


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
            
            not_dep_on_G2 = copy.deepcopy(G2)

            while (abs(max_elem - max_elem_pr) > self.e):
                G2 = not_dep_on_G2 + np.dot(L, G2) + np.dot(F, (np.dot(G2plus_one, G) + 2.0 *
                             np.dot(G1plus_one, G1) + np.dot(Gplus_one, G2)))
                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G2)
                n_iter += 1
            self.Gr[self.l_tilda - i - 1].append(G2)
            if self.is_verbose:
                print(f"Number of iterations to calculate matrix  Gr{self.l_tilda - i}[1] = {n_iter}")


            # вычисляем 3 момент
            B3 = self.Br[self.l_tilda - i - 1][2]
            L3 = self.Lr[self.l_tilda - i - 1][2]
            F3 = self.Fr[self.l_tilda - i - 1][2]
            G3plus_one = self.Gr[self.l_tilda - i][2]

            G3 = B3 + np.dot(L3, G) + 3.0 * np.dot(L2, G1) + 3.0 * np.dot(L1, G2) \
                + np.dot(F3, np.dot(Gplus_one, G)) \
                + 3.0 * np.dot(F2, (np.dot(G1plus_one, G) + np.dot(Gplus_one, G1))) \
                + 3.0 * np.dot(F1, (np.dot(G2plus_one, G) + 2.0 *
                                    np.dot(G1plus_one, G1) + np.dot(Gplus_one, G2)))

            max_elem = self.norm_mrx(G3)
            max_elem_pr = 0
            n_iter = 0
            
            not_dep_on_G3 = copy.deepcopy(G3)
            while (abs(max_elem - max_elem_pr) > self.e):
                G3 = not_dep_on_G3 +  np.dot(L, G3) \
                    + np.dot(F, (
                        np.dot(G3plus_one, G) + 3 * np.dot(G2plus_one, G1) + 3.0 * np.dot(G1plus_one, G2) + np.dot(
                             Gplus_one, G3)))

                max_elem_pr = max_elem
                max_elem = self.norm_mrx(G3)
                n_iter += 1
            self.Gr[self.l_tilda - i - 1].append(G3)
            if self.is_verbose:
                print(f"Number of iterations to calculate matrix  Gr{self.l_tilda - i}[2] = {n_iter}")

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
            FGG = np.dot(self.F[self.l_tilda], np.dot(
                self.G[self.l_tilda], self.G[self.l_tilda]))
        elif l_start == self.l_tilda - 1:
            BG = copy.deepcopy(self.B[l_start])
            LG = np.dot(self.L[l_start], self.G[l_start])
            FGG = np.dot(self.F[l_start], np.dot(
                self.G[self.l_tilda], self.G[l_start]))
        else:
            BG = copy.deepcopy(self.B[l_start])
            LG = np.dot(self.L[l_start], self.G[l_start])
            FGG = np.dot(self.F[l_start], np.dot(
                self.G[l_start + 1], self.G[l_start]))

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
                L1GG = np.dot(self.Lr[self.l_tilda][0], np.dot(
                    self.G[self.l_tilda], Ggap_minus_one))
            else:
                L1GG = np.dot(self.Lr[i][0], np.dot(self.G[i], Ggap_minus_one))

            # LG1G section
            if i >= self.l_tilda:
                LG1G = np.dot(self.L[self.l_tilda], np.dot(
                    self.Gr[self.l_tilda][0], Ggap_minus_one))
            else:
                LG1G = np.dot(self.L[i], np.dot(self.Gr[i][0], Ggap_minus_one))

            # LGG1 section
            if i >= self.l_tilda:
                LGG1 = np.dot(self.L[self.l_tilda], np.dot(
                    self.G[self.l_tilda], Gr_gap[0]))
            else:
                LGG1 = np.dot(self.L[i], np.dot(self.G[i], Gr_gap[0]))

            # F1GGG-section
            if i >= self.l_tilda:
                F1GGG = np.dot(self.Fr[self.l_tilda][0],
                               np.dot(self.G[self.l_tilda], np.dot(self.G[self.l_tilda], Ggap_minus_one)))
            elif i == self.l_tilda - 1:
                F1GGG = np.dot(self.Fr[i][0], np.dot(
                    self.G[self.l_tilda], np.dot(self.G[i], Ggap_minus_one)))
            else:
                F1GGG = np.dot(self.Fr[i][0], np.dot(
                    self.G[i + 1], np.dot(self.G[i], Ggap_minus_one)))

            # FG1GG-section
            if i >= self.l_tilda:
                FG1GG = np.dot(self.F[self.l_tilda],
                               np.dot(self.Gr[self.l_tilda][0], np.dot(self.G[self.l_tilda], Ggap_minus_one)))
            elif i == self.l_tilda - 1:
                FG1GG = np.dot(self.F[i], np.dot(
                    self.Gr[self.l_tilda][0], np.dot(self.G[i], Ggap_minus_one)))
            else:
                FG1GG = np.dot(self.F[i], np.dot(
                    self.Gr[i + 1][0], np.dot(self.G[i], Ggap_minus_one)))

            # FGG1G-section
            if i >= self.l_tilda:
                FGG1G = np.dot(self.F[self.l_tilda],
                               np.dot(self.G[self.l_tilda], np.dot(self.Gr[self.l_tilda][0], Ggap_minus_one)))
            elif i == self.l_tilda - 1:
                FGG1G = np.dot(self.F[i], np.dot(
                    self.G[self.l_tilda], np.dot(self.Gr[i][0], Ggap_minus_one)))
            else:
                FGG1G = np.dot(self.F[i], np.dot(
                    self.G[i + 1], np.dot(self.Gr[i][0], Ggap_minus_one)))

            # FGGG1-section
            if i >= self.l_tilda:
                FGGG1 = np.dot(self.F[self.l_tilda],
                               np.dot(self.G[self.l_tilda], np.dot(self.G[self.l_tilda], Gr_gap[0])))
            elif i == self.l_tilda - 1:
                FGGG1 = np.dot(self.F[i], np.dot(
                    self.G[self.l_tilda], np.dot(self.G[i], Gr_gap[0])))
            else:
                FGGG1 = np.dot(self.F[i], np.dot(
                    self.G[i + 1], np.dot(self.G[i], Gr_gap[0])))

            Gr_gap[0] = B1G + BG1 + L1GG + LG1G + \
                LGG1 + F1GGG + FG1GG + FGG1G + FGGG1
            Gr_gap[1] = Gr_gap[0]
            Gr_gap[2] = Gr_gap[0]

        return Gr_gap
