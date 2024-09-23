import numpy as np
import math
import cmath


class M_D_n:
    """
    Численный расчет многоканальной системы M/D/n
    с детерминированным обслуживанием
    """

    def __init__(self, l, b, n, e=1e-12, p_num=100):
        """
        l - интенсивность входящего потока
        b - время обслуживания в канале
        n - число каналов
        e - точность вычислений
        """
        self.l = l
        self.b = b
        self.n = n
        self.e = e
        self.p = [0.0] * p_num
        self.p_num = p_num

    def calc_p_up_to_n(self):
        self.get_z()
        A = np.zeros((self.n, self.n), dtype=complex)
        B = np.zeros(self.n, dtype=complex)
        row_num = 0
        for m in range(len(self.z_)):
            for j in range(self.n):
                right_z = np.power(self.z_[m], self.n).real
                delta_z = np.power(self.z_[m], j).real - right_z
                A[row_num, j] = delta_z
            row_num += 1
        for m in range(len(self.z_)):
            for j in range(self.n):
                right_z = np.power(self.z_[m], self.n).imag
                delta_z = np.power(self.z_[m], j).imag - right_z
                A[row_num, j] = delta_z
            row_num += 1

        for j in range(self.n):
            A[self.n - 1, j] = self.n - j

        B[self.n - 1] = self.n - self.l * self.b
        p = np.linalg.lstsq(A, B, rcond=1e-8)
        p_real = []
        for i in range(len(p[0])):
            p_real.append(p[0][i].real)
        return p_real

    def calc_p(self):
        p_up_to_n = self.calc_p_up_to_n()
        self.calc_q()
        summ = 0
        for i in range(len(p_up_to_n)):
            self.p[i] = p_up_to_n[i]
            summ += self.p[i]

        self.p[self.n] = self.p[0] / self.q_[0] - summ
        u = summ + self.p[self.n]

        is_negative = False
        for k in range(self.n + 1, self.p_num):
            if is_negative:
                break
            summ = 0
            for j in range(1, k - self.n):
                summ += self.q_[j] * self.p[k - j]
            value = (self.p[k - self.n] - u * self.q_[k - self.n] - summ) / self.q_[0]
            if value < 0:
                is_negative = True
            else:
                self.p[k] = value

        return self.p

    def calc_q(self):
        q0 = math.exp(-self.b * self.l)
        self.q_ = [0.0] * self.p_num
        self.q_[0] = q0
        for i in range(1, self.p_num):
            self.q_[i] = self.q_[i - 1] * (self.l * self.b) / i

    def get_z(self):
        """
        Нахождение корней z
        """
        z = []
        z_num = math.floor(self.n / 2)

        for i in range(z_num):
            z.append(complex(0, 0))

        for m in range(z_num):
            z[m] = 0.5 * cmath.exp(2.0 * (m + 1) * cmath.pi * complex(0, 1) / self.n)
            z_old = z[m]
            is_close = False
            while not is_close:
                left = 2 * (m + 1) * cmath.pi * complex(0, 1) / self.n
                right = self.l * self.b * (1.0 - z_old) / self.n
                z_new = cmath.exp(left - right)
                if math.fabs(z_new.real - z_old.real) < self.e:
                    is_close = True
                z_old = z_new
            z[m] = z_old
        self.z_ = z



