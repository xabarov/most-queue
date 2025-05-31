"""
Calculation of an M/H2/n queue with H2-warming, H2-cooling and H2-delay of the start of cooling 
using Takahasi-Takami method.
"""
import math
from itertools import chain

import numpy as np
from scipy.misc import derivative

from most_queue.rand_distribution import H2Distribution
from most_queue.theory.utils.binom_probs import calc_binom_probs
from most_queue.theory.utils.transforms import lst_exp as pls


class MGnH2ServingColdWarmDelay:
    """
    Calculation of an M/H2/n queue with H2-warming, H2-cooling and H2-delay of the start of cooling 
    using Takahasi-Takami method.
    Complex parameters are used. Complex parameters allow approximating distributions
    with arbitrary coefficients of variation (>1, <=1).
    """

    def __init__(self, l, b, b_warm, b_cold, b_cold_delay, n, buffer=None,
                 N=150, accuracy=1e-6, dtype="c16",
                 verbose=False, stable_w_pls=False, w_pls_dt=1e-3):
        """
        n: number of servers
        l: arrival rate
        b: initial moments of service time
        b_warm: initial moments of warming time
        b_cold: initial moments of cooling time
        b_cold_delay: initial moments of delay time before cooling starts
        N: number of levels for the Markov chain approximation
        accuracy: accuracy parameter for stopping iterations
        dtype: data type for calculations (default is complex16)
        verbose: flag to print intermediate results
        stable_w_pls: flag to use a more stable method for calculating w_plus
        w_pls_dt: time step for calculating w_plus when stable_w_pls is True
        """
        self.dt = np.dtype(dtype)
        if buffer:
            self.R = buffer + n
            self.N = self.R + 1
        else:
            self.N = N
            self.R = None

        self.e1 = accuracy
        self.n = n
        self.b = b
        self.verbose = verbose
        self.l = l
        self.stable_w_pls = stable_w_pls
        self.w_pls_dt = w_pls_dt

        if self.dt == 'c16':
            h2_params_service = H2Distribution.get_params_clx(b)
        else:
            h2_params_service = H2Distribution.get_params(b)

        # H2-parameters for service time
        self.y = [h2_params_service.p1, 1.0 - h2_params_service.p1]
        self.mu = [h2_params_service.mu1, h2_params_service.mu2]

        # H2-parameters for warm-up time
        self.b_warm = b_warm
        if self.dt == 'c16':
            h2_params_warm = H2Distribution.get_params_clx(b_warm)
        else:
            h2_params_warm = H2Distribution.get_params(b_warm)
        self.y_w = [h2_params_warm.p1, 1.0 - h2_params_warm.p1]
        self.mu_w = [h2_params_warm.mu1, h2_params_warm.mu2]

        # H2-parameters for cold-down time
        self.b_cold = b_cold
        if self.dt == 'c16':
            h2_params_cold = H2Distribution.get_params_clx(b_cold)
        else:
            h2_params_cold = H2Distribution.get_params(b_cold)
        self.y_c = [h2_params_cold.p1, 1.0 - h2_params_cold.p1]
        self.mu_c = [h2_params_cold.mu1, h2_params_cold.mu2]

        # H2-parameters for cold-down delay time
        self.b_cold_delay = b_cold_delay
        if self.dt == 'c16':
            h2_params_cold_delay = H2Distribution.get_params_clx(b_cold_delay)
        else:
            h2_params_cold_delay = H2Distribution.get_params(b_cold_delay)
        self.y_c_delay = [h2_params_cold_delay.p1,
                          1.0 - h2_params_cold_delay.p1]
        self.mu_c_delay = [h2_params_cold_delay.mu1, h2_params_cold_delay.mu2]

        self.cols = [] * N

        # Takahasi-Takami method parameters
        self.t = []
        self.b1 = []
        self.b2 = []
        if self.dt == 'c16':
            self.x = np.array([0.0 + 0.0j] * N)
            self.z = np.array([0.0 + 0.0j] * N)
            # Probabilities of states to be searched for
            self.p = np.array([0.0 + 0.0j] * N)
        else:
            self.x = np.array([0.0] * N)
            self.z = np.array([0.0] * N)
            # Probabilities of states to be searched for
            self.p = np.array([0.0] * N)

        self.num_of_iter_ = 0  # number of iterations of the algorithm

        # Transition matrices for the Takahasi-Takami method
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Y = []

        for i in range(N):

            if i < n + 1:
                if i == 0:
                    # 00 state + cold_delay_1 + cold_delay_2 + cold_1 + cold_2
                    self.cols.append(5)
                else:
                    # w1 w2 + normal H2 states + cold1 + cold2
                    self.cols.append(i + 5)
            else:
                self.cols.append(n + 5)

            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self._build_matrices()
        self._initial_probabilities()

    def get_probs_of_servers_busy(self):
        """
        Return probabilities of servers being busy.
        prob[i] - probability that i number of servers is busy.
        """

        probs = np.zeros((self.n + 1), dtype=self.dt)

        # servers are not busy, when system in [0,0] state, in cooling, and warming up states.

        probs[0] += self.Y[0][0, 0] + self.Y[0][0, 3] + self.Y[0][0, 4]

        # when cooling delay, 1 server is running

        probs[1] += self.Y[0][0, 1] + self.Y[0][0, 2] 
        
        for k in range(1, self.n):
            probs[k] += np.sum(self.Y[k][0, 2:k+3])

        for k in range(1, self.N):
            # warm-ups
            for i in range(2):
                probs[0] += self.Y[k][0, i]

            # coolings
            probs[0] += self.Y[k][0, -1] + self.Y[k][0, -2]

        probs[-1] = 1.0 - np.sum(probs)
        probs = probs.tolist()
        return [prob.real for prob in probs]


    def run(self):
        """
        Запускает расчет
        """
        if self.dt == 'c16':
            self.b1[0][0, 0] = 0.0 + 0.0j
            self.b2[0][0, 0] = 0.0 + 0.0j
            x_max2 = 0.0 + 0.0j
        else:
            self.b1[0][0, 0] = 0.0
            self.b2[0][0, 0] = 0.0
            x_max1 = 0.0
            x_max2 = 0.0

        x_max1 = np.max(self.x)

        self._calc_support_matrices()

        self.num_of_iter_ = 0  # кол-во итераций алгоритма

        while math.fabs(x_max2.real - x_max1.real) >= self.e1:
            x_max2 = x_max1
            self.num_of_iter_ += 1

            for j in range(1, self.N):  # по всем ярусам, кроме первого.

                # b':
                self.b1[j] = np.dot(self.t[j - 1], self.AG[j])

                # b":
                if j != (self.N - 1):
                    self.b2[j] = np.dot(self.t[j + 1], self.BG[j])
                else:
                    self.b2[j] = np.dot(self.t[j - 1], self.BG[j])

                c = self._calculate_c(j)

                x_znam = np.dot(c, self.b1[j]) + self.b2[j]
                if self.dt == 'c16':
                    self.x[j] = 0.0 + 0.0j
                else:
                    self.x[j] = 0.0
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                if self.dt == 'c16':
                    self.x[j] = (1.0 + 0.0j) / self.x[j]
                else:
                    self.x[j] = (1.0) / self.x[j]

                if self.R and j == (self.N - 1):
                    tA = np.dot(self.t[j - 1], self.A[j - 1])
                    tag = np.dot(tA, self.G[j])
                    tag_sum = 0
                    for t_i in range(tag.shape[1]):
                        tag_sum += tag[0, t_i]
                    self.z[j] = 1.0 / tag_sum
                    self.t[j] = self.z[j] * tag

                else:

                    self.z[j] = np.dot(c, self.x[j])
                    self.t[j] = np.dot(self.z[j], self.b1[j]) + \
                        np.dot(self.x[j], self.b2[j])

            if self.dt == 'c16':
                self.x[0] = (1.0 + 0.0j) / self.z[1]
            else:
                self.x[0] = 1.0 / self.z[1]

            self.t[0] = self.x[0]*(np.dot(self.t[1], self.B[1]).dot(self.G[0]))

            x_max1 = np.max(self.x)

            if self.verbose:
                print(f"End iter # {self.num_of_iter_}")

        self._calculate_p()
        self._calculate_y()

    def get_p(self) -> list[float]:
        """
        Get probabilities of states.
        """
        return [prob.real for prob in self.p]

    def get_cold_prob(self):
        """
        Get probability of being in the cold state.
        """
        p_cold = 0
        for k in range(self.N):
            p_cold += self.Y[k][0, -1] + self.Y[k][0, -2]
        return p_cold.real

    def get_warmup_prob(self):
        """
        Get probability of being in the warmup state.
        """
        p_warmup = 0
        for k in range(1, self.N):
            p_warmup += self.Y[k][0, 0] + self.Y[k][0, 1]
        return p_warmup.real

    def get_cold_delay_prob(self):
        """
        Get probability of being in the cold delay state.
        """
        p_cold_delay = self.Y[0][0, 1] + self.Y[0][0, 2]
        return p_cold_delay.real

    def get_idle_prob(self):
        """
        Get the probability of the server being idle.
        """
        return self.Y[0][0, 0]

    def get_w(self):
        """
        Get first three moments of waiting time in the queue.
        """
        w = [0.0] * 3

        for i in range(3):
            if self.stable_w_pls:
                max_mu = np.max(list(chain(np.array(self.mu_w).astype('float'), np.array(self.mu_c).astype('float'),
                                           np.array(self.mu).astype('float'))))

                dx = self.w_pls_dt / max_mu
            else:
                dx = self.w_pls_dt
            w[i] = derivative(self._calc_w_pls, 0, dx=dx, n=i + 1, order=9)

        w = [w_moment.real if isinstance(
            w_moment, complex) else w_moment for w_moment in w]
        return [-w[0].real, w[1].real, -w[2].real]

    def get_v(self):
        """
        Get first three moments of sojourn time in the queue.
        """
        v = [0.0] * 3
        w = self.get_w()
        b = self.b
        v[0] = w[0] + b[0]
        v[1] = w[1] + 2 * w[0] * b[0] + b[1]
        v[2] = w[2] + 3 * w[1] * b[0] + 3 * w[0] * b[1] + b[2]

        v = [v_moment.real if isinstance(
            v_moment, complex) else v_moment for v_moment in v]
        return v

    def _get_key_numbers(self, level):
        key_numbers = []
        for i in range(level + 1):
            key_numbers.append((self.n - i, i))
        return np.array(key_numbers, dtype=self.dt)

    def _calc_down_probs(self, from_level):
        if from_level == self.n:
            return np.eye(self.n + 1)
        b_matrix = self.B[from_level]
        probs = []
        for i in range(2, self.cols[from_level - 1] - 2):
            probs_on_level = []
            for j in range(2, self.cols[from_level - 1] - 2):
                if from_level != 1:
                    probs_on_level.append(b_matrix[i, j] / sum(b_matrix[i, :]))
                else:
                    probs_on_level.append(b_matrix[i, 0] / sum(b_matrix[i, :]))
            probs.append(probs_on_level)
        return np.array(probs, dtype=self.dt)

    def _calc_w_pls(self, s):
        w = 0

        # Calculate Laplace–Stieltjes transform in advance
        mu_w_pls = np.array(
            [pls(self.mu_w[0], s), pls(self.mu_w[1], s)])
        mu_c_pls = np.array(
            [pls(self.mu_c[0], s), pls(self.mu_c[1], s)])

        # Комбо переходов: охлаждение + разогрев
        # [i,j] = охлаждение в i, переход из состояния i охлаждения в j состояние разогрева, разогрев j
        c_to_w = np.zeros((2, 2), dtype=self.dt)
        for c_phase in range(2):
            for w_phase in range(2):
                c_to_w[c_phase, w_phase] = mu_c_pls[c_phase] * \
                    self.y_w[w_phase] * mu_w_pls[w_phase]

        # Если заявка попала в состояние [0] ей придется подождать окончание разогрева
        w += self.Y[0][0, 0] * \
            (self.y_w[0] * mu_w_pls[0] + self.y_w[1] * mu_w_pls[1])

        # Если заявка попала в фазу разогрева, хотя каналы свободны,
        # ей придется подождать окончание разогрева
        for k in range(1, self.n):
            for i in range(2):
                w += self.Y[k][0, i] * mu_w_pls[i]

        # Если заявка попала в фазу охлаждения, хотя каналы свободны,
        # ей придется подождать окончание охлаждения и разогрева
        for k in range(0, self.n):
            if k == 0:
                # Первый уровень. Здесь фазы такие: [0] [+][-]_delay (+)(-)_cold.
                # Поэтому у Y смещение + 3
                for i in range(2):
                    # Переход в [0] состояние, а из него -> в разогрев
                    w += self.Y[k][0, i + 3] * mu_c_pls[i] * \
                        (self.y_w[0] * mu_w_pls[0] + self.y_w[1] * mu_w_pls[1])
            else:
                cold_pos = k + 3
                for c_phase in range(2):
                    for w_phase in range(2):
                        w += self.Y[k][0, cold_pos + c_phase] * \
                            c_to_w[c_phase, w_phase]

        P = self._calc_down_probs(self.n + 1)
        # ключи яруса n, для n=3 [(3,0) (2,1) (1,2) (0,3)]
        key_numbers = self._get_key_numbers(self.n)
        a = np.array(
            [pls(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1], s) for j in
             range(self.n + 1)])
        waits_service_on_level_before = None

        # вероятности попадания в состояния обслуживания, вычисляются с помощью биноминального распределения
        probs = calc_binom_probs(self.n + 1, self.y[0])

        for k in range(self.n, self.N):

            # Если заявка попала в фазу разогрева и каналы заняты. Также есть k-n заявок в очереди
            # ей придется подождать окончание разогрева + обслуживание всех накопленных заявок

            if k == self.n:
                # переходы вниз считать не нужно

                # попала в фазу обслуживания
                for i in range(self.n + 1):
                    w += self.Y[k][0, i + 2] * a[i]

                # Взвешанное по биномиальным вероятностям обслуживание
                waits_service_on_level_before = a

                # взвешенные по вероятностям перехода из состояния разогрева
                pls_service_total = sum(a * probs)
                # попала в фазу разогрева
                for i in range(2):
                    w += self.Y[k][0, i] * mu_w_pls[i] * pls_service_total

                # попала в фазу охлаждения - охлаждение + разогрев + обслуживание
                cold_pos = self.n + 3

                for c_phase in range(2):
                    for w_phase in range(2):
                        w += self.Y[k][0, cold_pos + c_phase] * \
                            pls_service_total * c_to_w[c_phase, w_phase]

            else:

                Pa_before = np.dot(P,
                                   # вектор - условные вероятности * w на предыдущем слое
                                   waits_service_on_level_before.T)
                # вектор размерности числа состояний стандартного обслуживания.
                aPa_before = a * Pa_before
                # Каждый элемент - обслуживание в случае нахожения в данной позиции

                waits_service_on_level_before = aPa_before

                # попала в фазу обслуживания
                for i in range(self.n + 1):
                    w += self.Y[k][0, i + 2] * aPa_before[i]

                pls_service_total = sum(
                    aPa_before * probs)  # взвешенные по вероятностям перехода из состояния разогрева

                for i in range(2):
                    w += self.Y[k][0, i] * mu_w_pls[i] * pls_service_total

                # попала в фазу охлаждения - охлаждение + разогрев + обслуживание
                cold_pos = self.n + 3

                for c_phase in range(2):
                    for w_phase in range(2):
                        w += self.Y[k][0, cold_pos + c_phase] * \
                            c_to_w[c_phase, w_phase] * pls_service_total

        return w

    def _calc_serv_coev(self):
        b = self.b
        D = b[1] - b[0] * b[0]
        return math.sqrt(D) / b[0]

    def _initial_probabilities(self):
        """
        Initialize probabilities of microstates
        """
        # set initial probabilities of microstates to be uniform
        for i in range(self.N):
            self.t[i][0] = [1.0 / self.cols[i]] * self.cols[i]

        # ro = self.l * self.b[0] / self.n
        # va = 1.0  # M/
        # vb = self._calc_serv_coev()
        # self.x[0] = pow(ro, 2.0 / (va * va + vb * vb))

        self.x[0] = 0.4

    def _norm_probs(self):
        """
        Normalize probabilities of microstates to sum up to 1.0
        """
        total_prob = sum(self.p)
        self.p = [prob / total_prob for prob in self.p]

        if self.verbose:
            print(f"Summ of probs = {sum(self.p):.5f}")

    def _calculate_p(self):
        """
        Calculate probabilities of microstates
        """
        # version 1
        p_sum = 0
        p0_max = 1.0
        p0_min = 0.0

        while math.fabs(1.0 - p_sum.real) > 1e-6:
            p0_ = (p0_max + p0_min) / 2.0
            p_sum = p0_
            self.p[0] = p0_
            for j in range(self.N - 1):
                self.p[j + 1] = self.p[j] * self.x[j]
                p_sum += self.p[j + 1]

            if (p_sum > 1.0):
                p0_max = p0_
            else:
                p0_min = p0_

        self._norm_probs()

    def _calculate_y(self):
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))

    def _build_matrices(self):
        """
        Bulds matrices A, B, C, D
        """
        for i in range(self.N):
            self.A.append(self._build_big_a_matrix(i))
            self.B.append(self._build_big_b_matrix(i))
            self.C.append(self._build_big_c_matrix(i))
            self.D.append(self._build_big_d_matrix(i))

    def _calc_g_matrices(self):
        self.G = []
        for j in range(0, self.N):
            self.G.append(np.linalg.inv(self.D[j] - self.C[j]))

    def _calc_ag_matrices(self):
        self.AG = [0]
        for j in range(1, self.N):
            self.AG.append(np.dot(self.A[j - 1], self.G[j]))

    def _calc_bg_matrices(self):
        self.BG = [0]
        for j in range(1, self.N):
            if j != (self.N - 1):
                self.BG.append(np.dot(self.B[j + 1], self.G[j]))
            else:
                self.BG.append(np.dot(self.B[j], self.G[j]))

    def _calc_support_matrices(self):
        self._calc_g_matrices()
        self._calc_ag_matrices()
        self._calc_bg_matrices()

    def _calculate_c(self, j):
        """
        Calculate value of variable c participating in the calculation.
        """
        m = np.dot(self.b2[j], self.B[j])
        chisl = np.sum(m[0])

        m = np.dot(self.b1[j], self.B[j])
        znam2 = np.sum(m[0])

        m = np.dot(self.t[j - 1], self.A[j - 1])
        znam = np.sum(m[0])

        return chisl / (znam - znam2)

    def _insert_standart_A_into(self, mass, l, y1, left_pos, bottom_pos, level):
        row_num = level
        for i in range(row_num):
            mass[i + left_pos, i + bottom_pos] = l * y1
            mass[i + left_pos, i + bottom_pos + 1] = l * (1.0 - y1)

    def _build_big_a_matrix(self, num):
        if num < self.n:
            col = self.cols[num + 1]
            row = self.cols[num]
        else:
            col = self.cols[self.n]
            row = self.cols[self.n]

        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.A[self.n]
            return output
        if num == 0:
            # to warm phase
            output[0, 0] = self.l * self.y_w[0]
            output[0, 1] = self.l * self.y_w[1]
            # cold phase
            output[3, 4] = self.l
            output[4, 5] = self.l
            # delay phase
            output[1, 2] = self.l * self.y[0]
            output[1, 3] = self.l * self.y[1]
            output[2, 2] = self.l * self.y[0]
            output[2, 3] = self.l * self.y[1]

        else:

            if num < self.n:
                # warm block
                output[0, 0] = self.l
                output[1, 1] = self.l
                # second
                self._insert_standart_A_into(
                    output, self.l, self.y[0], 2, 2, level=num + 1)
                # cold block
                output[-2, -2] = self.l
                output[-1, -1] = self.l
            else:
                for i in range(row):
                    output[i, i] = self.l

        return output

    def _insert_standart_B_into(self, mass, y, mu, left_pos, bottom_pos, level, n):
        col = level
        for i in range(col):
            if level <= n:
                mass[i + left_pos, i + bottom_pos] = (level - i) * mu[0]
                mass[i + left_pos + 1, i + bottom_pos] = (i + 1) * mu[1]
            else:
                mass[i + left_pos, i +
                     bottom_pos] = (level - i - 1) * mu[0] * y[0] + i * mu[1] * y[1]
                if i != level - 1:
                    mass[i + left_pos, i + bottom_pos +
                         1] = (level - i - 1) * mu[0] * y[1]
                if i != level - 1:
                    mass[i + + left_pos + 1, i +
                         bottom_pos] = (i + 1) * mu[1] * y[0]

    def _build_big_b_matrix(self, num):
        if num == 0:
            return np.zeros((1, 1), dtype=self.dt)

        if num <= self.n:

            col = self.cols[num - 1]
            row = self.cols[num]
        else:
            col = self.cols[self.n + 1]
            row = self.cols[self.n + 1]

        output = np.zeros((row, col), dtype=self.dt)
        if num > self.n + 1:
            output = self.B[self.n + 1]
            return output

        if num == 1:
            # to cold delay phases
            output[2, 1] = self.mu[0] * self.y_c_delay[0]
            output[2, 2] = self.mu[0] * self.y_c_delay[1]

            output[3, 1] = self.mu[1] * self.y_c_delay[0]
            output[3, 2] = self.mu[1] * self.y_c_delay[1]

        else:

            if num < self.n + 1:
                self._insert_standart_B_into(
                    output, self.y, self.mu, 2, 2, num, self.n)
            else:
                self._insert_standart_B_into(
                    output, self.y, self.mu, 2, 2, num, self.n)

        return output

    def _build_big_c_matrix(self, num):
        if num < self.n:
            col = self.cols[num]
            row = col
        else:
            col = self.cols[self.n]
            row = col

        output = np.zeros((row, col), dtype=self.dt)
        if num > self.n:
            output = self.C[self.n]
            return output

        if num == 0:
            # serving cold
            output[3, 0] = self.mu_c[0]
            output[4, 0] = self.mu_c[1]
            # delay to cold
            output[1, 3] = self.mu_c_delay[0] * self.y_c[0]
            output[1, 4] = self.mu_c_delay[0] * self.y_c[1]
            output[2, 3] = self.mu_c_delay[1] * self.y_c[0]
            output[2, 4] = self.mu_c_delay[1] * self.y_c[1]
        else:
            probs = calc_binom_probs(num + 1, self.y[0])
            # warm up
            for i, prob in enumerate(probs):
                output[0, 2 + i] = self.mu_w[0] * prob
                output[1, 2 + i] = self.mu_w[1] * prob
            # cold
            output[-2, 0] = self.mu_c[0] * self.y_w[0]
            output[-2, 1] = self.mu_c[0] * self.y_w[1]

            output[-1, 0] = self.mu_c[1] * self.y_w[0]
            output[-1, 1] = self.mu_c[1] * self.y_w[1]

        return output

    def _build_big_d_matrix(self, num):
        if num < self.n:
            col = self.cols[num]
            row = col
        else:
            col = self.cols[self.n]
            row = col

        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.D[self.n]
            return output

        for i in range(row):
            if self.dt == 'c16':
                sumA = 0.0 + 0.0j
                sumB = 0.0 + 0.0j
                sumC = 0.0 + 0.0j
            else:
                sumA = 0.0
                sumB = 0.0
                sumC = 0.0

            for j in range(self.cols[num + 1]):
                sumA += self.A[num][i, j]

            if num != 0:
                for j in range(self.cols[num - 1]):
                    sumB += self.B[num][i, j]

            for j in range(col):
                sumC += self.C[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output
