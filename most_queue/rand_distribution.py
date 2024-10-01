import cmath
import math

import numpy as np
import scipy.special as sp
from scipy import stats


class Normal_dist:
    def __init__(self, params, generator=None):
        """
        Принимает список параметров в следующей последовательности:
        mean - среднее значение
        sko - СКО
        """

        self.mean = params[0]
        self.sko = params[1]
        self.type = 'Normal'
        self.generator = generator

    def generate(self):
        """
        Генерация псевдо-случайных чисел
        """

        return self.generate_static(self.mean, self.sko, self.generator)

    @staticmethod
    def generate_static(mean, sko, generator=None):
        """
        Генерация псевдо-случайных чисел
        Статический метод
        """
        if generator:
            return generator.normal(mean, sko)
        return np.random.normal(mean, sko)


class Uniform_dist:
    def __init__(self, params, generator=None):
        """
        Принимает список параметров в следующей последовательности:
        mean - среднее значение
        half_interval - полуинтервал влево и вправо от среднего
        """

        self.mean = params[0]
        self.half_interval = params[1]
        self.type = 'Uniform'
        self.generator = generator

    def generate(self):
        """
        Генерация псевдо-случайных чисел, подчиненных гиперэкспоненциальному распределению 2-го порядка.
        Вызов из экземпляра класса
        """

        return self.generate_static(self.mean, self.half_interval, self.generator)

    @staticmethod
    def generate_static(mean, half_interval, generator=None):
        """
        Генерация псевдо-случайных чисел, подчиненных равномерному распределению
        Статический метод
        """
        if generator:
            return generator.uniform(mean - half_interval, mean + half_interval)
        return np.random.uniform(mean - half_interval, mean + half_interval)

    @staticmethod
    def calc_theory_moments(mean, half_interval, num=3):
        """
        Метод вычисляет теоретические моменты
        """

        f = [0.0] * num
        for i in range(num):
            f[i] = (pow(mean + half_interval, i + 2) - pow(mean - half_interval, i + 2)) / (2 * half_interval * (i + 2))
        return f

    @staticmethod
    def get_params(moments):
        """
        Подбор параметров распределения по заданным начальным моментам.
        Возвращает среднее и полуинтервал
        """

        D = moments[1] - moments[0] * moments[0]
        mean = moments[0]
        half_interval = math.sqrt(3 * D)

        return [mean, half_interval]

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev):
        """
        Подбор параметров распределения по среднему и коэфф вариации
        Возвращает среднее и полуинтервал
        """

        D = pow(coev * f1, 2)
        half_interval = math.sqrt(3 * D)

        return f1, half_interval

    @staticmethod
    def get_pdf(mean, half_interval, t):
        """
        Возвращает значение функции плотности распределения вероятностей СВ
        """

        a = mean - half_interval
        b = mean + half_interval
        if t < a or t > b:
            return 0
        return 1.0 / (b - a)

    @staticmethod
    def get_cdf(mean, half_interval, t):
        """
        Возвращает значение функции распределения СВ
        """

        a = mean - half_interval
        b = mean + half_interval
        if t < a:
            return 0
        if t > b:
            return 1

        return (t - a) / (b - a)

    @staticmethod
    def get_tail(mean, half_interval, t):
        """
        Возвращает значение ДФР
        """

        return 1.0 - Uniform_dist.get_cdf(mean, half_interval, t)


class H2_dist:
    def __init__(self, params, generator=None):
        """
        Принимает список параметров в следующей последовательности - y1, mu1, mu2
        """

        self.y1 = params[0]
        self.m1 = params[1]
        self.m2 = params[2]
        self.params = params
        self.type = 'H'
        self.generator = generator

    def generate(self):
        """
        Генерация псевдо-случайных чисел, подчиненных гиперэкспоненциальному распределению 2-го порядка.
        Вызов из экземпляра класса
        """

        return self.generate_static(self.y1, self.m1, self.m2, self.generator)

    @staticmethod
    def generate_static(y1, m1, m2, generator=None):
        """
        Генерация псевдо-случайных чисел, подчиненных гиперэкспоненциальному распределению 2-го порядка.
        Статический метод
        """

        if generator:
            r = generator.random()
            res = -np.log(generator.random())
        else:
            r = np.random.rand()
            res = -np.log(np.random.rand())

        if r < y1:
            if m1 != 0:
                res = res / m1
            else:
                res = 1e10
        else:
            if m2 != 0:
                res = res / m2
            else:
                res = 1e10
        return res

    @staticmethod
    def calc_theory_moments(y1, m1, m2, num=3):
        """
        Метод вычисляет теоретические моменты H2 распределения
        """
        f = [0.0] * num
        y2 = 1.0 - y1
        for i in range(num):
            f[i] = math.factorial(i + 1) * (y1 / pow(m1, i + 1) + y2 / pow(m2, i + 1))
        return f

    @staticmethod
    def get_residual_params(params):
        y1 = params[0]
        y2 = 1.0 - y1
        mu1 = params[1]
        mu2 = params[2]

        res = []
        res.append(y1 * mu2 / (y1 * mu2 + y2 * mu1))
        res.append(mu1)
        res.append(mu2)

        return res

    @staticmethod
    def get_params(moments):
        """
        Метод Алиева для подбора параметров H2 распределения по заданным начальным моментам.
        Подбирает параметры только принадлежащие множеству R (не комплексные)
        Возвращает список с параметрами [y1, mu1, mu2]
        """

        v = moments[1] - moments[0] * moments[0]
        v = math.sqrt(v) / moments[0]
        res = [0.0] * 3
        if v < 1.0:
            return res

        q_max = 2.0 / (1.0 + v * v)
        t_min = 1.5 * ((1 + v * v) ** 2) * math.pow(moments[0], 3)
        q_min = 0.0
        tn = 0.0

        if t_min > moments[2]:
            # one phase distibution
            q_new = q_max
            mu1 = (1.0 - math.sqrt(q_new * (v * v - 1.0) / (2 * (1.0 - q_new)))) * moments[0]
            if math.isclose(mu1, 0):
                mu1 = 1e10
            else:
                mu1 = 1.0 / mu1
            res[0] = q_max
            res[1] = mu1
            res[2] = 1e6
            return res
        else:
            max_iteration = 10000
            tec = 0
            t1 = 0
            t2 = 0
            while abs(tn - moments[2]) > 1e-8 and tec < max_iteration:
                tec += 1
                q_new = (q_max + q_min) / 2.0
                t1 = (1.0 + math.sqrt((1.0 - q_new) * (v * v - 1.0) / (2 * q_new))) * moments[0]
                t2 = (1.0 - math.sqrt(q_new * (v * v - 1.0) / (2 * (1.0 - q_new)))) * moments[0]

                tn = 6 * (q_new * math.pow(t1, 3) + (1.0 - q_new) * math.pow(t2, 3))

                if tn - moments[2] > 0:
                    q_min = q_new
                else:
                    q_max = q_new
            res[0] = q_max
            res[1] = 1.0 / t1
            res[2] = 1.0 / t2
            return res

    @staticmethod
    def get_params_clx(moments, verbose=True, ee=0.001, e=0.02, e_percent=0.15, is_fitting=True):
        """
        Метод подбора параметров H2 распределения по заданным начальным моментам.
        Допускает комплексные значения параметров при коэффициенте вариации <1
        ee - точность проверки близости распределения к E1 и E2
        e - множитель моментов
        e_percent - процент повышения множителя e
        Возвращает список с параметрами [y1, mu1, mu2]
        """

        f = [0.0] * 3
        for i in range(3):
            f[i] = complex(moments[i] / math.factorial(i + 1))
        znam = (f[1] - pow(f[0], 2))
        c0 = (f[0] * f[2] - pow(f[1], 2)) / znam
        c1 = (f[0] * f[1] - f[2]) / znam

        d = pow(c1 / 2.0, 2.0) - c0

        if is_fitting:
            # проверка на близость распределения к экспоненциальному
            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
            if math.fabs(coev.real - 1.0) < ee:
                if verbose:
                    print("H2 is close to Exp. Multiply moments to (1+je), coev = {0:5.3f},"
                          " e = {1:5.3f}.".format(coev, e))
                f = []
                for i in range(len(moments)):
                    f.append(moments[i] * complex(1, (i + 1) * e))

                return H2_dist.get_params_clx(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                              is_fitting=is_fitting)

            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]

            # проверка на близость распределения к Эрланга 2-го порядка
            if math.fabs(coev.real - 1.0 / math.sqrt(2.0)) < ee:
                if verbose:
                    print("H2 is close to E2. Multiply moments to (1+je), coev = {0:5.3f},"
                          " e = {1:5.3f}.".format(coev, e))
                f = []
                for i in range(len(moments)):
                    # if i == 0:
                    #     f.append(moments[0])
                    # else:
                    f.append(moments[i] * complex(1, (i + 1) * e))
                return H2_dist.get_params_clx(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                              is_fitting=is_fitting)

        res = [0, 0, 0]  # y1, mu1, mu2
        c1 = complex(c1)
        x1 = -c1 / 2 + cmath.sqrt(d)
        x2 = -c1 / 2 - cmath.sqrt(d)
        y1 = (f[0] - x2) / (x1 - x2)
        res[0] = y1
        res[1] = 1 / x1
        res[2] = 1 / x2

        return res

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev, is_clx=False):
        """
        Подбор параметров H2 распределения по среднему и коэффициенту вариации
        Возвращает список с параметрами [y1, mu1, mu2]
        """

        f = [0, 0, 0]
        alpha = 1 / (coev ** 2)
        f[0] = f1
        f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)
        if is_clx:
            return H2_dist.get_params_clx(f)
        return H2_dist.get_params(f)

    @staticmethod
    def get_cdf(params, t):
        """
        Возвращает значение функции распределения СВ
        """
        if t < 0:
            return 0
        y = [params[0], 1 - params[0]]
        mu = [params[1], params[2]]
        res = 0
        for i in range(2):
            res += y[i] * math.exp(-mu[i] * t)
        return 1.0 - res

    @staticmethod
    def get_pdf(params, t):
        """
        Возвращает значение функции плотности распределения вероятностей СВ
        """
        if t < 0:
            return 0
        y = [params[0], 1 - params[0]]
        mu = [params[1], params[2]]
        res = 0
        for i in range(2):
            res += y[i] * mu[i] * math.exp(-mu[i] * t)
        return res

    @staticmethod
    def get_tail(params, t):
        """
        Возвращает значение ДФР
        """
        return 1.0 - H2_dist.get_cdf(params, t)


class Cox_dist:
    """
    Распределение Кокса 2-го порядка
    """

    def __init__(self, params, generator=None):
        """
        Принимает список параметров в следующей последовательности - y1, mu1, mu2
        """
        self.y1 = params[0]
        self.m1 = params[1]
        self.m2 = params[2]
        self.params = params
        self.type = 'C'
        self.generator = generator

    def generate(self):
        """
        Генерация псевдо-случайных чисел, подчиненных распределению Кокса 2-го порядка. Вызов из экземпляра класса
        """
        return self.generate_static(self.y1, self.m1, self.m2, self.generator)

    @staticmethod
    def generate_static(y1, m1, m2, generator):
        """
        Генерация псевдо-случайных чисел, подчиненных распределению Кокса 2-го порядка. Статический метод
        """
        if generator:
            r = generator.random()
            res = (-1.0 / m1) * np.log(generator.random())
        else:
            r = np.random.rand()
            res = (-1.0 / m1) * np.log(np.random.rand())
        if r < y1:
            if generator:
                res = res + (-1.0 / m2) * np.log(generator.random())
            else:
                res = res + (-1.0 / m2) * np.log(np.random.rand())
        return res

    @staticmethod
    def calc_theory_moments(y1, m1, m2):
        """
        Метод вычисляет теоретические начальные моменты распределения Кокса 2-го порядка
        """

        y2 = 1.0 - y1
        f = [0.0] * 3
        f[0] = y2 / m1 + y1 * (1.0 / m1 + 1.0 / m2)
        f[1] = 2.0 * (y2 / math.pow(m1, 2) + y1 * (1.0 / math.pow(m1, 2) + 1.0 / (m1 * m2) + 1.0 / math.pow(m2, 2)))
        f[2] = 6.0 * (y2 / (math.pow(m1, 3)) + y1 * (1.0 / math.pow(m1, 3) + 1.0 / (math.pow(m1, 2) * m2) +
                                                     1.0 / (math.pow(m2, 2) * m1) + 1.0 / math.pow(m2, 3)))

        return f

    @staticmethod
    def get_params(moments, ee=0.001, e=0.5, e_percent=0.25, verbose=True, is_fitting=True):
        """
        Метод вычисляет параметры распределения Кокса 2-го порядка по трем заданным начальным моментам [moments]
        Возвращает список с параметрами [y1, mu1, mu2]
        """
        f = [0.0] * 3

        if is_fitting:
            # проверка на близость распределения к экспоненциальному
            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
            if abs(moments[1] - moments[0] * moments[0]) < ee:
                if verbose:
                    print("Cox special 1. Multiply moments to (1+je), coev = {0:5.3f},"
                          " e = {1:5.3f}.".format(coev, e))
                f = []
                for i in range(len(moments)):
                    f.append(moments[i] * complex(1, (i + 1) * e))

                return Cox_dist.get_params(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                           is_fitting=is_fitting)

            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]

            # проверка на близость распределения к Эрланга 2-го порядка
            if abs(moments[1] - (3.0 / 4) * moments[0] * moments[0]) < ee:
                if verbose:
                    print("Cox special 2. Multiply moments to (1+je), coev = {0:5.3f},"
                          " e = {1:5.3f}.".format(coev, e))
                f = []
                for i in range(len(moments)):
                    # if i == 0:
                    #     f.append(moments[0])
                    # else:
                    f.append(moments[i] * complex(1, (i + 1) * e))
                return Cox_dist.get_params(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                           is_fitting=is_fitting)

        # # особые случаи:
        # if abs(moments[1] - moments[0] * moments[0]) < ee:
        #     print("Cox get params. Special case 1")
        #     return 0.0, 1.0 / moments[0], 0.0
        #
        # if abs(moments[1] - (3.0 / 4) * moments[0] * moments[0]) < ee:
        #     print("Cox get params. Special case 2")
        #     return 1.0, 2.0 / moments[0], 2.0 / moments[0]

        for i in range(3):
            f[i] = moments[i] / math.factorial(i + 1)

        d = np.power(f[2] - f[0] * f[1], 2) - 4.0 * (f[1] - np.power(f[0], 2)) * (f[0] * f[2] - np.power(f[1], 2))
        mu2 = f[0] * f[1] - f[2] + cmath.sqrt(d)
        mu2 /= 2.0 * (np.power(f[1], 2) - f[0] * f[2])
        mu1 = (mu2 * f[0] - 1.0) / (mu2 * f[1] - f[0])
        y1 = (mu1 * f[0] - 1.0) * mu2 / mu1

        return y1, mu1, mu2

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev):
        """
        Подбор параметров C2 распределения по среднему и коэффициенту вариации
        Возвращает список с параметрами [y1, mu1, mu2]
        """

        f = [0, 0, 0]
        alpha = 1 / (coev ** 2)
        f[0] = f1
        f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)

        return Cox_dist.get_params(f)


class Det_dist:
    """Детерминированное"""

    def __init__(self, b):
        """
        Принимает список параметров в следующей последовательности - alpha, K
        """
        self.b = b
        self.type = 'D'

    def generate(self):
        return Det_dist.generate_static(self.b)

    @staticmethod
    def generate_static(b):
        return b


class Pareto_dist:
    "Распределение Парето"

    def __init__(self, params, generator=None):
        """
        Принимает список параметров в следующей последовательности - alpha, K
        """
        self.a = params[0]
        self.k = params[1]
        self.params = params
        self.type = 'Pa'
        self.generator = generator

    def generate(self):
        return Pareto_dist.generate_static(self.a, self.k, self.generator)

    @staticmethod
    def get_pdf(t, a, k):
        """
        Возвращает значение функции плотности распределения вероятностей СВ
        """
        if t < 0:
            return 0
        return a * math.pow(k, a) / math.pow(t, a + 1)

    @staticmethod
    def get_cdf(params, t):
        """
        Возвращает значение функции распределения СВ
        params = [a, K]
        """
        return 1.0 - Pareto_dist.get_tail(params, t)

    @staticmethod
    def get_tail(params, t):
        """
        Возвращает значение ДФР
        params = [a, K]
        """
        if t < 0:
            return 0
        a = params[0]
        k = params[1]
        return math.pow(k / t, a)

    @staticmethod
    def calc_theory_moments(a, k, max_number=3):
        """
        Вычисление теоретических начальных моментов распределения по заданным параметрам [a, K]
        """
        f = []
        for i in range(max_number):
            if a > i + 1:
                f.append(a * math.pow(k, i + 1) / (a - i - 1))
            else:
                return f
        return f

    @staticmethod
    def generate_static(a, k, generator=None):
        if generator:
            return k * math.pow(generator.random(), -1 / a)
        return k * math.pow(np.random.rand(), -1 / a)

    @staticmethod
    def get_params(f):
        """
        Метод возвращает параметры a и K по 2-м начальным моментам списка f
        Добавлен для совместимости
        """
        return Pareto_dist.get_a_k(f)

    @staticmethod
    def get_a_k(f):
        """
        Метод возвращает параметры a и K по 2-м начальным моментам списка f
        """
        d = f[1] - f[0] * f[0]
        c = f[0] * f[0] / d
        disc = 4 * (1 + c)
        a = (2 + math.sqrt(disc)) / 2
        k = (a - 1) * f[0] / a
        return a, k

    @staticmethod
    def get_a_k_by_mean_and_coev(mean, coev):
        """
        Метод возвращает параметры a и K по среднему и коэффициенту вариации
        """
        d = pow(mean * coev, 2)
        c = pow(mean, 2) / d
        disc = 4 * (1 + c)
        a = (2 + math.sqrt(disc)) / 2
        k = (a - 1) * mean / a
        return a, k


class Erlang_dist:
    """
    Распределение Эрланга r-го порядка
    """

    def __init__(self, params, generator=None):
        """"
        Принимает список параметров в следующей последовательности - r, mu
        """
        self.r = params[0]
        self.mu = params[1]
        self.params = params
        self.type = 'E'
        self.generator = generator

    def generate(self):
        """
        Генератор псевдо-случайных чисел
        """
        return self.generate_static(self.r, self.mu, self.generator)

    @staticmethod
    def generate_static(r, mu, generator=None):
        """
        Генератор псевдо-случайных чисел. Статический метод
        """
        prod = 1
        for i in range(r):
            if generator:
                prod *= generator.random()
            else:
                prod *= np.random.rand()

        return -(1.0 / mu) * np.log(prod)

    @staticmethod
    def get_cdf(params, t):
        """
        Возвращает значение функции распределения СВ
        """
        if t < 0:
            return 0
        r = params[0]
        mu = params[1]
        res = 0
        for i in range(r):
            res += math.pow(mu * t, i) * math.exp(-mu * t) / math.factorial(i)
        return 1.0 - res

    @staticmethod
    def get_tail(params, t):
        """
        Возвращает значение ДФР
        """
        return 1.0 - Erlang_dist.get_cdf(params, t)

    @staticmethod
    def calc_theory_moments(r, mu, count=3):
        """
        Вычисляет теоретические начальные моменты распределения. По умолчанию - первые три
        r, mu - параметры распределения Эрланга
        """
        f = [0.0] * count
        for i in range(count):
            prod = 1
            for k in range(i + 1):
                prod *= r + k
            f[i] = prod / math.pow(mu, i + 1)
        return f

    @staticmethod
    def get_params(f):
        """
        Метод вычисляет параметры распределения Эрланга по двум начальным моментам
        Возвращает кортеж (r, mu) - параметры распределения Эрланга
        """
        r = int(math.floor(f[0] * f[0] / (f[1] - f[0] * f[0]) + 0.5))
        mu = r / f[0]
        return r, mu

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev):
        """
        Метод подбирает параметры распределения Эрланга по среднему и коэффициенту вариации
        Возвращает кортеж (r, mu) - параметры распределения Эрланга
        """
        f = [0, 0]
        f[0] = f1
        f[1] = (math.pow(coev, 2) + 1) * math.pow(f[0], 2)
        return Erlang_dist.get_params(f)


class Exp_dist:
    """
        Экспоненциальное распределение
    """

    def __init__(self, mu, generator=None):
        self.erl = Erlang_dist([1, mu])
        self.params = mu
        self.type = 'M'
        self.generator = generator

    def generate(self):
        return self.erl.generate()

    @staticmethod
    def generate_static(mu, generator=None):
        return Erlang_dist.generate_static(1, mu, generator)

    @staticmethod
    def calc_theory_moments(mu, count=3):
        return Erlang_dist.calc_theory_moments(r=1, mu=mu, count=count)

    @staticmethod
    def get_params(moments):
        return Erlang_dist.get_params(moments)[1]


class Gamma:
    """
    Гамма-распределение
    """

    def __init__(self, params, generator=None):
        self.mu = params[0]
        self.alpha = params[1]
        self.is_corrective = False
        self.g = []
        if len(params) > 2:
            self.is_corrective = True
            for i in range(2, len(params)):
                self.g.append(params[i])
        self.params = params
        self.type = 'Gamma'
        self.generator = generator

    @staticmethod
    def get_mu_alpha(b):
        """
        Статический метод аппроксимации параметров mu и alpha Гамма-распределения
        по двум заданным начальным моментам в списке "b"
        b: список из двух начальных моментов
        Возвращает кортеж из параметров mu и alpha
        """
        d = b[1] - b[0] * b[0]
        mu = b[0] / d
        alpha = mu * b[0]
        return mu, alpha

    @staticmethod
    def get_params(b):
        """
        Статический метод аппроксимации параметров Гамма-распределения поправочным многочленом
        b: список из произвольного числа начальных моментов
        Возвращает список параметров mu и alpha и, если число начальных моментов больше двух, значения g[j], j=0,N, где N - число моментов
        """
        d = b[1] - b[0] * b[0]
        mu = b[0] / d
        alpha = mu * b[0]
        if len(b) > 2:
            # подбор коэффициентов g
            A = []
            B = []
            for i in range(len(b) + 1):
                A.append([])
                if i == 0:
                    B.append(1)
                else:
                    B.append(b[i - 1])
                for j in range(len(b) + 1):
                    A[i].append(Gamma.get_gamma(alpha + i + j) / (pow(mu, i + j) * Gamma.get_gamma(alpha)))
            g = np.linalg.solve(A, B)
            return mu, alpha, g
        else:
            return mu, alpha

    @staticmethod
    def get_mu_alpha_by_mean_and_coev(mean, coev):
        """
        Возвращает список параметров mu и alpha по заданным среднему и коэфф. вариации
        """
        d = pow(mean * coev, 2)
        mu = mean / d
        alpha = mu * mean
        return mu, alpha

    def generate(self):
        return self.generate_static(self.mu, self.alpha, self.generator)

    @staticmethod
    def generate_static(mu, alpha, generator=None):
        theta = 1 / mu
        if generator:
            return generator.gamma(alpha, theta)
        return np.random.gamma(alpha, theta)

    @staticmethod
    def get_cdf(mu, alpha, t):
        """
        Возвращает значение функции распределения СВ
        """
        return stats.gamma.cdf(mu * t, alpha)

    @staticmethod
    def get_pdf(mu, alpha, t):
        """
        Функция плотности вероятности Гамма-распределения
        :param mu: параметр Гамма-распределения
        :param alpha: параметр Гамма-распределения
        :param t: время
        :return: значение плотности Гамма-распределения
        Добавлен для совместимости
        """
        return Gamma.get_f(mu, alpha, t)

    @staticmethod
    def get_f(mu, alpha, t):
        """
        Функция плотности вероятности Гамма-распределения
        :param mu: параметр Гамма-распределения
        :param alpha: параметр Гамма-распределения
        :param t: время
        :return: значение плотности Гамма-распределения
        """
        fract = sp.gamma(alpha)
        if math.fabs(fract) > 1e-12:
            if math.fabs(mu * t) > 1e-12:
                main = mu * math.pow(mu * t, alpha - 1) * math.exp(-mu * t) / fract
            else:
                main = 0
        else:
            main = 0
        return main

    @staticmethod
    def get_f_corrective(mu, alpha, g, t):
        """
        Функция плотности вероятности Гамма-распределения с поправочным многочлненом
        mu: параметр Гамма-распределения
        alpha: параметр Гамма-распределения
        g - массив поправочных коэффициентов
        t: время
        """
        fract = sp.gamma(alpha)
        if math.fabs(fract) > 1e-12:
            if math.fabs(mu * t) > 1e-12:
                main = mu * math.pow(mu * t, alpha - 1) * math.exp(-mu * t) / fract
            else:
                main = 0
        else:
            main = 0
        summ = 0
        for i in range(len(g)):
            summ += g[i] * pow(t, i)

        return main * summ

    @staticmethod
    def calc_theory_moments(mu, alpha, count=3):
        """
        mu, alpha - параметры распределения
        Вычисляет теоретические начальные моменты распределения. По умолчанию - первые три
        """
        f = [0.0] * count
        for i in range(count):
            prod = 1
            for k in range(i + 1):
                prod *= alpha + k
            f[i] = prod / math.pow(mu, i + 1)
        return f

    @staticmethod
    def get_pls(mu, alpha, s):
        return math.pow(mu / (mu + s), alpha)

    @staticmethod
    def get_gamma_incomplete(x, z, e=1e-12):

        return Gamma.get_gamma(x) - Gamma.get_gamma_small(x, z, e)

    @staticmethod
    def get_gamma_small(x, z, e=1e-12):
        summ = 0
        n = 0
        while True:
            elem = pow(-z, n) / (math.factorial(n) * (x + n))
            summ += elem
            if math.fabs(elem) < e:
                break
            n += 1
        gamma = summ * pow(z, x)
        return gamma

    @staticmethod
    def get_minus_gamma(x):
        gamma = sp.gamma(x)
        fraction = -math.pi / (x * math.sin(math.pi * x))
        return fraction / gamma

    @staticmethod
    def get_gamma(x):
        """
        Гамма-функиция Г(x)
        """
        if (x > 0.0) & (x < 1.0):
            return Gamma.get_gamma(x + 1.0) / x
        if x > 2:
            return (x - 1) * Gamma.get_gamma(x - 1)
        if x <= 0:
            return math.pi / (math.sin(math.pi * x) * Gamma.get_gamma(1 - x))
        return Gamma.gamma_approx(x)

    @staticmethod
    def gamma_approx(x):
        """
        Возвращает значение Гамма-функции при 1<=x<=2
        """
        p = [-1.71618513886549492533811e+0,
             2.47656508055759199108314e+1,
             -3.79804256470945635097577e+2,
             6.29331155312818442661052e+2,
             8.66966202790413211295064e+2,
             -3.14512729688483657254357e+4,
             -3.61444134186911729807069e+4,
             6.6456143820240544627855e+4]
        q = [-3.08402300119738975254354e+1,
             3.15350626979604161529144e+2,
             -1.01515636749021914166146e+3,
             -3.10777167157231109440444e+3,
             2.253811842098015100330112e+4,
             4.75584667752788110767815e+3,
             -1.34659959864969306392456e+5,
             -1.15132259675553483497211e+5]
        z = x - 1.0
        a = 0.0
        b = 1.0
        for i in range(0, 8):
            a = (a + p[i]) * z
            b = b * z + q[i]
        return a / b + 1.0


if __name__ == "__main__":

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('TkAgg')

    r = 3
    mu = 0.7
    erl_gen = [Erlang_dist.generate_static(r, mu) for x in range(1000)]
    print(sum(erl_gen) / len(erl_gen), Erlang_dist.calc_theory_moments(r, mu)[0])

    print(Gamma.get_minus_gamma(0.5), Gamma.get_gamma(-0.5))
    b1 = 1
    coev = 1.3
    b = [0.0] * 4
    alpha = 1 / (coev ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
    b[3] = b[2] * b[0] * (1.0 + 6 / alpha)

    mu, a, g = Gamma.get_params(b)
    print(mu, a, g)
    sko = math.sqrt(b[1] - pow(b[0], 2))
    x = np.linspace(0.05, sko, 100)
    y1 = []
    y2 = []
    y_h2 = []
    params_H2 = H2_dist.get_params(b)
    for i in range(len(x)):
        y1.append(Gamma.get_f_corrective(mu, a, g, x[i]))
        y2.append(Gamma.get_f(mu, a, x[i]))
        y_h2.append(H2_dist.get_pdf(params_H2, x[i]))

    fig, ax = plt.subplots()

    ax.plot(x, y1, label="Плотность Гамма с поправочным многочленом")
    ax.plot(x, y2, label="Плотность Гамма без поправочного многочлена")
    ax.plot(x, y_h2, label="Плотность H2")
    plt.legend()
    plt.show()

    b_coev = [0.3, 0.6, 0.8, 1.2, 2.5]
    print("\n")

    for j in range(len(b_coev)):
        b1 = 1
        b = [0.0] * 3
        alpha = 1 / (b_coev[j] ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev[j], 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        h2_params_clx = H2_dist.get_params_clx(b)
        h2_params = H2_dist.get_params(b)
        print("Коэффициент вариации: {0:3.3f}. Параметр alpha {1:3.3f}".format(b_coev[j], alpha))
        print("-" * 45)
        print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ нач момента", "Компл", "Действ"))
        print("-" * 45)
        for i in range(3):
            print("{0:^15d} |{1:^15.3f}|{2:^15.3f}".format(i + 1, h2_params_clx[i], h2_params[i]))
        print("-" * 45)
        print("\n")
