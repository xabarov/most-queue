import numpy as np
import math, cmath
from scipy import stats
from numba import jit

class H2_dist:
    def __init__(self, params):
        """
        Принимает список параметров в следующей последовательности - y1, mu1, mu2
        """
        self.y1 = params[0]
        self.m1 = params[1]
        self.m2 = params[2]
        self.params = params
        self.type = 'H'

    def generate(self):
        """
        Генерация псевдо-случайных чисел, подчиненных гиперэкспоненциальному распределению 2-го порядка.
        Вызов из экземпляра класса
        """
        return self.generate_static(self.y1, self.m1, self.m2)

    @staticmethod
    @jit()
    def generate_static(y1, m1, m2):
        """
        Генерация псевдо-случайных чисел, подчиненных гиперэкспоненциальному распределению 2-го порядка.
        Статический метод
        """
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
        f = [0.0]*num
        y2 = 1.0 - y1
        for i in range(num):
            f[i] = math.factorial(i+1)*(y1/pow(m1, i+1) + y2/pow(m2, i+1))
        return f

    @staticmethod
    def get_params(moments):
        """
        Метод Алиева для подбора параметров H2 распределения по заданным начальным моментам.
        Подбирает параметры только принадлежащие множеству R (не комплексные)
        """
        v = moments[1] - moments[0]*moments[0]
        v = math.sqrt(v)/moments[0]
        res = [0.0]*3
        if v < 1.0:
            return res

        q_max = 2.0/(1.0 + v*v)
        t_min = 1.5 * ((1 + v*v)**2)*math.pow(moments[0], 3)
        q_min = 0.0
        tn = 0.0

        if t_min > moments[2]:
            # one phase distibution
            q_new = q_max
            mu1 = (1.0 - math.sqrt(q_new*(v*v-1.0)/(2*(1.0-q_new))))*moments[0]
            if math.isclose(mu1, 0):
                mu1 = 1e10
            else:
                mu1 = 1.0/mu1
            res[0] = q_max
            res[1] = mu1
            res[2] = 1e6
            return res
        else:
            max_iteration = 1000
            tec = 0
            t1 = 0
            t2 = 0
            while abs(tn - moments[2]) > 1e-8 and tec < max_iteration:
                tec += 1
                q_new = (q_max+q_min)/2.0
                t1 = (1.0+math.sqrt((1.0-q_new)*(v*v-1.0)/(2*q_new)))*moments[0]
                t2 = (1.0 - math.sqrt(q_new*(v*v-1.0)/(2*(1.0-q_new))))*moments[0]

                tn = 6*(q_new*math.pow(t1, 3)+(1.0-q_new)*math.pow(t2, 3))

                if tn-moments[2] > 0:
                    q_min = q_new
                else:
                    q_max = q_new
            res[0] = q_max
            res[1] = 1.0/t1
            res[2] = 1.0/t2
            return res

    @staticmethod
    def get_params_clx(moments, verbose=True, ee=0.001, is_fitting=True, max_param_fitting_iter=100):
        """
        Метод подбора параметров H2 распределения по заданным начальным моментам.
        Допускает комплексные значения параметров при коэффициенте вариации <1
        ee - точность проверки близости распределения к E1 и E2
        """
        # f2_to_f1 = moments[1]/moments[0]
        # digits = [i+1 for i in range(100)]
        # for i in digits:
        #     if math.fabs(f2_to_f1.real-i) < e:
        #         print("H2 approx warning. Close to Erlang type. Add some small value to higher moments")
        #         for j in range(len(moments)):
        #             moments[j] *= complex(1, (j + 1) * e)
        #         return H2_dist.get_params_clx(moments)
        f = [0.0] * 3
        for i in range(3):
            f[i] = complex(moments[i] / math.factorial(i + 1))
        znam = (f[1] - f[0] ** 2)
        c0 = (f[0] * f[2] - f[1] ** 2) / znam
        c1 = (f[0] * f[1] - f[2]) / znam

        d = pow(c1 / 2, 2) - c0

        if is_fitting:
            # проверка на близость распределения к экспоненциальному
            is_fit = False
            iter_num = 0
            while not is_fit and iter_num<max_param_fitting_iter:

                coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
                e = 0.02*(iter_num+1)
                if math.fabs(coev.real - 1.0) < ee:
                    if verbose:
                        print("H2 is close to Exp. Multiply moments to (1+je), coev = {0:5.3f},"
                              " e = {1:5.3f}. Iter = {2:d}".format(coev, e, iter_num))
                    for i in range(3):
                        moments[i] *= complex(1, (i + 1) * e)
                else:
                    is_fit = True

                iter_num += 1

            is_fit = False
            iter_num = 0
            while not is_fit and iter_num < max_param_fitting_iter:
                coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
                e = (iter_num+1)*0.02

                # проверка на близость распределения к Эрланга 2-го порядка
                if math.fabs(coev.real - 1.0/math.sqrt(2.0)) < ee:
                    if verbose:
                        print("H2 is close to E2. Multiply moments to (1+je), coev = {0:5.3f},"
                              " e = {1:5.3f}. Iter = {2:d}".format(coev, e, iter_num))
                    for i in range(1, 3):
                        moments[i] *= complex(1, (i + 1) * e)
                else:
                    is_fit = True
                iter_num += 1

            # is_fit = False
            # iter_num = 0
            #
            # while not is_fit and iter_num < max_param_fitting_iter:
            #     e = 0.02*(iter_num+1)
            #     f = [0.0] * 3
            #     for i in range(3):
            #         f[i] = complex(moments[i] / math.factorial(i + 1))
            #     znam = (f[1] - f[0] ** 2)
            #     c0 = (f[0] * f[2] - f[1] ** 2) / znam
            #     c1 = (f[0] * f[1] - f[2]) / znam
            #
            #     d = pow(c1 / 2, 2) - c0
            #
            #     coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
            #
            #     if math.fabs(d.real) < 0.01:
            #         if verbose:
            #             print("H2. D is close to 0. Multiply moments to (1+je), coev = {0:5.3f},"
            #                   " e = {1:5.3f}. Iter = {2:d}".format(coev, e, iter_num))
            #         for i in range(1, 3):
            #             moments[i] *= complex(1, (i + 1) * e)
            #     else:
            #         is_fit = True
            #     iter_num += 1

        res = [0, 0, 0]  # y1, mu1, mu2
        # if d < 0:
        c1 = complex(c1)
        x1 = -c1 / 2 + cmath.sqrt(d)
        x2 = -c1 / 2 - cmath.sqrt(d)
        y1 = (f[0] - x2) / (x1 - x2)
        res[0] = y1
        res[1] = 1 / x1
        res[2] = 1 / x2

        # else:
        #     x1 = -c1 / 2 + math.pow(d, 0.5)
        #     x2 = -c1 / 2 - math.pow(d, 0.5)
        #     y1 = (f[0] - x2) / (x1 - x2)
        #
        #     res[0] = y1
        #     if math.isclose(x1, 0):
        #         res[1] = math.inf
        #     else:
        #         res[1] = 1 / x1
        #     if math.isclose(x2, 0):
        #         res[2] = math.inf
        #     else:
        #         res[2] = 1 / x2

        return res

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev):
        f = [0, 0, 0]
        alpha = 1/(coev**2)
        f[0] = f1
        f[1] = math.pow(f[0], 2)*(math.pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)
        return H2_dist.get_params(f)

    @staticmethod
    def get_cdf(params, t):
        if t < 0:
            return 0
        y = [params[0], 1-params[0]]
        mu = [params[1], params[2]]
        res = 0
        for i in range(2):
            res += y[i] * math.exp(-mu[i] * t)
        return 1.0 - res

    @staticmethod
    def get_tail(params, t):
        return 1.0 - H2_dist.get_cdf(params, t)


class Cox_dist:
    """
    Распределение Кокса 2-го порядка
    """
    def __init__(self, params):
        """
        Принимает список параметров в следующей последовательности - y1, mu1, mu2
        """
        self.y1 = params[0]
        self.m1 = params[1]
        self.m2 = params[2]
        self.params = params
        self.type = 'C'

    def generate(self):
        """
        Генерация псевдо-случайных чисел, подчиненных распределению Кокса 2-го порядка. Вызов из экземпляра класса
        """
        return self.generate_static(self.y1, self.m1, self.m2)

    @staticmethod
    def generate_static(y1, m1, m2):
        """
        Генерация псевдо-случайных чисел, подчиненных распределению Кокса 2-го порядка. Статический метод
        """
        r = np.random.rand()
        res = (-1.0/m1)*np.log(np.random.rand())
        if r < y1:
            res = res + (-1.0/m2)*np.log(np.random.rand())
        return res

    @staticmethod
    def calc_theory_moments(y1, m1, m2):
        """
        Метод вычисляет теоретические начальные моменты распределения Кокса 2-го порядка
        """

        y2 = 1.0 - y1
        f = [0.0]*3
        f[0] = y2/m1 + y1*(1.0/m1 + 1.0/m2)
        f[1] = 2.0*(y2/math.pow(m1, 2) + y1*(1.0/math.pow(m1, 2) + 1.0/(m1*m2) + 1.0/math.pow(m2, 2)))
        f[2] = 6.0*(y2/(math.pow(m1, 3)) + y1*(1.0/math.pow(m1, 3) + 1.0/(math.pow(m1, 2)*m2) +
                                               1.0/(math.pow(m2, 2)*m1) + 1.0/math.pow(m2, 3)))

        return f

    @staticmethod
    def get_params(moments):
        """
        Метод вычисляет параметры распределения Кокса 2-го порядка по трем заданным начальным моментам [moments]
        """
        f = [0.0]*3

        # особые случаи:
        if abs(moments[1] - moments[0]*moments[0]) < 1e-3:
            return 0.0, 1.0/moments[0], 0.0

        if abs(moments[1] - (3.0/4)*moments[0]*moments[0]) < 1e-3:
            return 1.0, 2.0 / moments[0], 2.0 / moments[0]

        for i in range(3):
            f[i] = moments[i]/math.factorial(i+1)

        d = pow(f[2]-f[0]*f[1], 2) - 4.0*(f[1]-f[0]*f[0])*(f[0]*f[2]-f[1]*f[1])
        mu2 = f[0]*f[1]-f[2] + cmath.sqrt(d)
        mu2 /= 2.0*(f[1]*f[1]-f[0]*f[2])
        mu1 = (mu2*f[0]-1.0)/(mu2*f[1]-f[0])
        y1 = (mu1*f[0]-1.0)*mu2/mu1

        return y1, mu1, mu2

    @staticmethod
    def get_params_clx(moments, verbose=True, ee=0.02, max_param_fitting_iter=100, is_fitting=False):
        """
        Метод вычисляет параметры распределения Кокса 2-го порядка по трем заданным начальным моментам [moments]
        """

        if is_fitting:
            is_fit = False
            iter = 0
            while not is_fit and iter < max_param_fitting_iter:
                f = [0.0] * 3

                for i in range(3):
                    f[i] = moments[i] / math.factorial(i + 1)

                d = pow(f[2] - f[0] * f[1], 2) - 4.0 * (f[1] - f[0] * f[0]) * (f[0] * f[2] - f[1] * f[1])
                coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
                e = 0.02 * (iter + 1)
                # проверка на близость распределения к экспоненциальному
                if math.fabs(d.real) < ee:
                    if verbose:
                        print("Cox d is close to 0. Multiply all moments to (1+je), coev = {0:5.3f},"
                              " e = {1:5.3f}. Iter = {2:d}".format(coev, e, iter))
                    for i in range(1, 3):
                        moments[i] *= complex(1, (i + 1) * e)
                else:
                    is_fit = True

                iter += 1

        else:
            f = [0.0] * 3

            for i in range(3):
                f[i] = moments[i] / math.factorial(i + 1)

            d = pow(f[2] - f[0] * f[1], 2) - 4.0 * (f[1] - f[0] * f[0]) * (f[0] * f[2] - f[1] * f[1])

        mu2 = f[0] * f[1] - f[2] + cmath.sqrt(d)
        mu2 /= 2.0 * (f[1] * f[1] - f[0] * f[2])
        mu1 = (mu2 * f[0] - 1.0) / (mu2 * f[1] - f[0])
        y1 = (mu1 * f[0] - 1.0) * mu2 / mu1

        return y1, mu1, mu2


class Pareto_dist:
    "Распределение Парето"
    def __init__(self, params):
        """
        Принимает список параметров в следующей последовательности - alpha, K
        """
        self.a = params[0]
        self.k = params[1]
        self.params = params
        self.type = 'Pa'

    def generate(self):
        return Pareto_dist.generate_static(self.a, self.k)

    @staticmethod
    def get_pdf(t, a, k):
        """
        Probability density function
        """
        if t < 0:
            return 0
        return a*math.pow(k, a)/math.pow(t, a+1)

    @staticmethod
    def get_cdf(params, t):
        """
        Cumulative distribution function
        """
        return 1.0 - Pareto_dist.get_tail(params, t)

    @staticmethod
    def get_tail(params, t):
        """
        Complementary cumulative distribution function (tail distribution)
        """
        if t < 0:
            return 0
        a = params[0]
        k = params[1]
        return math.pow(k/t, a)

    @staticmethod
    def calc_theory_moments(a, k, max_number=3):
        f = []
        for i in range(max_number):
            if a > i+1:
                f.append(a*math.pow(k, i+1)/(a-i-1))
            else:
                return f
        return f

    @staticmethod
    def generate_static(a, k):
        return k*math.pow(np.random.rand(), -1/a)

    @staticmethod
    def get_a_k(f):
        """
        Метод возвращает параметры a и K по 2-м начальным моментам списка f
        """
        d = f[1] - f[0]*f[0]
        c = f[0]*f[0]/d
        disc = 4*(1+c)
        a = (2+math.sqrt(disc))/2
        k = (a-1)*f[0]/a
        return a, k


class Erlang_dist:
    """
    Распределение Эрланга r-го порядка
    """
    def __init__(self, params):
        """"
        Принимает список параметров в следующей последовательности - r, mu
        """
        self.r = params[0]
        self.mu = params[1]
        self.params = params
        self.type = 'E'

    def generate(self):
        """
        Генератор псевдо-случайных чисел
        """
        return self.generate_static(self.r, self.mu)

    @staticmethod
    @jit()
    def generate_static(r, mu):
        """
        Генератор псевдо-случайных чисел. Статический метод
        """
        res = 0
        for i in range(r):
            res += -(1.0 / mu) * np.log(np.random.rand())
        return res

    @staticmethod
    def get_cdf(params, t):
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
        return 1.0 - Erlang_dist.get_cdf(params, t)

    @staticmethod
    def calc_theory_moments(r, mu, count=3):
        """
        Вычисляет теоретические начальные моменты распределения. По умолчанию - первые три
        """
        f = [0.0]*count
        for i in range (count):
            prod = 1
            for k in range(i+1):
                prod *= r+k
            f[i] = prod/math.pow(mu, i+1)
        return f

    @staticmethod
    def get_params(f):
        """
        Метод вычисляет параметры распределения Эрланга по двум начальным моментам
        """
        r = int(math.floor(f[0]*f[0]/(f[1]-f[0]*f[0])+0.5))
        mu = r/f[0]
        return r, mu

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev):
        """
        Метод подбирает параметры распределения Эрланга по среднему и коэффициенту вариации
        """
        f = [0, 0]
        f[0] = f1
        f[1] = (math.pow(coev, 2)+1)*math.pow(f[0], 2)
        return Erlang_dist.get_params(f)


class Exp_dist:
    """
        Экспоненциальное распределение
    """
    def __init__(self, mu):
        self.erl = Erlang_dist([1, mu])
        self.params = mu
        self.type = 'M'

    def generate(self):
        return self.erl.generate()

    @staticmethod
    @jit()
    def generate_static(mu):
        return Erlang_dist.generate_static(1, mu)

    @staticmethod
    def calc_theory_moments(mu, count=3):
        return Erlang_dist.calc_theory_moments(r=1, mu=mu, count=count)

    @staticmethod
    def get_params(moments):
        return Erlang_dist.get_params(moments)[1]


class Gamma:
    """
    содержит статические методы для Гамма-распределения
    get_mu_alpha - аппроксимация параметров mu и alpha по начальным моментам
    """

    def __init__(self, params):
        self.mu = params[0]
        self.alpha = params[1]
        self.params = params
        self.type = 'Gamma'

    @staticmethod
    def get_mu_alpha(b):
        """
        Статический метод аппроксимации параметров mu и alpha Гамма-распределения
        по двум заданным начальным моментам в списке "b"
        :param b: список из двух начальных моментов
        :return: кортеж из параметров mu и alpha
        """
        d = b[1] - b[0]*b[0]
        mu = b[0]/d
        alpha = mu*b[0]
        return mu, alpha

    def generate(self):
        return self.generate_static(self.mu, self.alpha)

    @staticmethod
    @jit()
    def generate_static(mu, alpha):
        theta = 1/mu
        return np.random.gamma(alpha, theta)

    @staticmethod
    def get_cdf(mu, alpha, t):
        return stats.gamma.cdf(mu*t, alpha)

    @staticmethod
    def get_f(mu, alpha, t):
        """
        Функция плотности вероятности Гамма-распределения
        :param mu: параметр Гамма-распределения
        :param alpha: параметр Гамма-распределения
        :param t: время
        :return: значение плотности Гамма-распределения
        """
        return mu*math.pow(mu*t,alpha-1)*math.exp(-mu*t)/Gamma.get_gamma(alpha)

    @staticmethod
    def calc_theory_moments(mu, alpha, count=3):
        """
        Вычисляет теоретические начальные моменты распределения. По умолчанию - первые три
        """
        f = [0.0]*count
        for i in range(count):
            prod = 1
            for k in range(i+1):
                prod *= alpha+k
            f[i] = prod/math.pow(mu, i+1)
        return f

    @staticmethod
    def get_pls(mu, alpha, s):
        return math.pow(mu/(mu+s), alpha)

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
        print("-"*45)
        print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ нач момента", "Компл", "Действ"))
        print("-" * 45)
        for i in range(3):
            print("{0:^15d} |{1:^15.3f}|{2:^15.3f}".format(i+1, h2_params_clx[i], h2_params[i]))
        print("-" * 45)
        print("\n")