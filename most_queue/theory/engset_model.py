from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.general_utils.conv import get_moments


class Engset:
    """
    Расчет СМО М\М\1 с конечным числом источников m
    """

    def __init__(self, lam, mu, m):
        """
        lam - интенсивность поступления заявок от каждого источника
        mu - интенсивность обслуживания
        m - число источников заявок
        """
        self.lam = lam
        self.mu = mu
        self.ro = lam / mu
        self.m = m
        self.calc_m_i()

    def calc_m_i(self):
        m_i = []
        m_i.append(1)
        prod = 1
        for i in range(self.m):
            prod *= (self.m - i)
            m_i.append(prod)
        self.m_i = m_i

    def get_p(self):
        """
        Вероятности состояний системы
        """
        summ = 0
        for i, mm in enumerate(self.m_i):
            summ += mm * pow(self.ro, i)

        ps = []
        ps.append(1.0 / summ)

        for i in range(1, self.m + 1):
            ps.append(ps[0] * self.m_i[i] * pow(self.ro, i))

        return ps

    def get_N(self):
        """
        Средние число заявок в системе
        """
        p0 = self.get_p()[0]
        N = 0
        for i, mm in enumerate(self.m_i):
            N += i * mm * pow(self.ro, i)
        N *= p0

        return N

    def get_Q(self):
        """
        Средние число заявок в очереди
        """
        p0 = self.get_p()[0]
        return self.get_N() - (1.0 - p0)

    def get_kg(self):
        """
        Вероятность того, что произвольно выбранный источник может послать заявку,
        т.е. коэффициент готовности
        """
        p0 = self.get_p()[0]
        return self.mu * (1.0 - p0) / (self.lam * self.m)

    def get_lamD(self):
        lamD = self.lam * (self.m - self.get_N())
        lamD2 = self.mu * (1.0 - self.get_p()[0])
        print(f'Control {lamD:3.3f} == {lamD2:3.3f}')
        return lamD

    def get_w1(self):
        """
        Средние время ожидания без дифф ПЛС
        """
        return self.get_Q() / self.get_lamD()

    def get_v1(self):
        """
        Средние время пребывания без дифф ПЛС
        """
        return self.get_N() / self.get_lamD()

    def get_w(self):
        """
        Начальные моменты времени ожидания через дифф ПЛС
        """
        h = 0.01
        ss = [x * h for x in range(5)]
        p0 = self.get_p()[0]
        N = self.get_N()

        ws_dots = [self.ws(s, p0, N) for s in ss]
        w_diff = diff5dots(ws_dots, h)

        return [-w_diff[0], w_diff[1], -w_diff[2]]

    def ws(self, s, p0, N):
        """
        ПЛС времени ожидания
        """
        summ = 0
        for i in range(self.m):
            summ += pow(self.lam, i) * self.m_i[i + 1] / pow(self.mu + s, i)

        return summ * p0 / (self.m - N)

    def get_v(self):
        """
        Начальные моменты времени пребывания через свертку с обслуживанием и дифф ПЛС времени ожидания
        """
        w = self.get_w()
        b = [1.0 / self.mu, 2.0 / pow(self.mu, 2), 6.0 / pow(self.mu, 3)]
        return get_moments(w, b)


