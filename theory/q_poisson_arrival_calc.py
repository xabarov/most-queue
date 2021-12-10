import math


def get_q_Gamma(l, mu, alpha, num=100):
    """
    Гамма-распределение
    l - интенсивность входного потока
    mu и alpha параметры Гамма-распределения
    num - число возвращаемых q[j]
    """
    q = [0.0] * num
    q[0] = math.pow(mu / (mu + l), alpha)
    for j in range(1, num):
        q[j] = q[j - 1] * l * (alpha + j - 1) / ((l + mu) * j)

    return q

def get_q_uniform(l, mean, half_interval, num=100):
    """
    Равномерное распределение на отрезке [mean-half_interval, mean+half_interval
    l - интенсивность входного потока
    mean - среднее
    half_interval - полуинтервал влево и вправо от среднего значения
    num - число возвращаемых q[j]
    """
    q = [0.0] * num
    for j in range(num):
        summ1 = 0
        for i in range(j+1):
            summ1 += l*pow(mean-half_interval,i)*math.exp(-l*(mean-half_interval))/math.factorial(i)
        summ2 = 0
        for i in range(j + 1):
            summ2 += l * pow(mean + half_interval, i) * math.exp(-l*(mean + half_interval)) / math.factorial(i)
        q[j] = (1.0/(2*l*half_interval))*(summ1 - summ2)

    return q

