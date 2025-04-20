import numpy as np

from most_queue.general_utils.tables import times_print
from most_queue.theory.engset_model import Engset


def test_engset():
    
    lam = 0.3
    mu = 1.0
    m = 7

    engset = Engset(lam, mu, m)

    ps = engset.get_p()

    print(f'Вероятности состояний системы')
    header = "{0:^15s}|{1:^15s}".format('№', 'p')
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for i in range(len(ps)):
        print("{0:^15d}|{1:^15.3f}".format(i, ps[i]))
    print('-' * len(header))
    print("{0:^15s}|{1:^15.3f}".format('Сумм', sum(ps)))
    print('-' * len(header))

    N = engset.get_N()
    Q = engset.get_Q()
    kg = engset.get_kg()

    print(f'N = {N:3.3f}, Q = {Q:3.3f}, kg = {kg:3.3f}')

    w1 = engset.get_w1()
    v1 = engset.get_v1()
    w = engset.get_w()
    v = engset.get_v()

    print(f'v1 = {v1:3.3f}, w1 = {w1:3.3f}')

    print(f'Начальные моменты ожидания и пребывания')
    header = "{0:^15s}|{1:^15s}|{2:^15s}".format('№', 'w', 'v')
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for i in range(3):
        print("{0:^15d}|{1:^15.3f}|{2:^15.3f}".format(i + 1, w[i], v[i]))
    print('-' * len(header))
    
if __name__ == "__main__":
    
    test_engset()