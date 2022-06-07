import numpy as np
import sys

def set_lambdas_prob(l, sigmas, n, file_name='lambdas_set'):
    stout_temp = sys.stdout
    sys.stdout = open(file_name+'.txt', 'w')
    lambdas = []
    k_num = len(l)
    lambdas.append(l)
    for k in range(k_num):
        if k != k_num - 1:
            print(l[k], end=',')
        else:
            print(l[k])

    for i in range(1, n):
        lambdas.append([0.0] * k_num)
        for k in range(k_num):
            delta = np.random.normal(0, sigmas[k])
            r = np.random.rand()
            if r < 0.5:
                delta = -delta
            if lambdas[i - 1][k] + delta < 0:
                lambdas[i][k] = lambdas[i - 1][k]
            else:
                lambdas[i][k] = lambdas[i - 1][k] + delta
            if k!= k_num-1:
                print(lambdas[i][k], end=',')
            else:
                print(lambdas[i][k])

    sys.stdout = stout_temp
    sys.stdout.close()
    return lambdas


def set_lambdas_trend_with_noize(l_start, l_end, sigmas, n, file_name='lambdas_trend_set'):
    stout_temp = sys.stdout
    sys.stdout = open(file_name+'.txt', 'w')
    lambdas = []
    k_num = len(l_start)
    lambdas.append(l_start)
    for k in range(k_num):
        if k != k_num - 1:
            print(l_start[k], end=',')
        else:
            print(l_start[k])

    delta_l = []
    steps = []
    for k in range(k_num):
        delta_l.append(l_end[k] - l_start[k])
        steps.append(delta_l[k]/n)

    for i in range(1, n):
        lambdas.append([0.0] * k_num)
        for k in range(k_num):
            delta_noize = np.random.normal(0, sigmas[k])
            r = np.random.rand()
            if r < 0.5:
                delta_noize = -delta_noize

            next_l = l_start[k] + i*steps[k]
            if next_l + delta_noize < 0:
                lambdas[i][k] = next_l
            else:
                lambdas[i][k] = next_l + delta_noize

            if k != k_num-1:
                print(lambdas[i][k], end=',')
            else:
                print(lambdas[i][k])
    sys.stdout = stout_temp
    sys.stdout.close()
    return lambdas

def load_lambdas_from_file(file_name='lambdas_set.txt'):
    lambdas = []
    f = open(file_name, 'r')
    for l in f:
        s = l.rstrip().split(',')
        a = []
        for j in s:
            a.append(float(j))
        lambdas.append(a)
    return lambdas


if __name__ == "__main__":
    L_first = [7, 3, 4, 1, 21]
    L_last = [7, 3, 4, 1, 21]
    for i in range(len(L_first)):
        L_first[i] /= (24 * 1.8)
        L_last[i] /= (24 * 1.3)
    sigmas = [0.02, 0.02, 0.02, 0.02, 0.02]
    jobs_num = 10000
    delta_jobs = 100
    set_num = int(jobs_num / delta_jobs)
    set_lambdas_trend_with_noize(L_first, L_last, sigmas, set_num, 'trend_real')


