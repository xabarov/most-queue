import csv
import numpy as np
import math
import rand_destribution as rd
import matplotlib.pyplot as plt
from datetime import datetime


def ks_test(data1, data2, bin=100):
    bin = min(len(data1), bin)
    d1_min = min(data1)
    d1_max = max(data1)
    step = (d1_max - d1_min) / bin
    data1_count = [0] * bin
    data2_count = [0] * bin

    d1_tek = d1_min
    for i in range(bin):
        if i == 0:
            for j in range(len(data1)):
                if data1[j] < d1_tek + step:
                    data1_count[i] += 1
                if data2[j] < d1_tek + step:
                    data2_count[i] += 1
        elif i == bin - 1:
            for j in range(len(data1)):
                if data1[j] >= d1_tek:
                    data1_count[i] += 1
                if data2[j] >= d1_tek:
                    data2_count[i] += 1
        else:
            for j in range(len(data1)):
                if data1[j] >= d1_tek and data1[j] < d1_tek + step:
                    data1_count[i] += 1
                if data2[j] >= d1_tek and data2[j] < d1_tek + step:
                    data2_count[i] += 1
        d1_tek += step

    # calc ps
    for j in range(bin):
        data1_count[j] /= len(data1)
        data2_count[j] /= len(data2)

    cdf1 = [0] * bin
    cdf2 = [0] * bin

    D = 0
    for j in range(bin):
        cdf1[j] = sum(data1_count[:j + 1])
        cdf2[j] = sum(data2_count[:j + 1])
        D = max(D, math.fabs(cdf1[j] - cdf2[j]))

    alpha = 0.95
    Ka = math.sqrt(-0.5 * math.log((1.0 - alpha) / 2))
    D = math.sqrt(bin) * D
    if D > Ka:
        print(
            "Значение параметра D = {0:1.5f} больше порога {1:1.5f}, "
            "нулевая гипотеза о близости распределений отвергнута".format(
                D, Ka))
    else:
        print("Значение параметра D = {0:1.5f} меньше порога {1:1.5f}, "
              "распределения близки ".format(
            D, Ka))


def reader(dataset_num=1):
    if dataset_num == 1:
        wait_times = read_data("Shared_Database.csv")
    elif dataset_num == 2:
        wait_times = read_data2("attendances.csv")
    else:
        wait_times = read_data3("Uber Request Data.csv")
    return wait_times


def read_data(data_path):
    """
    Читает данные, возвращает массив времен ожидания
    """
    wait_times = []
    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i != 0:
                wait_times.append(float(row[-1]))
    return wait_times


def read_data2(data_path, data_count=3000):
    wait_times = []
    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i != 0 and len(wait_times) < data_count:
                timestring = str(row[-1])
                if len(timestring.split(":")) != 3:
                    times = timestring.split(":")
                    minutes = int(times[0])
                    if (minutes >= 60):
                        hours = math.floor(minutes / 60)
                        minutes = minutes - hours * 60
                        timestring = str(hours) + ":" + str(minutes) + ":" + timestring[-1]
                        pt = datetime.strptime(timestring, '%H:%M:%S')
                        total_seconds = pt.second + pt.minute * 60 + + pt.hour * 3600
                    else:
                        pt = datetime.strptime(timestring, '%M:%S')
                        total_seconds = pt.second + pt.minute * 60
                    wait_times.append(total_seconds)
    return wait_times


def read_data3(data_path):
    fileobj = open(data_path)
    reader = csv.DictReader(fileobj, delimiter=',')
    deltas_sec = []
    for line in reader:
        status = line["Status"]
        if status != 'Trip Completed':
            continue
        dt_request = line["Request timestamp"]
        dt = dt_request.split(' ')
        if dt[0].find('/') != -1:
            date = dt[0].split('/')
        else:
            date = dt[0].split('-')

        day = int(date[0])
        month = int(date[1])
        year = int(date[2])
        time = dt[1].split(':')
        hour = int(time[0])
        min = int(time[1])
        if len(time) == 3:
            sec = int(time[2])
        else:
            sec = 0

        request_timestamp = datetime(year, month, day, hour, min, sec)

        dt_drop = line["Drop timestamp"]

        dt = dt_drop.split(' ')
        if dt[0].find('/') != -1:
            date = dt[0].split('/')
        else:
            date = dt[0].split('-')

        day = int(date[0])
        month = int(date[1])
        year = int(date[2])
        time = dt[1].split(':')
        hour = int(time[0])
        min = int(time[1])
        if len(time) == 3:
            sec = int(time[2])
        else:
            sec = 0

        drop_timestamp = datetime(year, month, day, hour, min, sec)

        delta = drop_timestamp - request_timestamp
        deltas_sec.append(delta.seconds)

    return deltas_sec


def get_moments(data, num_of_moments=3):
    """
    Формирует из массива данных статистические начальные моменты
    data - массив времен ожидания
    num_of_moments - требуемое число начальных моментов

    """
    moments = [0.0] * num_of_moments
    #  добавить код ниже (вместо pass):
    pass

    return moments


def get_variance(moments):
    """
    Возвращет значение дисперсии и коэффициента вариации по массиву начальных моментов распределения moments
    """

    #  заменить код ниже:
    variance = 0
    coev = 0

    return variance, coev


if __name__ == "__main__":

    dataset_num = 3
    dist_num = 2

    wait_times = reader(dataset_num)

    moments = get_moments(wait_times)
    variance, coev = get_variance(moments)

    if dist_num == 1:

        head = "Аппроксимация Гамма-распределением:"
        print("-" * len(head))
        print(head)
        print("-" * len(head))

        params = rd.Gamma.get_params(moments)
        mu = params[0]
        alpha = params[1]

        f_teor = rd.Gamma.calc_theory_moments(mu, alpha)
        print("Параметры Гамма-распределения:")
        print("mu = {0:<15.3f}    alpha = {1:<15.3f}\n".format(mu, alpha))
        print("Коэффициент вариации:")
        print("{0:<15.3f}\n".format(coev))

        print("Начальные моменты:")
        head = "{0:^15s}|{1:^15s}|{2:^15s}|{2:^15s}".format("-", "1", "2", "3")
        print("-" * len(head))
        print(head)
        print("-" * len(head))
        print("{0:^15s}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}".format("Стат", *moments))
        print("{0:^15s}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}".format("Теор", *f_teor))

        gamma_data = []
        for i in range(len(wait_times)):
            gamma_data.append(rd.Gamma.generate_static(mu, alpha))

        plt.hist([wait_times, gamma_data], label=['Стат', 'Гамма'])
        plt.legend()
        plt.show()

        ks_test(wait_times, gamma_data)

    elif dist_num == 2:

        head = "Аппроксимация распределением Парето:"
        print("-" * len(head))
        print(head)
        print("-" * len(head))

        params = rd.Pareto_dist.get_params(moments)
        a = params[0]
        k = params[1]

        f_teor = rd.Pareto_dist.calc_theory_moments(a, k)
        f_teor.append(0)
        print("Параметры распределения Парето:")
        print("a = {0:<15.3f}     K = {1:<15.3f}\n".format(a, k))

        print("Коэффициент вариации:")
        print("{0:<15.3f}\n".format(coev))

        print("Начальные моменты:")
        head = "{0:^15s}|{1:^15s}|{2:^15s}|{2:^15s}".format("-", "1", "2", "3")
        print("-" * len(head))
        print(head)
        print("-" * len(head))
        print("{0:^15s}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}".format("Стат", *moments))
        print("{0:^15s}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}".format("Теор", *f_teor))

        pareto_data = []
        for i in range(len(wait_times)):
            pareto_data.append(rd.Pareto_dist.generate_static(a, k))

        plt.hist([wait_times, pareto_data], label=['Stat', 'Pareto'], density=True)
        plt.legend()
        plt.show()

        ks_test(wait_times, pareto_data)

    elif dist_num == 3:

        head = "Аппроксимация нормальным распределением:"
        print("-" * len(head))
        print(head)
        print("-" * len(head))

        print("Коэффициент вариации:")
        print("{0:<15.3f}\n".format(coev))

        normal_data = []
        for i in range(len(wait_times)):
            normal_data.append(rd.Normal_dist.generate_static(moments[0], math.sqrt(variance)))

        plt.hist([wait_times, normal_data], label=['Stat', 'Normal'], density=True)
        plt.legend()
        plt.show()

        # make_test(wait_times, normal_data)
        ks_test(wait_times, normal_data)
