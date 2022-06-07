import csv
import datetime
import math
import rand_destribution as rd
import matplotlib.pyplot as plt

def getIntervalsFromWeblog(file_obj):
    """
    Получает интервалы из файла, содержащего время поступления запросов на сервер (
    поле 'Time')
    """
    reader = csv.DictReader(file_obj, delimiter = ';')
    timestamps = []
    for line in reader:
        dt = line["InvoiceDate"].split(' ')
        date = dt[0].split(".")
        time = dt[1].split(":")
        day = int(date[0])
        month = int(date[1])
        year = int(date[2])
        hour = int(time[0])
        min = int(time[1])
        if len(time) == 3:
            sec = int(time[2])
        else:
            sec = 0

        timestamps.append(datetime.datetime(year, month, day, hour, min, sec))

    deltas_sec = []
    for i in range(len(timestamps)-1):
        delta = timestamps[i+1]-timestamps[i]
        delta_sec = delta.seconds
        if delta_sec != 0:
            deltas_sec.append(delta.seconds)

    return deltas_sec


fileobj = open('Online Retail.csv')
deltas = getIntervalsFromWeblog(fileobj)


f = [0, 0, 0]
N = len(deltas)

for d in deltas:
    for j in range(3):
        f[j] += math.pow(d/60, j+1)

for j in range(3):
    f[j] /= N

variance = f[1]-f[0]**2
coev = math.sqrt(variance)/f[0]

print("Статистические начальные моменты:")
print("{0:<15.3f}{1:<15.3f}{2:<15.3f}\n".format(*f))
print("Коэффициент вариации:")
print("{0:<15.3f}\n".format(coev))

print("Аппроксимация распределением Парето:")

params = rd.Pareto_dist.get_a_k(f)
a = params[0]
k = params[1]

f_teor = rd.Pareto_dist.calc_theory_moments(a, k)
print("Параметры распределения Парето:")
print("a = {0:<15.3f}     K = {1:<15.3f}\n".format(a, k))

print("Теоретические начальные моменты:")
for mom in f_teor:
    print("{0:<15.3f}".format(mom), end="   ")

pareto_data = []
for i in range(len(deltas)):
    pareto_data.append(rd.Pareto_dist.generate_static(a, k))

plt.hist([deltas, pareto_data], label=['Stat', 'Pareto'], density=True)
plt.legend()
plt.show()


