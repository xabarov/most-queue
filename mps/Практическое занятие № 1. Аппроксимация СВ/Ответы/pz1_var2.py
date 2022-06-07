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
    reader = csv.DictReader(file_obj, delimiter = ',')
    timestamps = []
    for line in reader:
        dt = line["Time"]
        if dt[0] != '[':
            continue
        dt = dt.split('/')
        day = int(dt[0][1:])
        month = switch_month(dt[1])
        yeartime = dt[2].split(":")
        year = int(yeartime[0])
        hour = int(yeartime[1])
        min = int(yeartime[2])
        sec = int(yeartime[3])
        timestamps.append(datetime.datetime(year, month, day, hour, min, sec))

    deltas_sec = []
    for i in range(len(timestamps)-1):
        delta = timestamps[i+1]-timestamps[i]
        deltas_sec.append(delta.seconds)

    return deltas_sec

def switch_month(month_text):
    switcher = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }
    return switcher.get(month_text, "Invalid month")

fileobj = open('weblog.csv')
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
k = params[1]
a = params[0]

f_teor = rd.Pareto_dist.calc_theory_moments(a, k)
print("Параметры распределения Парето:")
print("a = {0:<15.3f}    K = {1:<15.3f}\n".format(a, k))

print("Теоретические начальные моменты:")
for mom in f_teor:
    print("{0:<15.3f}".format(mom), end="  ")

pareto_data = []
for i in range(len(deltas)):
    pareto_data.append(rd.Pareto_dist.generate_static(a, k))

plt.hist([deltas, pareto_data], label=['Stat', 'Pareto'])
plt.legend()
plt.show()

# print("\n\n Интервалы между запросами, сек\n")
# row_length = 10
# i = 0
# for d in deltas:
#
#     if i != row_length:
#         print("{:<5d}".format(d), end='   ')
#         i += 1
#     else:
#         print("{:<5d}".format(d), end='\n')
#         i = 0

# sys.stdout.close()