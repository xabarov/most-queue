import csv
import datetime
import math
import rand_destribution as rd
import matplotlib.pyplot as plt

def getIntervalsFromWeblog(file_obj):

    reader = csv.DictReader(file_obj, delimiter = ',')
    deltas_sec = []
    for line in reader:
        status = line["Status"]
        if status!= 'Trip Completed':
            continue
        dt_request = line["Request timestamp"]
        dt = dt_request.split(' ')
        if dt[0].find('/')!=-1:
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

        request_timestamp = datetime.datetime(year, month, day, hour, min, sec)

        dt_drop = line["Drop timestamp"]

        dt = dt_drop.split(' ')
        if dt[0].find('/')!=-1:
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

        drop_timestamp = datetime.datetime(year, month, day, hour, min, sec)

        delta = drop_timestamp-request_timestamp
        deltas_sec.append(delta.seconds)

    return deltas_sec


fileobj = open('Uber Request Data.csv')
deltas = getIntervalsFromWeblog(fileobj)

f = [0, 0, 0]
N = len(deltas)

for i in range(len(deltas)):
    deltas[i] = deltas[i]/60

for d in deltas:
    for j in range(3):
        f[j] += math.pow(d, j+1)

for j in range(3):
    f[j] /= N

variance = f[1]-f[0]**2
coev = math.sqrt(variance)/f[0]

print("Статистические начальные моменты:")
print("{0:<15.3f}{1:<15.3f}{2:<15.3f}\n".format(*f))
print("Коэффициент вариации:")
print("{0:<15.3f}\n".format(coev))

print("Аппроксимация Гамма - распределением:")

params = rd.Gamma.get_mu_alpha(f)
mu = params[0]
alpha = params[1]

f_teor = rd.Gamma.calc_theory_moments(mu, alpha)
print("Параметры Гамма-распределения:")
print("alpha = {0:<15.3f}    mu = {1:<15.3f}\n".format(alpha, mu))

print("Теоретические начальные моменты:")
for mom in f_teor:
    print("{0:<15.3f}".format(mom), end="  ")

gamma_data = []
for i in range(len(deltas)):
    gamma_data.append(rd.Gamma.generate_static(mu, alpha))

plt.hist([deltas, gamma_data], label=['Stat', 'Gamma'], density=True)
plt.legend()
plt.savefig("gist.png")

# print("\n\n Время поездки, мин\n")
# row_length = 10
# i = 0
# for d in deltas:
#
#     if i != row_length:
#         print("{:<5.3f}".format(d/60), end='   ')
#         i += 1
#     else:
#         print("{:<5.3f}".format(d/60), end='\n')
#         i = 0

