import mmnr_calc
import smo_im
import numpy as np
import matplotlib.pyplot as plt

print("Вариант 14")

n = 1
r = 500
l = 1.0

jobs_count = 100000

roes = np.linspace(0.1, 0.9, 20)
w1_im = []
w1_teor = []
error = []
for ro in roes:
    smo = smo_im.SmoIm(n, r)
    smo.set_sources(l, 'M')
    mu = l/(ro*n)
    smo.set_servers(mu, 'M')
    smo.run(jobs_count)
    q = mmnr_calc.M_M_n_formula.getQ(l, mu, n, r)
    w1_teor.append(q/l)
    w1_im.append(smo.w[0])
    error.append(100*(q/l - smo.w[0])/(q/l))

print("Среднее время ожидания в СМО от коэффициента загрузки системы")
print("-"*62)
print("{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("ro", "ИМ", "Теор", "Err"))
str_f = "{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}"
print("-"*62)
for i in range(len(roes)):
    print(str_f.format(roes[i], w1_im[i], w1_teor[i], error[i]))
print("-"*62)
fig, ax = plt.subplots()

ax.plot(roes, w1_im, label="ИМ")
ax.plot(roes, w1_teor, label="Числ")

ax.plot(roes, error, label="относ ошибка ИМ")
plt.legend()
plt.show()

print("Вариант 14")

n = 1
r = 50
l = 1.0

jobs_count = [x*10000 for x in range(1, 30)]

ro = 0.8
w1_im = []
w1_teor = []
error = []
for jobs in jobs_count:
    smo = smo_im.SmoIm(n, r)
    smo.set_sources(l, 'M')
    mu = l/(ro*n)
    smo.set_servers(mu, 'M')
    smo.run(jobs)
    q = mmnr_calc.M_M_n_formula.getQ(l, mu, n, r)
    w1_teor.append(q/l)
    w1_im.append(smo.w[0])
    error.append(100*(q/l - smo.w[0])/(q/l))

print("Среднее время ожидания в СМО от числа обс заявок ИМ")
print("-"*62)
print("{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("ro", "ИМ", "Теор", "Err"))
str_f = "{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}"
print("-"*62)
for i in range(len(jobs_count)):
    print(str_f.format(jobs_count[i], w1_im[i], w1_teor[i], error[i]))
print("-"*62)
fig, ax = plt.subplots()

# ax.plot(jobs_count, w1_im, label="ИМ")
# ax.plot(jobs_count, w1_teor, label="Числ")
ax.plot(jobs_count, error, label="относ ошибка ИМ")
plt.legend()
plt.show()






