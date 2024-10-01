from most_queue.rand_distribution import H2_dist, Gamma
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# Входные данные

b1 = 3      # мат. ожидание
coev = 1.2  # коэффициент вариации
a_s = 4     # ассиметрия

# задаем начальные моменты распределения

b = [0, 0, 0]
b[0] = b1
b[1] = pow(b[0], 2)*(1 + pow(coev, 2))

# дисперсия
D = b[1] - pow(b[0], 2)

# СКО
sigma = math.sqrt(D)

# 3 начальный момент
b[2] = a_s*pow(sigma, 3) + 3*b[1]*b[0] - 2*pow(b[0], 2)


# Гамма-аппроксимация
gamma_params = Gamma.get_mu_alpha(b)

# H2-аппроксимация
h2_params = H2_dist.get_params(b)

n = 100000

b_approx_gamma = [0, 0, 0]
b_approx_h2 = [0, 0, 0]

for i in tqdm(range(n)):
    val_gamma = Gamma.generate_static(gamma_params[0], gamma_params[1])
    val_h2 = H2_dist.generate_static(*h2_params)
    for k in range(3):
        b_approx_gamma[k] += pow(val_gamma, k+1)
        b_approx_h2[k] += pow(val_h2, k + 1)

for k in range(3):
    b_approx_gamma[k] /= n
    b_approx_h2[k] /= n

print("{0:^40s}".format("Начальные моменты СВ"))
print("{0:^3s}|{1:^15s}|{2:^15s}|{3:^15s}".format("№", "Реал",  "Gamma", "H2"))
print("-" * 50)
for i in range(3):
    print("{0:^4d}|{1:^15.3g}|{2:^15.3g}|{3:^15.3g}".format(i+1, b[i], b_approx_gamma[i], b_approx_h2[i]))


dots_num = 10000

x = np.linspace(0, 1*sigma, dots_num)
y_gamma = []
y_h2 = []

for i in range(dots_num):
    y_gamma.append(Gamma.get_pdf(gamma_params[0], gamma_params[1], x[i]))
    y_h2.append(H2_dist.get_pdf(h2_params, x[i]))

fig, ax = plt.subplots()
linestyles = ["solid", "dotted", "dashed", "dashdot"]

ax.plot(x, y_gamma, label="Gamma approx", linestyle=linestyles[0])
ax.plot(x, y_h2, label="H2 approx", linestyle=linestyles[1])

# ax.set_ylabel("")
# ax.set_xlabel("c")

plt.legend()
plt.savefig("Gamma vs H2.png", dpi=300)
