# Руководство по численным методам расчета СМО

Это руководство описывает использование модуля численных методов библиотеки Most-Queue для аналитического расчета характеристик систем массового обслуживания.

## Введение

Численные методы позволяют получить точные аналитические результаты для СМО, для которых существуют математические решения. В отличие от симуляции, расчет дает мгновенные результаты без необходимости моделирования большого числа заявок.

## Базовый класс BaseQueue

Все классы расчета наследуются от базового класса `BaseQueue`, который предоставляет единый интерфейс:

```python
from most_queue.theory.base_queue import BaseQueue
```

### Общий API

Все классы расчета следуют единому паттерну:

1. **Создание объекта** — инициализация с параметрами системы
2. **`set_sources()`** — настройка потока поступления
3. **`set_servers()`** — настройка обслуживания
4. **`run()`** — выполнение расчета
5. **Получение результатов** — доступ к характеристикам системы

## Примеры классов расчета

### M/G/1 система

Класс `MG1Calc` для расчета системы M/G/1 (пуассоновский поток, произвольное распределение времени обслуживания).

```python
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution

# Создание калькулятора
mg1 = MG1Calc()

# Настройка потока поступления (пуассоновский)
mg1.set_sources(l=0.5)  # λ = 0.5

# Настройка обслуживания через моменты распределения
# Сначала создаем параметры H2-распределения
h2_params = H2Distribution.get_params_by_mean_and_cv(
    mean=2.0,      # среднее время обслуживания
    cv=0.8         # коэффициент вариации
)

# Вычисляем моменты распределения
b = H2Distribution.calc_theory_moments(h2_params, 5)
# b[0] - среднее, b[1] - второй момент, и т.д.

# Устанавливаем моменты обслуживания
mg1.set_servers(b)

# Выполняем расчет
results = mg1.run()

# Получаем результаты
print(f"Среднее время ожидания: {results.w[0]:.4f}")
print(f"Среднее время пребывания: {results.v[0]:.4f}")
print(f"Коэффициент загрузки: {results.utilization:.4f}")
```

### GI/M/1 система

Класс `GIM1Calc` для расчета системы GI/M/1 (общий поток поступления, экспоненциальное обслуживание).

```python
from most_queue.theory.fifo.gi_m_1 import GIM1Calc
from most_queue.random.distributions import GammaDistribution

# Создание калькулятора
gim1 = GIM1Calc()

# Настройка потока поступления через моменты
# Создаем параметры гамма-распределения
gamma_params = GammaDistribution.get_params_by_mean_and_cv(
    mean=2.0,      # среднее межприходное время
    cv=0.6
)

# Вычисляем моменты межприходных времен
a = GammaDistribution.calc_theory_moments(gamma_params)
gim1.set_sources(a)

# Настройка обслуживания (экспоненциальное)
mu = 0.6  # интенсивность обслуживания
gim1.set_servers(mu)

# Выполняем расчет
results = gim1.run()

print(f"GI/M/1 результаты:")
print(f"  Среднее время ожидания: {results.w[0]:.4f}")
print(f"  Среднее время пребывания: {results.v[0]:.4f}")
```

### M/M/c система

Класс `MMnrCalc` для расчета системы M/M/c (пуассоновский поток, экспоненциальное обслуживание, c каналов).

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

# Создание калькулятора для M/M/3
mm3 = MMnrCalc(n=3)  # 3 канала

# Настройка потока
mm3.set_sources(l=2.0)  # λ = 2.0

# Настройка обслуживания
mm3.set_servers(mu=1.0)  # μ = 1.0

# Выполняем расчет
results = mm3.run(num_of_moments=4)

print(f"M/M/3 результаты:")
print(f"  Среднее время ожидания: {results.w[0]:.4f}")
print(f"  Коэффициент загрузки: {results.utilization:.4f}")
```

### M/M/c/r система с ограниченной очередью

Тот же класс `MMnrCalc` с параметром `r`:

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

# Система M/M/3/20 (3 канала, очередь до 20 заявок)
mm3r = MMnrCalc(n=3, r=20)

mm3r.set_sources(l=2.0)
mm3r.set_servers(mu=1.0)

results = mm3r.run()
print(f"Вероятность потери: {1 - sum(results.p):.4f}")
```

## Работа с моментами распределений

### Что такое моменты?

Моменты распределения — это числовые характеристики случайной величины:
- **Первый момент** (b[0]) — математическое ожидание (среднее значение)
- **Второй момент** (b[1]) — математическое ожидание квадрата
- **Третий момент** (b[2]) — математическое ожидание куба
- И т.д.

### Получение моментов из распределений

Библиотека предоставляет методы для вычисления моментов различных распределений:

```python
from most_queue.random.distributions import (
    H2Distribution, 
    GammaDistribution,
    ErlangDistribution
)

# H2-распределение
h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=0.8)
b_h2 = H2Distribution.calc_theory_moments(h2_params, num=5)

# Гамма-распределение
gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
b_gamma = GammaDistribution.calc_theory_moments(gamma_params, num=5)

# Распределение Эрланга
erlang_params = ErlangDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.5)
b_erlang = ErlangDistribution.calc_theory_moments(erlang_params, num=5)
```

### Пример: расчет M/G/1 с различными распределениями

```python
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution, GammaDistribution

arrival_rate = 0.4
service_mean = 2.5
service_cv = 0.7

# Вариант 1: H2-распределение
h2_params = H2Distribution.get_params_by_mean_and_cv(service_mean, service_cv)
b_h2 = H2Distribution.calc_theory_moments(h2_params, 5)

mg1_h2 = MG1Calc()
mg1_h2.set_sources(l=arrival_rate)
mg1_h2.set_servers(b_h2)
results_h2 = mg1_h2.run()

# Вариант 2: Гамма-распределение
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
b_gamma = GammaDistribution.calc_theory_moments(gamma_params, 5)

mg1_gamma = MG1Calc()
mg1_gamma.set_sources(l=arrival_rate)
mg1_gamma.set_servers(b_gamma)
results_gamma = mg1_gamma.run()

# Сравнение результатов
print(f"H2: среднее время ожидания = {results_h2.w[0]:.4f}")
print(f"Gamma: среднее время ожидания = {results_gamma.w[0]:.4f}")
```

## Структура результатов

### QueueResults

Все классы расчета возвращают объект `QueueResults`:

```python
@dataclass
class QueueResults:
    v: list[float] | None = None      # моменты времени пребывания
    w: list[float] | None = None      # моменты времени ожидания
    p: list[float] | None = None      # вероятности состояний
    pi: list[float] | None = None     # вероятности перед поступлением
    utilization: float | None = None   # коэффициент загрузки
    duration: float = 0.0              # время расчета в секундах
```

### Доступ к результатам

```python
results = calc.run()

# Моменты времени ожидания
w_mean = results.w[0]      # среднее время ожидания
w_second = results.w[1]    # второй момент

# Моменты времени пребывания
v_mean = results.v[0]       # среднее время пребывания

# Вероятности состояний
p0 = results.p[0]          # вероятность простоя
p1 = results.p[1]          # вероятность 1 заявки в системе

# Коэффициент загрузки
ro = results.utilization

# Время расчета
calc_time = results.duration
```

## Сравнение расчета и симуляции

Для проверки корректности можно сравнить результаты расчета и симуляции:

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution, H2Params
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

arrival_rate = 0.5
service_mean = 2.0
service_cv = 0.8

# Параметры H2-распределения
h2_params = H2Distribution.get_params_by_mean_and_cv(service_mean, service_cv)
b = H2Distribution.calc_theory_moments(h2_params, 5)

# Расчет
mg1_calc = MG1Calc()
mg1_calc.set_sources(l=arrival_rate)
mg1_calc.set_servers(b)
calc_results = mg1_calc.run()

# Симуляция
qs = QsSim(num_of_channels=1)
qs.set_sources(arrival_rate, "M")
qs.set_servers(h2_params, "H")
sim_results = qs.run(50000)

# Сравнение
print("Сравнение моментов времени ожидания:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nСравнение моментов времени пребывания:")
print_sojourn_moments(sim_results.v, calc_results.v)
```

## Доступные классы расчета

### FIFO системы

- **`MG1Calc`** — M/G/1 система
- **`GIM1Calc`** — GI/M/1 система
- **`GiMn`** — GI/M/c система
- **`MMnrCalc`** — M/M/c/r система
- **`MDnCalc`** — M/D/c система
- **`EkDnCalc`** — E_k/D/c система
- **`MGnCalc`** — M/G/c система (метод Такахаси-Таками)

### Системы с приоритетами

- **`MG1Preemptive`** — M/G/1 с прерываемым приоритетом
- **`MG1NonPreemptive`** — M/G/1 с непрерываемым приоритетом
- **`MGnInvarApproximation`** — M/G/c с приоритетами (метод инвариантных соотношений)

### Специализированные системы

- **`BatchMM1`** — M^x/M/1 с пакетным поступлением
- **`EngsetCalc`** — закрытая система Engset
- **`ForkJoinMarkovianCalc`** — Fork-Join система M/M/c
- И другие (см. [Модели СМО](models.md))

## Параметры расчета

### CalcParams

Некоторые классы принимают параметры расчета:

```python
from most_queue.theory.calc_params import CalcParams

calc_params = CalcParams(
    p_num=100,              # число вероятностей состояний для расчета
    tolerance=1e-6,         # точность вычислений
    approx_distr="gamma"    # тип распределения для аппроксимации
)

mg1 = MG1Calc(calc_params=calc_params)
```

## Советы по использованию

1. **Проверяйте устойчивость** — убедитесь, что ρ < 1 перед расчетом
2. **Используйте достаточно моментов** — для точности нужны несколько моментов распределения
3. **Сравнивайте с симуляцией** — проверяйте результаты на простых случаях
4. **Обрабатывайте ошибки** — некоторые системы могут не иметь решения
5. **Используйте подходящие распределения** — выбирайте распределения, соответствующие реальным данным

## Производительность

Численные методы обычно работают значительно быстрее симуляции:

```python
import time

# Расчет
start = time.time()
results = calc.run()
calc_time = time.time() - start

# Симуляция
start = time.time()
results = qs.run(50000)
sim_time = time.time() - start

print(f"Расчет: {calc_time:.4f} сек")
print(f"Симуляция: {sim_time:.4f} сек")
print(f"Ускорение: {sim_time/calc_time:.1f}x")
```

---

**См. также:**
- [Симуляция СМО](simulation.md) — имитационное моделирование
- [Распределения](distributions.md) — справочник по распределениям
- [Модели СМО](models.md) — каталог поддерживаемых моделей

