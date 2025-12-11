# Расширенные примеры использования

Этот раздел содержит практические примеры использования библиотеки Most-Queue для решения реальных задач.

## Пример 1: Моделирование call-центра

### Задача

Смоделировать call-центр с несколькими операторами и двумя типами звонков: обычные и приоритетные (VIP).

### Решение

```python
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_sojourn_multiclass

# Параметры call-центра
num_operators = 10
num_classes = 2  # обычные и VIP звонки

# Интенсивности поступления
# Обычные звонки: 5 звонков в минуту
# VIP звонки: 1 звонок в минуту
arrival_rates = [5.0 / 60, 1.0 / 60]  # переводим в секунды

# Средние времена обслуживания
# Обычные: 3 минуты
# VIP: 5 минут (более сложные запросы)
service_means = [3.0 * 60, 5.0 * 60]  # в секундах
service_cv = 0.7

# Создание симулятора с непрерываемым приоритетом
# (VIP звонки имеют приоритет, но начатый разговор не прерывается)
qs = PriorityQueueSimulator(num_operators, num_classes, "NP")

# Настройка потоков
sources = []
servers_params = []

for j in range(num_classes):
    sources.append({"type": "M", "params": arrival_rates[j]})
    
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=service_means[j],
        cv=service_cv
    )
    servers_params.append({"type": "Gamma", "params": gamma_params})

qs.set_sources(sources)
qs.set_servers(servers_params)

# Симуляция на 10000 звонков
qs.run(10000)

# Анализ результатов
print("Результаты моделирования call-центра:")
print(f"Обычные звонки:")
print(f"  Среднее время ожидания: {qs.v[0][0] / 60:.2f} минут")
print(f"  Среднее время пребывания: {qs.v[0][0] / 60:.2f} минут")

print(f"\nVIP звонки:")
print(f"  Среднее время ожидания: {qs.v[1][0] / 60:.2f} минут")
print(f"  Среднее время пребывания: {qs.v[1][0] / 60:.2f} минут")

# Проверка загрузки
total_load = sum(arrival_rates[i] * service_means[i] for i in range(num_classes))
utilization = total_load / num_operators
print(f"\nКоэффициент загрузки: {utilization:.2%}")
```

## Пример 2: Анализ облачной инфраструктуры

### Задача

Проанализировать производительность облачного сервера с несколькими виртуальными машинами, обрабатывающими запросы с разными характеристиками.

### Решение

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import H2Distribution
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

# Параметры сервера
num_vms = 8  # число виртуальных машин

# Поток запросов: 100 запросов в секунду
arrival_rate = 100.0

# Время обработки запроса: среднее 50 мс, CV = 1.2
service_mean = 0.05  # секунды
service_cv = 1.2

# Создание параметров H2-распределения для моделирования
# высокого коэффициента вариации
h2_params = H2Distribution.get_params_by_mean_and_cv(
    mean=service_mean,
    cv=service_cv
)

# Симуляция
qs = QsSim(num_of_channels=num_vms)
qs.set_sources(arrival_rate, "M")
qs.set_servers(h2_params, "H")

results = qs.run(100000)

# Анализ результатов
print("Анализ облачного сервера:")
print(f"Число виртуальных машин: {num_vms}")
print(f"Интенсивность запросов: {arrival_rate} зап/сек")
print(f"\nРезультаты:")
print(f"  Среднее время ожидания: {results.w[0] * 1000:.2f} мс")
print(f"  Среднее время обработки: {results.v[0] * 1000:.2f} мс")
print(f"  Коэффициент загрузки: {results.utilization:.2%}")

# Анализ вероятностей состояний
p = results.p
print(f"\nВероятности состояний:")
print(f"  Простой сервера: {p[0]:.2%}")
print(f"  1-4 запроса в обработке: {sum(p[1:5]):.2%}")
print(f"  5-8 запросов в обработке: {sum(p[5:9]):.2%}")
print(f"  Очередь (9+ запросов): {sum(p[9:]):.2%}")

# Рекомендации
if results.utilization > 0.8:
    print("\n⚠️  Внимание: высокая загрузка! Рекомендуется увеличить число ВМ.")
elif results.w[0] > 0.1:
    print("\n⚠️  Внимание: большое время ожидания! Рекомендуется оптимизация.")
```

## Пример 3: Оптимизация транспортной системы

### Задача

Оптимизировать работу станции техобслуживания с несколькими постами и различными типами работ.

### Решение

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_waiting_moments

# Параметры станции
arrival_rate = 10.0 / 60  # 10 машин в час = машин в минуту

# Время обслуживания: среднее 20 минут, CV = 0.6
service_mean = 20.0
service_cv = 0.6

# Тестируем разное число постов
num_posts_options = [2, 3, 4, 5]

print("Анализ различных конфигураций станции:")
print(f"Интенсивность поступления: {arrival_rate * 60:.1f} машин/час")
print(f"Среднее время обслуживания: {service_mean} минут\n")

best_config = None
best_waiting_time = float('inf')

for num_posts in num_posts_options:
    # Расчет интенсивности обслуживания для заданной загрузки
    target_utilization = 0.75
    service_rate = arrival_rate / (num_posts * target_utilization)
    
    # Симуляция
    qs = QsSim(num_of_channels=num_posts)
    qs.set_sources(arrival_rate, "M")
    
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=1.0 / service_rate,
        cv=service_cv
    )
    qs.set_servers(gamma_params, "Gamma")
    
    results = qs.run(50000)
    
    # Анализ
    waiting_time_min = results.w[0]
    utilization = results.utilization
    
    print(f"{num_posts} поста(ов):")
    print(f"  Среднее время ожидания: {waiting_time_min:.2f} минут")
    print(f"  Коэффициент загрузки: {utilization:.2%}")
    
    if waiting_time_min < best_waiting_time:
        best_waiting_time = waiting_time_min
        best_config = num_posts
    
    print()

print(f"Рекомендуемая конфигурация: {best_config} поста(ов)")
print(f"Ожидаемое время ожидания: {best_waiting_time:.2f} минут")
```

## Пример 4: Сравнение симуляции и расчета

### Задача

Проверить корректность симуляции, сравнив результаты с аналитическими расчетами.

### Решение

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution
from most_queue.io.tables import (
    print_waiting_moments,
    print_sojourn_moments,
    probs_print
)

# Параметры системы M/G/1
arrival_rate = 0.4
service_mean = 2.0
service_cv = 0.8

# Создание параметров H2-распределения
h2_params = H2Distribution.get_params_by_mean_and_cv(
    mean=service_mean,
    cv=service_cv
)

# Вычисление моментов для расчета
b = H2Distribution.calc_theory_moments(h2_params, 5)

print("Сравнение симуляции и расчета для M/G/1:")
print(f"Интенсивность поступления: {arrival_rate}")
print(f"Среднее время обслуживания: {service_mean}")
print(f"Коэффициент вариации: {service_cv}\n")

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
print("Моменты времени ожидания:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nМоменты времени пребывания:")
print_sojourn_moments(sim_results.v, calc_results.v)

print("\nВероятности состояний:")
probs_print(sim_results.p, calc_results.p, size=10)

# Проверка точности
w_error = abs(sim_results.w[0] - calc_results.w[0]) / calc_results.w[0] * 100
v_error = abs(sim_results.v[0] - calc_results.v[0]) / calc_results.v[0] * 100

print(f"\nОтносительная ошибка:")
print(f"  Время ожидания: {w_error:.2f}%")
print(f"  Время пребывания: {v_error:.2f}%")
```

## Пример 5: Анализ производительности с различными распределениями

### Задача

Исследовать влияние коэффициента вариации времени обслуживания на характеристики системы.

### Решение

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import (
    H2Distribution,
    GammaDistribution,
    ErlangDistribution
)

arrival_rate = 0.5
service_mean = 2.0
num_channels = 2

# Различные коэффициенты вариации
cvs = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]

print("Влияние коэффициента вариации на характеристики системы:")
print(f"Интенсивность поступления: {arrival_rate}")
print(f"Среднее время обслуживания: {service_mean}")
print(f"Число каналов: {num_channels}\n")

results_table = []

for cv in cvs:
    # Выбор распределения в зависимости от CV
    if cv < 1.0:
        # Используем Эрланга для CV < 1
        params = ErlangDistribution.get_params_by_mean_and_cv(
            mean=service_mean,
            cv=cv
        )
        dist_type = "E"
    elif cv == 1.0:
        # Экспоненциальное для CV = 1
        params = 1.0 / service_mean
        dist_type = "M"
    else:
        # Используем H2 для CV > 1
        params = H2Distribution.get_params_by_mean_and_cv(
            mean=service_mean,
            cv=cv
        )
        dist_type = "H"
    
    # Симуляция
    qs = QsSim(num_of_channels=num_channels)
    qs.set_sources(arrival_rate, "M")
    qs.set_servers(params, dist_type)
    
    results = qs.run(50000)
    
    results_table.append({
        'CV': cv,
        'waiting': results.w[0],
        'sojourn': results.v[0],
        'utilization': results.utilization
    })

# Вывод результатов
print(f"{'CV':<8} {'Ожидание':<12} {'Пребывание':<12} {'Загрузка':<10}")
print("-" * 45)
for r in results_table:
    print(f"{r['CV']:<8.2f} {r['waiting']:<12.4f} {r['sojourn']:<12.4f} {r['utilization']:<10.2%}")

# Выводы
print("\nВыводы:")
print("- С увеличением CV время ожидания увеличивается")
print("- Система становится менее предсказуемой")
print("- Рекомендуется минимизировать вариативность времени обслуживания")
```

## Пример 6: Визуализация результатов

### Задача

Визуализировать вероятности состояний системы для различных конфигураций.

### Решение

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
import matplotlib.pyplot as plt

# Параметры
arrival_rate = 2.0
service_rate = 1.0
num_channels_options = [1, 2, 3, 4]

# Расчет вероятностей для разных конфигураций
probabilities = {}

for n in num_channels_options:
    calc = MMnrCalc(n=n)
    calc.set_sources(l=arrival_rate)
    calc.set_servers(mu=service_rate)
    results = calc.run()
    probabilities[n] = results.p[:15]  # первые 15 состояний

# Визуализация
fig, ax = plt.subplots(figsize=(12, 6))

for n, probs in probabilities.items():
    states = list(range(len(probs)))
    ax.plot(states, probs, marker='o', label=f'M/M/{n}')

ax.set_xlabel('Число заявок в системе')
ax.set_ylabel('Вероятность')
ax.set_title('Распределение вероятностей состояний')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('state_probabilities.png', dpi=150)
print("График сохранен в state_probabilities.png")
```

## Советы по использованию примеров

1. **Адаптируйте параметры** — измените значения под вашу задачу
2. **Проверяйте устойчивость** — убедитесь, что ρ < 1
3. **Используйте достаточно заявок** — для точности нужно 50000+ заявок
4. **Сравнивайте результаты** — используйте расчет для проверки симуляции
5. **Анализируйте вероятности** — они дают полную картину поведения системы

---

**См. также:**
- [Быстрый старт](getting_started.md) — основы использования
- [Симуляция СМО](simulation.md) — детали симуляции
- [Численные методы](calculation.md) — аналитические расчеты

