# Системы с приоритетами

Системы с приоритетами позволяют разделить заявки на классы с разными приоритетами обслуживания. Библиотека Most-Queue поддерживает как прерываемый (PR), так и непрерываемый (NP) приоритет.

## Типы приоритетов

### Прерываемый приоритет (PR - Preemptive Resume)

При прерываемом приоритете обслуживание низкоприоритетной заявки может быть прервано при поступлении заявки более высокого приоритета. После прерывания обслуживание низкоприоритетной заявки возобновляется с того места, где было прервано (resume).

**Характеристики:**
- Высокоприоритетные заявки обслуживаются немедленно
- Низкоприоритетные заявки могут быть прерваны
- После прерывания обслуживание возобновляется

### Непрерываемый приоритет (NP - Non-Preemptive)

При непрерываемом приоритете начатое обслуживание не прерывается. Приоритеты учитываются только при выборе следующей заявки из очереди после завершения текущего обслуживания.

**Характеристики:**
- Начатое обслуживание завершается полностью
- Приоритеты влияют только на порядок выбора из очереди
- Более справедливая дисциплина для низкоприоритетных заявок

## Симуляция систем с приоритетами

### Класс PriorityQueueSimulator

Класс `PriorityQueueSimulator` используется для симуляции многоканальных систем с приоритетами.

### Создание симулятора

```python
from most_queue.sim.priority import PriorityQueueSimulator

# Создание симулятора
# num_of_channels - число каналов
# num_of_classes - число классов приоритетов
# prty_type - тип приоритета: "PR" или "NP"
qs = PriorityQueueSimulator(
    num_of_channels=5,
    num_of_classes=3,
    prty_type="PR"  # или "NP"
)
```

### Настройка потоков поступления

Для каждого класса приоритета настраивается отдельный поток поступления:

```python
# Список словарей с параметрами потоков для каждого класса
sources = []

for j in range(num_of_classes):
    sources.append({
        "type": "M",                    # тип распределения
        "params": arrival_rates[j]      # параметры (для M - интенсивность)
    })

qs.set_sources(sources)
```

### Настройка обслуживания

Для каждого класса настраиваются параметры времени обслуживания:

```python
from most_queue.random.distributions import GammaDistribution

servers_params = []

for j in range(num_of_classes):
    # Параметры распределения времени обслуживания для класса j
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=service_means[j],
        cv=service_cv
    )
    servers_params.append({
        "type": "Gamma",
        "params": gamma_params
    })

qs.set_servers(servers_params)
```

### Полный пример симуляции

```python
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_sojourn_multiclass

# Параметры системы
num_channels = 5
num_classes = 3
arrival_rates = [0.1, 0.2, 0.3]
service_means = [2.25, 4.5, 6.75]
service_cv = 0.8

# Создание симулятора с прерываемым приоритетом
qs = PriorityQueueSimulator(num_channels, num_classes, "PR")

# Настройка потоков
sources = []
servers_params = []

for j in range(num_classes):
    # Поток поступления
    sources.append({"type": "M", "params": arrival_rates[j]})
    
    # Параметры обслуживания
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(
        mean=service_means[j],
        cv=service_cv
    )
    servers_params.append({"type": "Gamma", "params": gamma_params})

qs.set_sources(sources)
qs.set_servers(servers_params)

# Запуск симуляции
qs.run(50000)

# Получение результатов
v_sim = qs.v  # моменты времени пребывания для каждого класса
# v_sim[i][j] - j-й момент для класса i
```

## Расчет систем с приоритетами

### M/G/1 с прерываемым приоритетом

Класс `MG1Preemptive` для расчета одноканальной системы:

```python
from most_queue.theory.priority.preemptive.mg1 import MG1Preemptive

calc = MG1Preemptive(num_of_classes=3)

# Интенсивности поступления для каждого класса
calc.set_sources([0.1, 0.2, 0.3])

# Моменты времени обслуживания для каждого класса
# b[i][j] - j-й момент для класса i
b = [
    [2.25, 5.06, 15.19],  # класс 1 (высший приоритет)
    [4.5, 24.3, 145.8],   # класс 2
    [6.75, 54.68, 410.1]  # класс 3 (низший приоритет)
]
calc.set_servers(b)

results = calc.run()

# Результаты для каждого класса
print(f"Класс 1: среднее время пребывания = {results.v[0][0]:.4f}")
print(f"Класс 2: среднее время пребывания = {results.v[1][0]:.4f}")
print(f"Класс 3: среднее время пребывания = {results.v[2][0]:.4f}")
```

### M/G/1 с непрерываемым приоритетом

Класс `MG1NonPreemptive`:

```python
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptive

calc = MG1NonPreemptive(num_of_classes=3)
calc.set_sources([0.1, 0.2, 0.3])
calc.set_servers(b)
results = calc.run()
```

### M/G/c с приоритетами

Класс `MGnInvarApproximation` для многоканальных систем:

```python
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation

# Прерываемый приоритет
calc_pr = MGnInvarApproximation(n=5, priority="PR")
calc_pr.set_sources([0.1, 0.2, 0.3])
calc_pr.set_servers(b)
results_pr = calc_pr.get_v()

# Непрерываемый приоритет
calc_np = MGnInvarApproximation(n=5, priority="NP")
calc_np.set_sources([0.1, 0.2, 0.3])
calc_np.set_servers(b)
results_np = calc_np.get_v()
```

## Сравнение PR и NP приоритетов

### Пример сравнения

```python
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.random.distributions import GammaDistribution
from most_queue.io.tables import print_sojourn_multiclass

num_channels = 5
num_classes = 3
arrival_rates = [0.1, 0.2, 0.3]
service_means = [2.25, 4.5, 6.75]
service_cv = 0.8

# Подготовка параметров
gamma_params = []
for j in range(num_classes):
    gamma_params.append(
        GammaDistribution.get_params_by_mean_and_cv(
            mean=service_means[j],
            cv=service_cv
        )
    )

sources = [{"type": "M", "params": arrival_rates[j]} for j in range(num_classes)]
servers_params = [{"type": "Gamma", "params": gamma_params[j]} for j in range(num_classes)]

# Прерываемый приоритет
qs_pr = PriorityQueueSimulator(num_channels, num_classes, "PR")
qs_pr.set_sources(sources)
qs_pr.set_servers(servers_params)
qs_pr.run(50000)

# Непрерываемый приоритет
qs_np = PriorityQueueSimulator(num_channels, num_classes, "NP")
qs_np.set_sources(sources)
qs_np.set_servers(servers_params)
qs_np.run(50000)

# Сравнение результатов
print("Прерываемый приоритет (PR):")
for i in range(num_classes):
    print(f"  Класс {i+1}: {qs_pr.v[i][0]:.4f}")

print("\nНепрерываемый приоритет (NP):")
for i in range(num_classes):
    print(f"  Класс {i+1}: {qs_np.v[i][0]:.4f}")
```

### Особенности

**Прерываемый приоритет (PR):**
- Высокоприоритетные заявки обслуживаются быстрее
- Низкоприоритетные заявки могут ждать очень долго
- Подходит для критически важных заявок

**Непрерываемый приоритет (NP):**
- Более справедливое распределение времени ожидания
- Низкоприоритетные заявки не "голодают"
- Подходит для систем, где важна справедливость

## Структура результатов

### Многоклассовые результаты

Для систем с приоритетами результаты структурированы по классам:

```python
# Моменты времени пребывания
v = results.v  # v[i][j] - j-й момент для класса i

# Моменты времени ожидания
w = results.w  # w[i][j] - j-й момент для класса i

# Вероятности состояний (обычно для низкоприоритетных заявок)
p = results.p
```

### Пример анализа результатов

```python
results = calc.run()

print("Анализ результатов по классам:")
for i in range(num_classes):
    print(f"\nКласс {i+1} (приоритет {i+1}):")
    print(f"  Среднее время ожидания: {results.w[i][0]:.4f}")
    print(f"  Среднее время пребывания: {results.v[i][0]:.4f}")
    
    # Второй момент для вычисления дисперсии
    if len(results.w[i]) > 1:
        variance = results.w[i][1] - results.w[i][0]**2
        print(f"  Дисперсия времени ожидания: {variance:.4f}")
```

## Практические рекомендации

### Выбор типа приоритета

1. **Используйте PR**, когда:
   - Критически важно быстрое обслуживание высокоприоритетных заявок
   - Низкоприоритетные заявки могут ждать
   - Примеры: системы реального времени, экстренные службы

2. **Используйте NP**, когда:
   - Важна справедливость обслуживания
   - Низкоприоритетные заявки не должны "голодать"
   - Примеры: справедливое распределение ресурсов

### Настройка классов

1. **Определите число классов** — обычно 2-5 классов достаточно
2. **Настройте интенсивности** — учитывайте реальное распределение заявок
3. **Выберите распределения обслуживания** — используйте данные о реальных временах обслуживания
4. **Проверьте загрузку** — убедитесь, что система устойчива для всех классов

### Анализ результатов

1. **Сравните времена ожидания** — проверьте, что приоритеты работают как ожидается
2. **Проверьте справедливость** — для NP приоритета низкоприоритетные заявки не должны ждать слишком долго
3. **Оптимизируйте параметры** — настройте интенсивности и распределения для достижения целей

## Примеры использования

Подробные примеры можно найти в тестах:
- `test_qs_sim_prty.py` — симуляция систем с приоритетами
- `test_mmn_prty_busy_approx.py` — расчет M/M/c с приоритетами
- `test_m_ph_n_prty.py` — системы с фазовым распределением и приоритетами

---

**См. также:**
- [Симуляция СМО](simulation.md) — основы симуляции
- [Численные методы](calculation.md) — аналитические расчеты
- [Сети очередей](networks.md) — сети с приоритетами в узлах

