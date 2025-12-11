# Руководство по симуляции СМО

Это руководство описывает использование модуля симуляции библиотеки Most-Queue для моделирования систем массового обслуживания.

## Введение

Симуляция (имитационное моделирование) позволяет моделировать поведение СМО, для которых нет аналитических решений или которые имеют сложную структуру. Библиотека Most-Queue предоставляет класс `QsSim` для симуляции различных типов СМО.

## Базовый класс QsSim

### Создание симулятора

```python
from most_queue.sim.base import QsSim

# Создание симулятора с указанием числа каналов
qs = QsSim(num_of_channels=3)

# С дополнительными параметрами
qs = QsSim(
    num_of_channels=3,      # число каналов обслуживания
    buffer=50,              # максимальная длина очереди (None = бесконечная)
    verbose=True,           # вывод информации о процессе
    buffer_type="list"      # тип буфера: "list" или "deque"
)
```

### Параметры конструктора

- **`num_of_channels`** (int) — число каналов обслуживания (обязательный параметр)
- **`buffer`** (int, optional) — максимальная длина очереди. Если `None`, очередь неограниченная
- **`verbose`** (bool) — вывод подробной информации во время симуляции (по умолчанию `True`)
- **`buffer_type`** (str) — тип реализации очереди: `"list"` или `"deque"`

## Настройка потока поступления

### Метод set_sources()

Метод `set_sources()` настраивает поток поступления заявок в систему.

```python
qs.set_sources(params, kendall_notation="M")
```

**Параметры:**
- **`params`** — параметры распределения межприходных времен
- **`kendall_notation`** (str) — обозначение распределения по Кендаллу

### Примеры настройки потока

#### Экспоненциальное распределение (M)

```python
# Пуассоновский поток с интенсивностью λ = 0.5
qs.set_sources(0.5, "M")
```

#### Гиперэкспоненциальное распределение (H)

```python
from most_queue.random.distributions import H2Distribution, H2Params

# Создание параметров H2-распределения
h2_params = H2Params(p1=0.3, mu1=1.0, mu2=2.0)
qs.set_sources(h2_params, "H")
```

#### Гамма-распределение

```python
from most_queue.random.distributions import GammaDistribution, GammaParams

# Создание параметров по среднему и коэффициенту вариации
gamma_params = GammaDistribution.get_params_by_mean_and_cv(
    mean=2.0,      # среднее межприходное время
    cv=0.5         # коэффициент вариации
)
qs.set_sources(gamma_params, "Gamma")
```

#### Детерминированное распределение (D)

```python
# Постоянный интервал между заявками
qs.set_sources(2.0, "D")  # интервал = 2.0 единицы времени
```

## Настройка обслуживания

### Метод set_servers()

Метод `set_servers()` настраивает распределение времени обслуживания для всех каналов.

```python
qs.set_servers(params, kendall_notation="M")
```

**Параметры:**
- **`params`** — параметры распределения времени обслуживания
- **`kendall_notation`** (str) — обозначение распределения

### Примеры настройки обслуживания

#### Экспоненциальное обслуживание (M)

```python
# Интенсивность обслуживания μ = 1.0
qs.set_servers(1.0, "M")
```

#### Гамма-распределение времени обслуживания

```python
from most_queue.random.distributions import GammaDistribution

# Создание параметров по среднему времени обслуживания и CV
service_mean = 2.5
service_cv = 0.8
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
qs.set_servers(gamma_params, "Gamma")
```

## Запуск симуляции

### Метод run()

После настройки потока и обслуживания запускается симуляция:

```python
results = qs.run(num_of_jobs)
```

**Параметры:**
- **`num_of_jobs`** (int) — количество заявок для обработки

**Возвращает:**
- Объект `QueueResults` с результатами симуляции

### Полный пример

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import GammaDistribution

# Создание симулятора M/G/3
qs = QsSim(num_of_channels=3)

# Настройка пуассоновского потока
qs.set_sources(0.8, "M")  # λ = 0.8

# Настройка гамма-распределения времени обслуживания
service_mean = 3.0
service_cv = 0.6
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
qs.set_servers(gamma_params, "Gamma")

# Запуск симуляции
results = qs.run(50000)

# Получение результатов
print(f"Среднее время ожидания: {results.w[0]:.4f}")
print(f"Среднее время пребывания: {results.v[0]:.4f}")
print(f"Коэффициент загрузки: {results.utilization:.4f}")
```

## Результаты симуляции

### Структура QueueResults

Объект результатов содержит следующие атрибуты:

- **`w`** (list[float]) — начальные моменты времени ожидания в очереди
  - `w[0]` — среднее время ожидания (первый момент)
  - `w[1]` — второй момент
  - `w[2]` — третий момент
  - и т.д.

- **`v`** (list[float]) — начальные моменты времени пребывания в системе
  - `v[0]` — среднее время пребывания
  - `v[1]` — второй момент
  - и т.д.

- **`p`** (list[float]) — вероятности состояний системы
  - `p[0]` — вероятность, что в системе 0 заявок
  - `p[1]` — вероятность, что в системе 1 заявка
  - и т.д.

- **`utilization`** (float) — коэффициент загрузки системы (0 ≤ ρ ≤ 1)

- **`duration`** (float) — время выполнения симуляции в секундах

### Получение результатов напрямую

После выполнения `run()` результаты также доступны через атрибуты объекта `qs`:

```python
qs.run(10000)

# Прямой доступ к результатам
w_sim = qs.w          # моменты времени ожидания
v_sim = qs.v          # моменты времени пребывания
p_sim = qs.get_p()    # вероятности состояний
ro = qs.load          # коэффициент загрузки
```

## Примеры использования

### Пример 1: M/M/1 система

```python
from most_queue.sim.base import QsSim

qs = QsSim(num_of_channels=1)
qs.set_sources(0.5, "M")   # λ = 0.5
qs.set_servers(1.0, "M")   # μ = 1.0

results = qs.run(10000)

print(f"M/M/1 результаты:")
print(f"  Среднее время ожидания: {results.w[0]:.4f}")
print(f"  Среднее время пребывания: {results.v[0]:.4f}")
print(f"  Коэффициент загрузки: {results.utilization:.4f}")
```

### Пример 2: GI/M/1 система

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import GammaDistribution

qs = QsSim(num_of_channels=1)

# Гамма-распределение межприходных времен
arrival_mean = 2.0
arrival_cv = 0.7
gamma_arrival = GammaDistribution.get_params_by_mean_and_cv(arrival_mean, arrival_cv)
qs.set_sources(gamma_arrival, "Gamma")

# Экспоненциальное обслуживание
qs.set_servers(0.6, "M")  # μ = 0.6

results = qs.run(20000)
print(f"GI/M/1 результаты:")
print(f"  Среднее время ожидания: {results.w[0]:.4f}")
```

### Пример 3: M/M/c/r система с ограниченной очередью

```python
from most_queue.sim.base import QsSim

# Система с 3 каналами и максимальной очередью 20
qs = QsSim(num_of_channels=3, buffer=20)

qs.set_sources(2.0, "M")   # λ = 2.0
qs.set_servers(1.0, "M")   # μ = 1.0

results = qs.run(50000)

print(f"M/M/3/20 результаты:")
print(f"  Среднее время ожидания: {results.w[0]:.4f}")
print(f"  Число потерянных заявок: {qs.dropped}")
print(f"  Вероятность потери: {qs.dropped / qs.arrived:.4f}")
```

## Сравнение с аналитическими результатами

Для проверки корректности симуляции можно сравнить результаты с аналитическими расчетами:

```python
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.io.tables import print_waiting_moments, print_sojourn_moments

# Параметры системы
arrival_rate = 0.8
service_rate = 1.0
num_channels = 2
num_jobs = 30000

# Симуляция
qs = QsSim(num_channels)
qs.set_sources(arrival_rate, "M")
qs.set_servers(service_rate, "M")
sim_results = qs.run(num_jobs)

# Аналитический расчет
calc = MMnrCalc(n=num_channels)
calc.set_sources(l=arrival_rate)
calc.set_servers(mu=service_rate)
calc_results = calc.run()

# Сравнение
print("Сравнение моментов времени ожидания:")
print_waiting_moments(sim_results.w, calc_results.w)

print("\nСравнение моментов времени пребывания:")
print_sojourn_moments(sim_results.v, calc_results.v)
```

## Параметры симуляции

### Количество заявок

Для получения стабильных результатов рекомендуется использовать достаточно большое число заявок:
- Минимум: 10,000 заявок
- Рекомендуется: 50,000+ заявок
- Для систем с высокой загрузкой: 100,000+ заявок

### Коэффициент загрузки

Перед запуском симуляции можно проверить коэффициент загрузки:

```python
qs = QsSim(num_of_channels=2)
qs.set_sources(1.5, "M")
qs.set_servers(1.0, "M")

load = qs.calc_load()
print(f"Коэффициент загрузки: {load:.4f}")

if load >= 1.0:
    print("Внимание: система перегружена!")
else:
    print("Система устойчива")
    results = qs.run(50000)
```

## Интерпретация результатов

### Время ожидания vs время пребывания

- **Время ожидания** (`w`) — время от поступления заявки до начала обслуживания
- **Время пребывания** (`v`) — полное время от поступления до завершения обслуживания
- Связь: `v = w + время_обслуживания`

### Вероятности состояний

Вероятности `p[i]` показывают долю времени, которое система проводит в состоянии с `i` заявками:

```python
results = qs.run(50000)
p = results.p

print(f"Вероятность простоя: {p[0]:.4f}")
print(f"Вероятность, что в системе 1 заявка: {p[1]:.4f}")
print(f"Вероятность, что в системе 2+ заявки: {sum(p[2:]):.4f}")
```

### Коэффициент загрузки

Коэффициент загрузки показывает, какая доля времени каналы заняты:
- ρ < 1 — система устойчива
- ρ = 1 — система на границе устойчивости
- ρ > 1 — система перегружена (очередь растет)

## Специализированные классы симуляции

Библиотека также предоставляет специализированные классы для различных типов СМО:

- **`PriorityQueueSimulator`** — системы с приоритетами (см. [Приоритетные системы](priorities.md))
- **`ForkJoinSim`** — Fork-Join системы
- **`QueueingSystemBatchSim`** — системы с пакетным поступлением
- **`NetworkSimulator`** — сети очередей (см. [Сети очередей](networks.md))

## Советы по использованию

1. **Начинайте с простых моделей** — проверьте работу на M/M/1 перед сложными системами
2. **Используйте достаточно заявок** — для точных результатов нужно много данных
3. **Проверяйте коэффициент загрузки** — убедитесь, что система устойчива
4. **Сравнивайте с аналитикой** — когда возможно, проверяйте результаты расчетом
5. **Анализируйте вероятности состояний** — они дают полную картину поведения системы

---

**См. также:**
- [Численные методы](calculation.md) — аналитические расчеты
- [Распределения](distributions.md) — справочник по распределениям
- [Примеры использования](examples.md) — расширенные примеры

