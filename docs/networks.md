# Сети очередей

Сети очередей (сети СМО) представляют собой системы, состоящие из нескольких узлов (отдельных СМО), между которыми могут перемещаться заявки. Библиотека Most-Queue поддерживает моделирование и расчет открытых сетей очередей.

## Введение в сети очередей

### Основные понятия

- **Узел сети** — отдельная СМО в сети
- **Матрица переходов** — правила маршрутизации заявок между узлами
- **Внешний поток** — заявки, поступающие в сеть извне
- **Выходной поток** — заявки, покидающие сеть

### Типы сетей

- **Открытая сеть** — заявки могут поступать из внешней среды и покидать сеть
- **Закрытая сеть** — фиксированное число заявок циркулирует в сети

## Симуляция сетей очередей

### Класс NetworkSimulator

Класс `NetworkSimulator` используется для имитационного моделирования сетей очередей.

### Создание сети

```python
from most_queue.sim.networks.network import NetworkSimulator

# Создание симулятора сети
network = NetworkSimulator()
```

### Настройка матрицы переходов

Матрица переходов определяет, как заявки перемещаются между узлами сети.

```python
import numpy as np

# Матрица переходов R
# R[i, j] - вероятность перехода из узла i в узел j
# R[i, 0] - вероятность выхода из сети из узла i
# Первая строка (индекс 0) - вход в сеть
# Последний столбец (индекс n+1) - выход из сети

R = np.matrix([
    [1, 0, 0, 0, 0, 0],      # вход: все заявки идут в узел 1
    [0, 0.4, 0.6, 0, 0, 0],  # узел 1: 40% в узел 2, 60% в узел 3
    [0, 0, 0.2, 0.4, 0.4, 0], # узел 2: 20% остаются, 40% в узел 4, 40% в узел 5
    [0, 0, 0, 0, 1, 0],      # узел 3: все в узел 5
    [0, 0, 0, 0, 1, 0],      # узел 4: все в узел 5
    [0, 0, 0, 0, 0, 1],      # узел 5: все выходят
])

network.set_sources(arrival_rate=1.0, R=R)
```

### Настройка узлов сети

Каждый узел настраивается отдельно с указанием числа каналов и параметров обслуживания.

```python
from most_queue.random.distributions import H2Distribution

# Параметры обслуживания для каждого узла
serv_params = []
num_channels = [3, 2, 3, 4, 3]  # число каналов в каждом узле

for i in range(5):  # 5 узлов
    # Создание параметров H2-распределения для узла i
    h2_params = H2Distribution.get_params_by_mean_and_cv(
        mean=2.0,
        cv=0.8
    )
    serv_params.append({
        "type": "H",
        "params": h2_params
    })

# Настройка узлов
network.set_nodes(serv_params=serv_params, n=num_channels)
```

### Запуск симуляции

```python
# Запуск симуляции на 50000 заявок
results = network.run(50000)

# Результаты
print(f"Среднее время пребывания в сети: {results.v[0]:.4f}")
print(f"Интенсивности в узлах: {results.intensities}")
print(f"Загрузки узлов: {results.loads}")
```

### Полный пример симуляции

```python
import numpy as np
from most_queue.sim.networks.network import NetworkSimulator
from most_queue.random.distributions import H2Distribution

# Создание сети
network = NetworkSimulator()

# Матрица переходов для сети с 3 узлами
R = np.matrix([
    [1, 0, 0, 0],      # вход -> узел 1
    [0, 0.5, 0.5, 0], # узел 1 -> узел 2 (50%) или узел 3 (50%)
    [0, 0, 0, 1],     # узел 2 -> выход
    [0, 0, 0, 1],     # узел 3 -> выход
])

network.set_sources(arrival_rate=2.0, R=R)

# Настройка узлов
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

network.set_nodes(serv_params=serv_params, n=num_channels)

# Симуляция
results = network.run(50000)

print(f"Время пребывания: {results.v[0]:.4f}")
print(f"Интенсивности: {results.intensities}")
print(f"Загрузки: {results.loads}")
```

## Расчет сетей очередей

### Класс OpenNetworkCalc

Класс `OpenNetworkCalc` используется для аналитического расчета открытых сетей очередей методом декомпозиции.

### Пример расчета

```python
import numpy as np
from most_queue.theory.networks.open_network import OpenNetworkCalc
from most_queue.random.distributions import H2Distribution

# Создание калькулятора
net_calc = OpenNetworkCalc()

# Матрица переходов
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Настройка внешнего потока
net_calc.set_sources(R=R, arrival_rate=2.0)

# Вычисление моментов обслуживания для каждого узла
b = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    b.append(H2Distribution.calc_theory_moments(h2_params, 4))

# Настройка узлов
net_calc.set_nodes(b=b, n=num_channels)

# Расчет
results = net_calc.run()

print(f"Интенсивности в узлах: {results.intensities}")
print(f"Загрузки узлов: {results.loads}")
print(f"Среднее время пребывания: {results.v[0]:.4f}")
```

## Сети с приоритетами

### Класс PriorityNetworkSimulator

Для симуляции сетей с приоритетными дисциплинами в узлах используется класс `PriorityNetworkSimulator`.

### Пример сети с приоритетами

```python
import numpy as np
from most_queue.sim.networks.priority_network import PriorityNetworkSimulator
from most_queue.random.distributions import GammaDistribution

# Создание сети с приоритетами
network = PriorityNetworkSimulator()

# Матрица переходов
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.6, 0.4, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Настройка потоков для каждого класса приоритета
arrival_rates = [1.0, 0.5]  # интенсивности для двух классов
network.set_sources(arrival_rates=arrival_rates, R=R)

# Настройка узлов с приоритетами
serv_params = []
num_channels = [2, 3]
num_classes = 2

for i in range(2):
    # Параметры обслуживания для каждого класса в узле
    node_params = []
    for j in range(num_classes):
        gamma_params = GammaDistribution.get_params_by_mean_and_cv(
            mean=1.5 + j * 0.5,  # разные средние для разных классов
            cv=0.7
        )
        node_params.append({
            "type": "Gamma",
            "params": gamma_params
        })
    serv_params.append(node_params)

network.set_nodes(
    serv_params=serv_params,
    n=num_channels,
    priority="PR"  # или "NP"
)

# Симуляция
results = network.run(50000)

# Результаты для каждого класса
print(f"Время пребывания класса 1: {results.v[0][0]:.4f}")
print(f"Время пребывания класса 2: {results.v[1][0]:.4f}")
```

### Расчет сетей с приоритетами

```python
from most_queue.theory.networks.open_network_prty import OpenNetworkPrtyCalc

calc = OpenNetworkPrtyCalc()
calc.set_sources(arrival_rates=[1.0, 0.5], R=R)

# Моменты обслуживания для каждого класса в каждом узле
b = []  # b[i][j] - моменты для узла i, класса j
calc.set_nodes(b=b, n=num_channels, priority="PR")

results = calc.run()
```

## Оптимизация сетей

Библиотека также предоставляет методы оптимизации матрицы переходов сети для минимизации времени пребывания заявок.

### Пример оптимизации

```python
from most_queue.theory.networks.opt.transition import TransitionOptimization

# Создание оптимизатора
optimizer = TransitionOptimization()

# Начальная матрица переходов
R0 = np.matrix([...])

# Оптимизация
R_opt = optimizer.optimize(
    R0=R0,
    arrival_rate=2.0,
    b=b,
    n=num_channels
)

print(f"Оптимизированная матрица:\n{R_opt}")
```

## Структура результатов

### NetworkResults

```python
@dataclass
class NetworkResults:
    v: list[float] | None              # моменты времени пребывания в сети
    intensities: list[float] | None    # эффективные интенсивности в узлах
    loads: list[float] | None          # загрузки узлов
    duration: float = 0.0              # время расчета/симуляции
    arrived: int = 0                    # число поступивших заявок (симуляция)
    served: int = 0                     # число обслуженных заявок (симуляция)
```

### NetworkResultsPriority

Для сетей с приоритетами:

```python
@dataclass
class NetworkResultsPriority:
    v: list[list[float]] | None        # моменты для каждого класса
    intensities: list[list[float]] | None  # интенсивности для каждого класса
    loads: list[float] | None          # загрузки узлов
    duration: float = 0.0
    arrived: int = 0
    served: int = 0
```

## Построение матрицы переходов

### Правила построения

1. **Первая строка (индекс 0)** — вход в сеть
   - `R[0, i]` — вероятность поступления заявки в узел `i-1`
   - Сумма должна быть равна 1

2. **Строки 1..n** — узлы сети
   - `R[i, j]` — вероятность перехода из узла `i-1` в узел `j-1`
   - `R[i, 0]` — вероятность выхода из сети из узла `i-1`
   - `R[i, n+1]` — выход из сети (обычно 1 для последнего столбца)

3. **Последний столбец** — выход из сети
   - Обычно все элементы равны 1

### Пример построения

```python
import numpy as np

# Сеть с 3 узлами
num_nodes = 3

R = np.matrix([
    # Вход -> узлы
    [1, 0, 0, 0],           # все заявки идут в узел 1
    
    # Узел 1
    [0, 0.3, 0.5, 0.2],     # 30% остаются, 50% в узел 2, 20% выходят
    
    # Узел 2
    [0, 0, 0.4, 0.6],       # 40% остаются, 60% выходят
    
    # Узел 3
    [0, 0, 0, 1],           # все выходят
])
```

## Советы по работе с сетями

1. **Проверяйте матрицу переходов** — убедитесь, что сумма вероятностей в каждой строке равна 1
2. **Проверяйте устойчивость** — каждый узел должен быть устойчив (ρ < 1)
3. **Используйте симуляцию для проверки** — сравните результаты расчета и симуляции
4. **Анализируйте загрузки узлов** — найдите узкие места в сети
5. **Оптимизируйте маршрутизацию** — используйте методы оптимизации для улучшения характеристик

## Примеры использования

Подробные примеры можно найти в тестах:
- `test_network_no_prty.py` — сеть без приоритетов
- `test_network_im_prty.py` — сеть с приоритетами
- `test_network_opt.py` — оптимизация сети

---

**См. также:**
- [Симуляция СМО](simulation.md) — основы симуляции
- [Приоритетные системы](priorities.md) — работа с приоритетами
- [Примеры использования](examples.md) — практические примеры

