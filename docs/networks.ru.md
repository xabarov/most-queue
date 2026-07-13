# Сети очередей

[🇬🇧 English version](networks.md)

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

## Точные сети Джексона (product-form)

Для марковской открытой сети (пуассоновский внешний поток, экспоненциальные
узлы M/M/n) класс `JacksonNetworkCalc` даёт **точное** решение в форме
произведения (Джексон, 1957/1963). Средние значения точны, поэтому класс
служит эталоном для приближённой декомпозиции `OpenNetworkCalc`.

```python
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc

calc = JacksonNetworkCalc()
calc.set_sources(arrival_rate=1.0, R=R)      # тот же формат маршрутизации, что и в OpenNetworkCalc
calc.set_nodes(mu=[1.0, 2.0, 1.5], n=[2, 3, 2])
res = calc.run()
print(res.v[0], res.mean_jobs, res.loads)    # точные средние
```

## QNA — непуассоновские внутренние потоки (Уитт)

Базовая декомпозиция считает все внутренние потоки пуассоновскими. Класс
`OpenNetworkCalcQNA` реализует **Queueing Network Analyzer** Уитта (1983):
квадратичный коэффициент вариации интервалов между заявками
распространяется через операции выхода, разрежения и слияния потоков, а
каждый узел аппроксимируется как GI/G/n с поправкой Крамера —
Лангенбах-Бельца. На сетях с высоковариативным обслуживанием ошибка падает
существенно (например, с ~20% до ~2% на тандеме H2 с c² = 4 при загрузке 0.8).

```python
from most_queue.theory.networks.qna import OpenNetworkCalcQNA

qna = OpenNetworkCalcQNA()
qna.set_sources(arrival_rate=1.0, R=R, arrival_cv2=1.0)
qna.set_nodes(b=b, n=num_channels)           # начальные моменты обслуживания по узлам
res = qna.run()
print(res.v[0], qna.arrival_cv2_nodes)       # среднее время пребывания + c² потока в узлах
```

## Закрытые сети (MVA и свёртка Бьюзена)

В закрытой сети нет внешнего потока: фиксированная популяция из N заявок
циркулирует по узлам (модель Гордона — Ньюэлла). Класс `ClosedNetworkCalc`
содержит три решателя:

- `method="mva"` — точный Mean Value Analysis (Райзер — Лавенберг, 1980),
  включая многоканальные станции (через маргинальные вероятности) и
  delay-узлы с бесконечным числом приборов — для delay-узла передайте
  `n=[..., None, ...]`;
- `method="convolution"` — алгоритм свёртки Бьюзена (1973) для
  нормализационной константы G(N); совпадает с MVA до машинной точности;
- `method="schweitzer"` — приближённый MVA Швейцера — Барда для больших
  популяций (многоканальные станции — аппроксимация Зайдмана).

```python
import numpy as np
from most_queue.theory.networks.closed_network import ClosedNetworkCalc

# Модель центрального сервера: CPU + 2 диска, 8 заявок
routing = np.array([
    [0.1, 0.5, 0.4],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
])

calc = ClosedNetworkCalc(method="mva")
calc.set_sources(R=routing, N=8)             # матрица m x m, суммы строк равны 1
calc.set_nodes(b=[0.02, 0.06, 0.08], n=[2, 1, 1])
res = calc.run()
print(res.throughput, res.mean_jobs, res.v[0])   # X, L_i, среднее время цикла N/X
```

Парный симулятор — `ClosedNetworkSim` (`most_queue.sim.networks.closed_network`)
с тем же интерфейсом `set_sources` / `set_nodes` (распределения обслуживания
в нотации Кендалла) и параметром `seed`.

## G-сети (отрицательные заявки, product-form Геленбе)

Для узлов M/M/1 класс `GNetworkCalc` даёт **точное** решение G-сети в форме
произведения (Геленбе, 1991): заявка после обслуживания переходит в
следующий узел как позитивная либо как **негативный сигнал**, удаляющий
одну заявку из непустого узла. Внешние позитивные и негативные пуассоновские
потоки задаются по узлам. Нелинейные уравнения трафика решаются методом
простой итерации. Класс дополняет приближённую декомпозицию
`NegativeNetworkCalc` (DISASTER/RCS, узлы M/G/n) точным эталоном для
марковского одноканального случая.

```python
import numpy as np
from most_queue.theory.networks.g_network import GNetworkCalc

calc = GNetworkCalc()
calc.set_sources(
    positive_rates=[0.5, 0.2],
    P_plus=np.array([[0.0, 0.4], [0.2, 0.0]]),    # переходы позитивными заявками
    P_minus=np.array([[0.0, 0.2], [0.1, 0.0]]),   # переходы негативными сигналами
    negative_rates=[0.1, 0.0],                     # внешние негативные потоки
)
calc.set_nodes(mu=[1.0, 1.5])
res = calc.run()
print(res.loads, res.mean_jobs, res.negative_intensities)
```

## BCMP-сети (мультиклассовый product-form)

Теорема BCMP (Баскет — Чанди — Мунц — Паласиос, 1975) распространяет форму
произведения на **мультиклассовые** сети с четырьмя типами станций: FCFS
(экспоненциальное обслуживание, интенсивность не зависит от класса), PS,
LCFS-PR и IS (delay); станции PS/LCFS-PR/IS нечувствительны к виду
распределения — важны только средние времена обслуживания.

- `BCMPOpenNetworkCalc` — открытая сеть, пуассоновские потоки и матрицы
  маршрутизации по классам; точные средние по классам.
- `BCMPClosedNetworkCalc` — закрытая многоцепочечная сеть, решается
  **точным мультичейн-MVA** (рекурсия по всем векторам популяций).

```python
from most_queue.theory.networks.bcmp_network import BCMPClosedNetworkCalc

calc = BCMPClosedNetworkCalc()
calc.set_sources(R=[routing_class1, routing_class2], N=[3, 2])
calc.set_nodes(
    s=[[0.5, 0.8], [0.3, 0.4]],                  # s[узел][класс] — средние времена обслуживания
    station_types=["ps", "fcfs"],
)
res = calc.run()
print(res.throughput, res.mean_jobs)             # по классам
```

## Тандемы с конечными буферами (блокировка после обслуживания)

`TandemBlockingCalc` — тандем производственной линии, где узел i вмещает не
более K_i заявок: заявка, закончившая обслуживание, остаётся на приборе
(блокируя его), пока следующий узел полон; внешние заявки, заставшие первый
узел полным, теряются. Двухпроходная декомпозиция (Brandwajn–Jow 1988;
Dallery–Frein 1993): пропускная способность в пределах ~1% от точной CTMC на
коротких линиях. Парный симулятор — `TandemBlockingSim`
(`most_queue.sim.networks.tandem_blocking`).

```python
from most_queue.theory.networks.blocking import TandemBlockingCalc

calc = TandemBlockingCalc()
calc.set_sources(arrival_rate=0.8)
calc.set_nodes(mu=[1.0, 1.2], capacity=[4, 3])   # None = узел без ограничения
res = calc.run()
print(calc.throughput, calc.loss_prob, calc.blocking_probs)
```

## Fork-join станции внутри сети

`OpenNetworkCalcForkJoin` встраивает fork-join станции в маршрутизируемую
открытую сеть: заявка разветвляется на k параллельных одноканальных ветвей и
продолжает путь после завершения последней подзадачи (аппроксимации отклика
Нельсона–Тантави / Вармы). Парный симулятор — `ForkJoinNetworkSim`.

```python
from most_queue.theory.networks.fork_join_network import OpenNetworkCalcForkJoin

net = OpenNetworkCalcForkJoin()
net.set_sources(arrival_rate=0.5, R=R)
net.set_nodes([
    {"kind": "queue", "mu": 0.4, "n": 2},
    {"kind": "fork_join", "mu": 1.0, "k": 3},
])
res = net.run()
```

## MAP-вход сети

`NetworkSimulator.set_sources(..., source_kendall="MAP", source_params=map_params)`
подаёт в сеть пачечный MAP-поток; на аналитической стороне передайте в QNA
вариабельность интервалов MAP через `map_arrival_cv2(map_params)`. QNA —
двухмоментный метод: он учитывает c² интервалов, но не автокорреляцию,
поэтому для сильно коррелированных MAP даёт нижнюю оценку перегрузки.

## Нестационарные сети (PSA)

`TimeVaryingNetworkCalc` решает открытую марковскую сеть с интенсивностью
λ(t) кусочно-стационарной аппроксимацией — стационарный снимок сети Джексона
в каждой точке сетки (точно при медленной модуляции). Парный симулятор —
`TimeVaryingNetworkSim` (NHPP прореживанием, статистика по фазовым корзинам).

```python
from most_queue.theory.networks.time_varying_network import TimeVaryingNetworkCalc

calc = TimeVaryingNetworkCalc()
calc.set_sources(lam_fn=lambda t: 0.5 + 0.2 * math.sin(2 * math.pi * t / 2000), R=R)
calc.set_nodes(mu=[1.0, 1.4], n=[1, 1])
res = calc.run(t_grid=range(0, 2000, 100))   # res.v, res.mean_jobs_total по точкам
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

## Сети с отрицательными заявками

### Расчет сетей с отрицательными заявками

Для аналитического расчета сетей с отрицательными заявками используется класс `NegativeNetworkCalc`, реализующий метод потоковой декомпозиции. Подробное описание математического метода расчета с формулами приведено в документе [Расчет сетей с отрицательными заявками](negative_networks_calculation.ru.md).

### Класс NegativeNetworkCalc

Класс `NegativeNetworkCalc` используется для аналитического расчета открытых сетей очередей с отрицательными заявками методом декомпозиции.

### Пример расчета

```python
import numpy as np
from most_queue.theory.networks.negative_network import NegativeNetworkCalc
from most_queue.sim.negative import NegativeServiceType
from most_queue.random.distributions import H2Distribution

# Создание калькулятора с глобальными отрицательными заявками
net_calc = NegativeNetworkCalc(negative_arrival_type="global")

# Матрица переходов
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Настройка источников
net_calc.set_sources(
    arrival_rate=2.0,
    R=R,
    negative_arrival_rate=0.1  # глобальная интенсивность отрицательных заявок
)

# Вычисление моментов обслуживания для каждого узла
b = []
num_channels = [2, 3, 2]
negative_types = [
    NegativeServiceType.DISASTER,  # узел 1: тип DISASTER
    NegativeServiceType.RCS,       # узел 2: тип RCS
    NegativeServiceType.DISASTER,  # узел 3: тип DISASTER
]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    b.append(H2Distribution.calc_theory_moments(h2_params, 4))

# Настройка узлов
net_calc.set_nodes(b=b, n=num_channels, negative_types=negative_types)

# Расчет
results = net_calc.run()

print(f"Интенсивности в узлах: {results.intensities}")
print(f"Загрузки узлов: {results.loads}")
print(f"Среднее время пребывания: {results.v[0]:.4f}")
```

### Пример с поузловыми отрицательными заявками

```python
# Создание калькулятора с поузловыми отрицательными заявками
net_calc = NegativeNetworkCalc(negative_arrival_type="per_node")

# Настройка источников с индивидуальными интенсивностями
net_calc.set_sources(
    arrival_rate=2.0,
    R=R,
    negative_arrival_rates=[0.1, 0.05, 0.15]  # индивидуальные интенсивности для каждого узла
)

# Остальная настройка аналогична предыдущему примеру
```

### Класс NegativeNetwork

Для симуляции сетей с отрицательными заявками (negative jobs) в каждом узле используется класс `NegativeNetwork`. Отрицательные заявки могут прерывать обслуживание обычных (положительных) заявок в зависимости от типа отрицательного обслуживания.

### Типы отрицательных заявок

Отрицательные заявки могут иметь следующие типы воздействия на систему:

- **DISASTER** — удаляет все заявки из узла (в обслуживании и в очереди)
- **RCS** (Remove Customer in Service) — удаляет одну заявку из обслуживания
- **RCH** (Remove Customer at Head) — удаляет заявку из начала очереди
- **RCE** (Remove Customer at End) — удаляет заявку из конца очереди

### Типы поступления отрицательных заявок

`NegativeNetwork` поддерживает два режима поступления отрицательных заявок:

1. **"global"** — отрицательные заявки поступают глобально и влияют на все узлы одновременно
2. **"per_node"** — каждый узел имеет свой собственный поток отрицательных заявок

### Создание сети с отрицательными заявками

```python
from most_queue.sim.networks.negative_network import NegativeNetwork
from most_queue.sim.negative import NegativeServiceType
import numpy as np

# Создание сети с глобальными отрицательными заявками
network = NegativeNetwork(negative_arrival_type="global")

# Или с индивидуальными отрицательными заявками для каждого узла
network = NegativeNetwork(negative_arrival_type="per_node")
```

### Настройка источников

#### Глобальные отрицательные заявки

```python
# Матрица переходов
R = np.matrix([
    [1, 0, 0, 0],      # вход -> узел 1
    [0, 0.5, 0.5, 0], # узел 1 -> узел 2 (50%) или узел 3 (50%)
    [0, 0, 0, 1],     # узел 2 -> выход
    [0, 0, 0, 1],     # узел 3 -> выход
])

# Настройка источников с глобальными отрицательными заявками
network.set_sources(
    positive_arrival_rate=2.0,      # интенсивность положительных заявок
    R=R,
    negative_arrival_rate=0.1       # интенсивность глобальных отрицательных заявок
)
```

#### Индивидуальные отрицательные заявки для каждого узла

```python
# ВАЖНО: set_nodes() должен быть вызван ПЕРЕД set_sources() для per_node типа
network.set_nodes(...)  # см. ниже

# Настройка источников с индивидуальными отрицательными заявками
network.set_sources(
    positive_arrival_rate=2.0,
    R=R,
    negative_arrival_rates=[0.1, 0.05, 0.15]  # интенсивности для каждого узла
)
```

### Настройка узлов

```python
from most_queue.random.distributions import H2Distribution

# Параметры обслуживания для каждого узла
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

# Типы отрицательных заявок для каждого узла
negative_types = [
    NegativeServiceType.DISASTER,  # узел 1: удаляет все заявки
    NegativeServiceType.RCS,       # узел 2: удаляет заявку из обслуживания
    NegativeServiceType.RCH,        # узел 3: удаляет заявку из начала очереди
]

# Настройка узлов
network.set_nodes(
    serv_params=serv_params,
    n=num_channels,
    negative_types=negative_types,  # типы отрицательных заявок
    buffers=[None, 50, None]        # размеры буферов (опционально)
)
```

### Полный пример

```python
import numpy as np
from most_queue.sim.networks.negative_network import NegativeNetwork
from most_queue.sim.negative import NegativeServiceType
from most_queue.random.distributions import H2Distribution

# Создание сети с глобальными отрицательными заявками
network = NegativeNetwork(negative_arrival_type="global")

# Матрица переходов
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

# Настройка источников
network.set_sources(
    positive_arrival_rate=2.0,
    R=R,
    negative_arrival_rate=0.1  # глобальная интенсивность отрицательных заявок
)

# Настройка узлов
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

# Все узлы используют DISASTER по умолчанию
network.set_nodes(serv_params=serv_params, n=num_channels)

# Симуляция
results = network.run(50000)

print(f"Время пребывания: {results.v[0]:.4f}")
print(f"Обслужено заявок: {results.served}")
print(f"Поступило заявок: {results.arrived}")
```

### Пример с индивидуальными отрицательными заявками

```python
# Создание сети с индивидуальными отрицательными заявками
network = NegativeNetwork(negative_arrival_type="per_node")

# Сначала настраиваем узлы (обязательно перед set_sources для per_node)
serv_params = []
num_channels = [2, 3, 2]

for i in range(3):
    h2_params = H2Distribution.get_params_by_mean_and_cv(mean=1.5, cv=0.7)
    serv_params.append({"type": "H", "params": h2_params})

negative_types = [
    NegativeServiceType.DISASTER,
    NegativeServiceType.RCS,
    NegativeServiceType.RCH,
]

network.set_nodes(
    serv_params=serv_params,
    n=num_channels,
    negative_types=negative_types
)

# Затем настраиваем источники
R = np.matrix([
    [1, 0, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

network.set_sources(
    positive_arrival_rate=2.0,
    R=R,
    negative_arrival_rates=[0.1, 0.05, 0.15]  # индивидуальные интенсивности
)

# Симуляция
results = network.run(50000)
```

### Важные замечания

1. **Порядок вызовов для per_node типа**: При использовании `negative_arrival_type="per_node"` необходимо сначала вызвать `set_nodes()`, а затем `set_sources()`, так как для настройки источников нужно знать количество узлов.

2. **Отключение отрицательных заявок**: Чтобы отключить отрицательные заявки, передайте `negative_arrival_rate=None` (для global) или `negative_arrival_rates=None` (для per_node).

3. **Типы отрицательных заявок по умолчанию**: Если не указать `negative_types` в `set_nodes()`, все узлы будут использовать `NegativeServiceType.DISASTER` по умолчанию.

4. **Результаты**: `NegativeNetwork` возвращает стандартный объект `NetworkResults` с информацией о времени пребывания, количестве обслуженных и поступивших заявок.

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
- `test_negative_network.py` — сеть с отрицательными заявками

---

**См. также:**
- [Симуляция СМО](simulation.ru.md) — основы симуляции
- [Приоритетные системы](priorities.ru.md) — работа с приоритетами
- [Примеры использования](examples.ru.md) — практические примеры

