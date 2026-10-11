# Руководство по симуляции СМО

[🇬🇧 English version](simulation.md)

Это руководство описывает использование модуля симуляции библиотеки Most-Queue для моделирования систем массового обслуживания.

## Введение

Для заявок, занимающих несколько серверов одновременно, см.
[MSJ с общим обслуживанием и backfilling](models/msj.ru.md). `MsjGeneralSim` поддерживает
генерацию пуассоновского входа и replay неизменяемых трасс, прогнозы длительности,
warm-up по приходам, квантили по классам и диагностику нарушенных резервирований.
Дисциплины резервирования: EASY (защита головы) и conservative (все ожидающие
заявки), наряду с FCFS. Добавлены FirstFit, MSF, MSFQ, Adaptive Quickswap без
прогноза, а также явно прерывающий ServerFilling:
[правила и пример](msj_packing.md). ServerFilling требует степеней двойки,
продолжает работу без потерь/накладных расходов и включает паузы в W. Его нельзя
трактовать как непрерывающий baseline; интервалы исполнения доступны в журнале.
Отдельный [MsjCheckpointSim](msj_checkpoint.md) добавляет фазы checkpoint/resume
с удержанием ресурсов и блокировкой новых прерываний на время overhead.
Полезная загрузка измеряется отдельно; нулевые затраты и защита дают прежний SF.
Опция [min_service_time](msj_protected_service.md) защищает каждый полезный
интервал после старта/resume и добавляет события пересмотра по истечении q.
При q=0 модель прежняя; q>0 меняет расписание даже при бесплатном c=r=0.
Обучение прогнозов по признакам и split-conformal калибровка общая или по ресурсным классам:
[MSJ runtime prediction](msj_runtime_prediction.md).
История с независимым правым цензурированием и прогноз остатка по возрасту:
[MSJ age-aware forecasts](msj_age_runtime.md). Опция `remaining_predictor`
обновляет активные работы, явно сообщает недоступный хвост и сохраняет прежние обещания.

Для исторических данных см. [калибровку по реальной трассе](real_trace_calibration.md):
строгий SWF-адаптер, временное разделение без утечки исходов и сравнение
empirical/Exp/PH/lognormal при общих приходах и точных потребностях K.
Первый эксперимент использует подвыборку завершённых работ, не весь кластер.
Продолжение — [история и зависимые длительности](real_trace_temporal.md): четыре
заранее заданные границы, expanding/recent, отдельно среднее и форма, точный K,
циклические блоки рангов с rank-iid-контролем. Прогнозы фиксированы по прошлому;
семантика дисциплин не меняется.
Для начальных running/waiting и раздельного учёта исходов см.
[lifecycle replay](real_trace_lifecycle.md). `MsjLifecycleSim.run_lifecycle`
использует прежние непрерывающие диспетчеры, различая completed/cancelled/timeout.
Runtime cap от старта не превращает прекращение в успешное завершение;
ненаблюдаемое время отмены в очереди не реконструируется.
Для Acme/Kalos см. [современную GPU-трассу](modern_gpu_trace.md): аудит end-start,
точные запросы GPU и отдельные failed/node_failed. Записанный timed_out не задаёт
выдуманный runtime budget. Номинальный общий пул не восстанавливает квоты и
топологию; равенство дисциплин при нулевой очереди не проверяет их качество.

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

- **`PriorityQueueSimulator`** — системы с приоритетами (см. [Приоритетные системы](priorities.ru.md))
- **`ForkJoinSim`** — Fork-Join системы
- **`QueueingSystemBatchSim`** — системы с пакетным поступлением
- **`NetworkSimulator`** — сети очередей (см. [Сети очередей](networks.ru.md))
- **`SizeBasedQsSim`** — одноканальная M/G/1 с дисциплинами по размеру или предсказанию (см. ниже)

## SizeBasedQsSim (размер-зависимые дисциплины)

Теоретические формулы для SRPT/SJF/PSJF/SPJF, схема численного интегрирования в калькуляторах и типичная процедура сравнения с симуляцией описаны на странице **[SRPT / SPJF: методы и верификация](srpt_spjf_methods.ru.md)**.

Класс `most_queue.sim.size_based.SizeBasedQsSim` наследует `QsSim`, но:

1. **Размер при приходе** — для всех дисциплин кроме `FCFS` при создании заявки вызывается `_sample_size`: из того же распределения, что и `set_servers`, сэмплируется `Task.original_size`, в `Task.service_remaining` записывается полный объём работы, опционально `Task.predicted_size` через предиктор.
2. **Очередь** — min-heap по ключу ранга (`PrioritySizeQueue`): для SRPT/SPRPT перед вытеснением обновляется `service_remaining` по фактическому остатку на канале.
3. **Предиктор** — `set_predictor(obj)` для `SPJF`, `PSPJF`, `SPRPT`; объект с методом `predict(true_size, rng) -> float`. Встроенный **`PerfectSimPredictor`**: \(Y=X\). Для шума в симуляции: `most_queue.sim.utils.predictor.ExpNoiseSimPredictor`, `LognormalNoiseSimPredictor`.
4. **Slowdown** — при `track_slowdown=True` после каждого завершения обслуживания добавляется отношение \(T/X\) в список; чтение копии: `get_slowdown()`.

Поддерживаемые значения `discipline`: `"FCFS"`, `"SJF"`, `"PSJF"`, `"SRPT"`, `"SPJF"`, `"PSPJF"`, `"SPRPT"`. Сейчас только **`num_of_channels=1`**.

**Пример SRPT (H₂-обслуживание, пуассоновский вход):**

```python
import numpy as np
from most_queue.random.distributions import H2Distribution
from most_queue.sim.size_based import SizeBasedQsSim

sim = SizeBasedQsSim(1, discipline="SRPT", verbose=False)
sim.generator = np.random.default_rng(42)
h2 = H2Distribution.get_params_by_mean_and_cv(0.7, 1.2)
sim.set_servers(h2, "H")
sim.set_sources(1.0, "M")
res = sim.run(100_000)
print(res.v[0], res.w[0])
```

**Пример SPJF с экспоненциальным шумом предсказания:**

```python
import numpy as np
from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.sim.utils.predictor import ExpNoiseSimPredictor

sim = SizeBasedQsSim(1, discipline="SPJF", verbose=False)
sim.set_predictor(ExpNoiseSimPredictor())
sim.generator = np.random.default_rng(7)
sim.set_servers(1.0, "M")
sim.set_sources(0.5, "M")
res = sim.run(50_000)
```

Сравнение с `QsSim` в режиме `FCFS`: см. тест `tests/test_size_based_fcfs_regression.py`.

## Советы по использованию

Для **неограниченного динамического батчинга** доступен
[`DynamicBatchingSim`](models/dynamic-batching.ru.md#имитация):
пять законов с аффинным средним, `set_sources/set_servers/run(num_of_jobs=...)`.
Прогрев исключается из всех метрик; последний пакет сохраняется целиком, поэтому
число измеренных заявок может превышать запрошенный минимум.
Эмпирическая CDF не является аналитической гарантией хвоста.

1. **Начинайте с простых моделей** — проверьте работу на M/M/1 перед сложными системами
2. **Используйте достаточно заявок** — для точных результатов нужно много данных
3. **Проверяйте коэффициент загрузки** — убедитесь, что система устойчива
4. **Сравнивайте с аналитикой** — когда возможно, проверяйте результаты расчетом
5. **Анализируйте вероятности состояний** — они дают полную картину поведения системы

---

## Переменная длина вывода и закрытый elastic batching

[`TokenBatchingSim`](models/token-batching.ru.md) поддерживает dynamic, fixed и
закрытый elastic. Полная длина каждой заявки выбирается при приходе; elastic
отдаёт короткие ответы раньше и уменьшает активную группу на токеновых границах.
`batch_size` задаёт физический cap dynamic/elastic или точный размер fixed-пакета.
Новые заявки не входят в работающий пакет. `run(num_of_jobs=30_000)` возвращает
эмпирические моменты, фактическую временную PMF, занятость и CDF, последовательно
исключая прогрев по целым пакетам. Неопределённость проверяется повторными seeds
и увеличением бюджета. Для общих неаффинных времён доступен также
`DynamicBatchingSim.set_service_callback(sampler, mean_bound=(a,c))`.

## Prefill и decode на границах итераций

[ГЛ2 replay](research/llm-mark-order-validation-2026.md) использует audited
successful API cohorts BurstGPT, сохраняет whole pairs отдельно в measured
suffix и закрепляет warmup. Не восстанавливает исходный GPU TTFT.
Независимый [finite-loss oracle](../examples/stage_aware_ttft_oracle.py)
дополнительно сверяет mid-iteration потери и accepted-arrival rewards
[стадийного калькулятора](models/stage-aware-ttft.ru.md).

`KvCacheCongestionSim` — независимая [LPF/restart модель](models/kv-cache-congestion.ru.md):
целые запросы, класс и decode-возраст, Poisson/replay, фиксированный горизонт
без drain. Cap целый; дробная флюидная скорость не округляется автоматически.
FCFS-допуск с блокировкой головы не равен пропорциональной карте. Сохраняются
unfinished jobs, потерянная работа и точные балансы, не censored latency.
[Независимые проверки Л5](research/kv-cache-congestion-validation-2026.md).

### Гетерогенное групповое обслуживание

См. также [гетерогенное групповое обслуживание](models/heterogeneous-bulk-service.ru.md):
`BulkServiceHeterogeneousSim` хранит индивидуальные экспоненциальные приборы,
общую FCFS очередь, threshold/input приоритет и раздельные RNG.
Horizon без drain; площади учитывают незавершённых, cohort tails — все
приходы когорты с достаточным follow-up, не только завершённых.
Есть request/area identities и arrival/service replay.
[Проверки Л8](research/heterogeneous-bulk-service-validation-2026.md) сопоставляют
DES с независимой физической CTMC и точным PH-калькулятором.

### SMDP-управление батчингом

`BatchSmdpSim` независимо моделирует [запуск пакета Л6](models/smdp-batching.ru.md)
в неограниченной event-time FCFS-очереди. Фиксированные время/энергия пакетов,
Poisson или отсортированный replay, callable реального населения либо явное
хвостовое расширение конечной политики, горизонт без drain. Приход во время
обслуживания не вызывает решения. Балансы запросов/площади включают unfinished.
Энергия начисляется при старте; стоимость без warmup использует area/rate прокси
отклика, не среднее только завершённых задержек. [Отчёт](research/smdp-batching-validation-2026.md)
разделяет конечный оптимум, устойчивость хвоста и короткие наблюдения DES.

`KvCacheStabilitySim` — отдельная [модель KV-памяти](models/kv-cache-stability.ru.md):
постоянные слоты, joint lengths, растущий footprint, ongoing-first максимальная
упаковка и бесплатная выгрузка на CPU. `run(num_of_slots, warmup_slots=...,
arrivals_by_slot=..., record_trace=...)` наблюдает горизонт **без drain**.
Счётчики включают warmup; throughput, доля занятости и memory_fraction — после
него. Смещённые w/v только завершённых заявок и стационарный PMF не выдаются.
Сохранение работы, packing LP, насыщенные циклы и flow-проверки:
[отчёт Л4](research/kv-cache-stability-validation-2026.md).
Это не hard-resident/paid-eviction дисциплина и не физический GPU-симулятор.

[`PrefillDecodeSim`](models/prefill-decode.ru.md) обслуживает индивидуальные
целые токены, без экспоненциальной подмены обслуживания. На boundary допускает
FCFS до cap, выдаёт по токену каждому decoder и расходует оставшийся бюджет
на FCFS-prefill. Приход не прерывает уже работающую итерацию. Поддерживает
`set_sources`, `set_servers`, `run(num_of_jobs)`, явный
`replay(arrivals, inputs, outputs)` и sampler индивидуальных длин.
Возвращает эмпирические W/V-моменты, TTFT, token gaps, timestamp-массивы,
временную занятость и busy fraction. Запись итераций включается отдельно.

`itl_mean` учитывает интервалы после первого токена; `all_decode_mean`
включает первый decode-интервал для сопоставления с конвенцией статьи.
Warmup исключается по индексу прихода; time occupancy заканчивается на
последнем приходе, до drain. Завершение конечной перегруженной когорты не
доказывает стационарности. Соседний `PrefillDecodeCalc` решает марковское
приближение, поэтому оно не точный эталон DES. См.
[независимые M/D/1 и trace-проверки](research/prefill-decode-validation-2026.md).

**См. также:**

Независимый `examples.stage_aware_ttft_oracle.replay_loss` проверяет и
`ReducedStageAwareTtftCalc`: реальные потери внутри итерации, cohort timestamps
первого токена/завершения, без импорта аналитических rewards.
[Отчёт о сокращении](research/stage-aware-ttft-reduction-validation-2026.md)
отделяет finite-loss арбитр от infinite token DES.

`PhaseAwareTtftCalc`: fresh50000-job three-seed token DES, saturated loads.6/.85,
M2/M4;12 средних внутри replicate t95 intervals. Finite-cohort drain не proof
устойчивости. [Проверка](research/phase-aware-stability-validation-2026.md).

`InfiniteStageAwareTtftCalc` сверяется с независимым бесконечным
`PrefillDecodeSim.replay` при iid Poisson/mark inputs. Аналитический model
interval и DES replicate confidence interval описывают разные ошибки;
случайное выборочное среднее не обязано попасть в узкий model interval.
[Infinite проверка](research/infinite-stage-aware-ttft-validation-2026.md)
отделяет fresh30000-job репликации от исторических reused references.

- [Численные методы](calculation.ru.md) — аналитические расчеты
- [Распределения](distributions.ru.md) — справочник по распределениям
- [Примеры использования](examples.ru.md) — расширенные примеры
