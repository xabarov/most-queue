# Roadmap: поддержка дисциплин SRPT и SPJF в `most_queue`

> Источники-первоисточники, на которые опираемся:
> - Schrage L.E. *A Proof of the Optimality of the Shortest Remaining Processing Time Discipline.* Operations Research, 16(3), 1968, 687–690.
> - Schrage L.E., Miller L.W. *The Queue M/G/1 with the Shortest Remaining Processing Time Discipline.* Operations Research, 14(4), 1966, 670–684.
> - Mitzenmacher M. *Scheduling with Predictions and the Price of Misprediction.* ITCS 2020.
> - Mitzenmacher M., Shahout R. *Queueing, Predictions, and LLMs: Challenges and Open Problems.* Stochastic Systems, 2025 (arXiv:2503.07545).
> - Резюме разобраны в `works/queueing_systems_review/SRPT/shrage.md` и `works/queueing_systems_review/SRPT/SPJF.md`.

## 1. Цель

Добавить в библиотеку поддержку size-based scheduling-дисциплин (с известными или предсказанными размерами заявок) и научиться сравнивать аналитические формулы с результатами имитационного моделирования по образцу `tests/test_mg1_calc.py`.

Покрываемые дисциплины (минимально-целевой набор для первой итерации, **single-server, infinite buffer, Poisson arrivals**):

| Сокр. | Полное название                          | Preemptive? | На что смотрит ранг                |
|-------|-------------------------------------------|-------------|------------------------------------|
| FCFS  | First-Come-First-Served (baseline M/G/1)  | нет         | время прихода                       |
| SJF   | Shortest Job First (= SPT)                | нет         | истинный размер $x$                 |
| PSJF  | Preemptive Shortest Job First             | да          | истинный размер $x$ (изначальный)   |
| **SRPT** | Shortest Remaining Processing Time     | да          | остаток $x - a(t)$                  |
| **SPJF** | Shortest Predicted Job First           | нет         | предсказание $y$                    |
| PSPJF | Preemptive Shortest Predicted Job First   | да          | предсказание $y$                    |
| SPRPT | Shortest Predicted Remaining PT           | да          | $\max(0, y - a(t))$                 |

Стартовый minimum: **SRPT**, **SPJF**, **SJF**, **PSJF** для **M/G/1**. Остальные — следующая итерация.

## 2. Что нужно от теории

### 2.1 SRPT для M/G/1 (Schrage–Miller 1966)

Условное среднее время отклика для задачи размера $x$:

$$
\mathbb{E}[T^{\text{SRPT}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\, dt + x^2 (1 - F(x))}{2(1 - \rho_x)^2} + \int_0^x \frac{dt}{1 - \rho_t},
$$

где $\rho_x = \lambda \int_0^x t f(t)\, dt$ — нагрузка от задач размера ? $x$.

Безусловное среднее: $\mathbb{E}[T^{\text{SRPT}}] = \int_0^\infty f(x)\, \mathbb{E}[T^{\text{SRPT}}(x)]\, dx$, откуда $\mathbb{E}[W^{\text{SRPT}}] = \mathbb{E}[T] - \mathbb{E}[B]$.

### 2.2 SJF (non-preemptive SPT) для M/G/1

Формула Conway–Maxwell–Miller (классика, через priority M/G/1):

$$
\mathbb{E}[W^{\text{SJF}}(x)] = \frac{\lambda \mathbb{E}[S^2]}{2 (1 - \rho_{\le x})^2}
$$

(в варианте для непрерывного приоритета по размеру). Безусловное — интегрированием по $f(x)$.

### 2.3 PSJF для M/G/1

$$
\mathbb{E}[T^{\text{PSJF}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\, dt}{2 (1 - \rho_{\le x})^2} + \frac{x}{1 - \rho_{\le x}}.
$$

### 2.4 SPJF (Mitzenmacher 2020) для M/G/1 с предсказаниями

Распределение пары $(X, Y)$ — истинного размера и предсказания — задаётся плотностью $g(x, y)$. Эффективная нагрузка от задач с предсказанием $\le y$:

$$
\rho'_y = \lambda \int_0^y \!\! \int_0^\infty t\, g(t, z)\, dt\, dz.
$$

Условное среднее ожидание:

$$
\mathbb{E}[W^{\text{SPJF}}(y)] = \frac{\lambda \, \mathbb{E}[S^2]}{2 (1 - \rho'_y)^2}.
$$

Безусловное: $\mathbb{E}[W^{\text{SPJF}}] = \int g_Y(y)\, \mathbb{E}[W^{\text{SPJF}}(y)]\, dy$. При идеальных предсказаниях ($Y = X$) переходит в SJF.

### 2.5 Численные требования

Все формулы предполагают известные:
- $\lambda$ (берём напрямую от пользователя, как в `MG1Calc`),
- плотность/CDF $f(x), F(x)$ или хотя бы кусочно-сэмплируемое распределение,
- для SPJF — joint $(X, Y)$ или предсказательная функция $y = h(x) + \text{шум}$.

Реализация — численное интегрирование (`scipy.integrate.quad` / `cumulative_trapezoid`) на сетке. Поддерживаемые «known-form» распределения берём из `most_queue.random.distributions` (`H`, `Gamma`, `E`, `Pa`, `Uniform`, `M`, `D`, `Norm`).

## 3. Что нужно от симуляции

### 3.1 Архитектурный сдвиг: «sample size at arrival»

Сейчас `Server.start_service` сэмплирует время обслуживания только в момент старта. Для size-based дисциплин планировщик **должен видеть размер заявки на момент прихода**. Изменения:

- В `most_queue/sim/utils/tasks.py` ввести/использовать поля `Task.size` (истинный размер, генерируется источником) и `Task.predicted_size` (опционально — предсказание).
- Источник размеров вынести из `Server.start_service` в новый компонент-сэмплер «size at arrival», параметризуемый теми же `kendall_notation/params`, что и сервер.
- Прокинуть «remaining work» через существующий механизм `Task.service_remaining` (он уже есть, см. `Server.start_service`/`preempt_service`).
- Предиктор предсказаний для SPJF задаётся отдельно: либо чистая функция `predict(size, rng) -> y` (например, $Y = X$ для perfect predictions, $Y \sim \mathrm{Exp}(1/X)$ для шумной модели из статьи), либо параметризованный объект.

### 3.2 Новый симулятор `SizeBasedQsSim`

Файл: `most_queue/sim/size_based.py`. Наследуется от `QsSim` (или от `BaseSimulationCore`).

Конструктор:

```python
SizeBasedQsSim(
    num_of_channels: int = 1,
    discipline: Literal["FCFS","SJF","PSJF","SRPT","SPJF","PSPJF","SPRPT"] = "SRPT",
    buffer: int | None = None,
    verbose: bool = True,
)
```

Ключевые отличия от `QsSim`:

- `set_servers(params, kendall_notation)` создаёт **size-сэмплер**, а не Server'ы (распределение времени обслуживания).
- `set_predictor(predictor)` (только для S{P}{P}JF/SPRPT): объект с методом `predict(size, rng) -> float`. По умолчанию — perfect (`y = x`).
- Очередь — приоритетная по выбранной дисциплине: реализуется через `heapq` (новый `most_queue/sim/utils/queue_priority_size.py`), ключ — `(rank, arrival_time, id)`.
- Preemption (для SRPT/PSJF/PSPJF/SPRPT): на каждом приходе проверяем «есть ли в системе заявка с большим текущим рангом, чем у новой?», если да — вытесняем. Прерванная задача с обновлённым `service_remaining` отправляется обратно в очередь.
- При завершении обслуживания — берём из очереди задачу с минимальным рангом.
- Метрики совпадают со стандартным `QsSim`: $w, v, p$ — переиспользуем `refresh_v_stat`/`refresh_w_stat`.

### 3.3 Опциональные побочные метрики

Для верификации против Schrage и публикации результатов полезно сохранить:
- `T_per_size`: словарь `bucket(size) -> mean response time`, чтобы сравнивать с условным $\mathbb{E}[T^{\text{SRPT}}(x)]$.
- `slowdown` распределение ($T/x$) — это стандартный показатель для SRPT.

## 4. Структура изменений в коде

```
most_queue/
??? sim/
?   ??? size_based.py                 # NEW — единый симулятор для FCFS/SJF/PSJF/SRPT/SPJF/...
?   ??? utils/
?       ??? tasks.py                  # MOD — size, predicted_size, service_remaining
?       ??? queue_priority_size.py    # NEW — heap-очередь с динамическими ключами
??? theory/
    ??? srpt/                         # NEW package
        ??? __init__.py
        ??? mg1_srpt.py               # MG1SrptCalc        (Schrage–Miller 1966)
        ??? mg1_sjf.py                # MG1SjfCalc         (non-preemptive size priority)
        ??? mg1_psjf.py               # MG1PsjfCalc        (preemptive shortest job first)
        ??? mg1_spjf.py               # MG1SpjfCalc        (Mitzenmacher 2020)
        ??? utils/
            ??? load_below.py         # ?_{?x}, ?'_y — единая утилита
            ??? predictor.py          # модели предсказаний (perfect, exp-noise, lognormal-noise)
```

Все классы наследуются от `BaseQueue` и возвращают `QueueResults`. `set_sources(l: float)`, `set_servers(b: list[float] | (params, kendall))` — две сигнатуры (по моментам, как у `MG1Calc`, и по «known distribution» для случаев, где нужна полноценная плотность).

## 5. Тесты

Новые файлы в `tests/` (дублируют структуру `test_mg1_calc.py`):

| Файл                          | Что проверяем                                                                                       |
|-------------------------------|------------------------------------------------------------------------------------------------------|
| `tests/test_mg1_srpt.py`      | $\mathbb{E}[W^{\text{SRPT}}], \mathbb{E}[T^{\text{SRPT}}]$ vs `SizeBasedQsSim(discipline="SRPT")`    |
| `tests/test_mg1_sjf.py`       | non-preemptive SPT vs sim                                                                            |
| `tests/test_mg1_psjf.py`      | PSJF vs sim                                                                                          |
| `tests/test_mg1_spjf.py`      | SPJF: (a) perfect predictor ? должно совпасть с SJF; (b) exp-шум $Y \mid X = x \sim \mathrm{Exp}(1/x)$ |
| `tests/test_srpt_vs_fcfs.py`  | санити-чек: $\mathbb{E}[T^{\text{SRPT}}] \le \mathbb{E}[T^{\text{FCFS}}]$ для всех тестируемых $\rho$ |

Параметры берём из `tests/default_params.yaml` (там же — допуски `MOMENTS_RTOL`, `MOMENTS_ATOL`).

Дополнительно — для воспроизведения таблицы из Mitzenmacher–Shahout 2025 (раздел 3.3 в `SPJF.md`) сделать `tests/test_mg1_predictions_table.py`: $S \sim \mathrm{Exp}(1)$, $Y \mid X = x \sim \mathrm{Exp}(1/x)$, $\rho \in \{0.5, 0.8, 0.9, 0.95, 0.99\}$, считаем sim для FCFS/SJF/SPJF/PSJF/PSPJF/SRPT/SPRPT и сверяем с табличными значениями статьи (как regression-точка).

## 6. План работ по этапам

### Этап 1. Подготовка инфраструктуры (без новых дисциплин)

1. Расширить `Task`: поля `size: float | None`, `predicted_size: float | None`, `original_size: float | None` (для SRPT/PSJF, чтобы помнить начальный размер при preempt).
2. Реализовать `most_queue/sim/utils/queue_priority_size.py` (heap + сравнение через ключевую функцию). Покрыть юнит-тестами.
3. Вынести «sample-at-arrival» механику: добавить в `BaseSimulationCore` (или в новый `SizeAwareSimMixin`) хук `_sample_size(task)`, чтобы существующий `QsSim` оставался обратно-совместим.
4. Регрессия: убедиться, что все существующие тесты в `tests/` зелёные.

### Этап 2. Симулятор `SizeBasedQsSim`

1. Реализовать FCFS-режим как sanity-check (должен давать те же числа, что `QsSim` для M/G/1).
2. Добавить SJF (non-preemptive size).
3. Добавить SRPT (preemptive remaining size, требует `Task.service_remaining`).
4. Добавить SPJF (non-preemptive с `predicted_size`) + интерфейс `set_predictor`.
5. Добавить PSJF, PSPJF, SPRPT.
6. Прогнать сравнение FCFS-варианта с результатами `QsSim`. Должны сходиться к 3-4 знакам.

### Этап 3. Теория для M/G/1

1. `MG1SrptCalc`: численная реализация формулы Schrage–Miller через `scipy.integrate.quad`. Поддержка распределений из `most_queue.random.distributions` (нужна `pdf/cdf`).
2. `MG1SjfCalc`, `MG1PsjfCalc` — те же интегралы, но без последнего интегрального члена.
3. `MG1SpjfCalc`: принимает joint $(X, Y)$ — для начала через явную параметризацию (perfect / exp-noise / lognormal-noise), под капотом — те же квадратуры.
4. Утилита `load_below(l, dist, x)` для $\rho_{\le x}$ и $\rho'_y$.

### Этап 4. Тесты sim vs theory

1. `test_mg1_srpt.py`, `test_mg1_sjf.py`, `test_mg1_psjf.py`, `test_mg1_spjf.py` по образцу `test_mg1_calc.py`.
2. `test_srpt_vs_fcfs.py` — sanity inequalities.
3. `test_mg1_predictions_table.py` — repro Mitzenmacher–Shahout таблицы.
4. Подобрать `NUM_OF_JOBS` так, чтобы при $\rho = 0.7, \mathrm{cv} = 1.2$ ошибка $W$ от sim укладывалась в `MOMENTS_RTOL=0.5` (черновая оценка: 200k–500k заявок).

### Этап 5. Документация и примеры

1. `docs/calculation.md`, `docs/simulation.md` — секции «Size-based scheduling».
2. Обновить `docs/models.md`, `docs/concepts.md`: ввести термины SRPT/SJF/SPJF/PSJF/PoM (price of misprediction).
3. Пример в `examples/`: воспроизведение таблицы 3.3 из статьи (sim) с цветным выводом и графиком $\rho \to E[T]$ для каждой дисциплины.
4. Краткий jupyter-tutorial в `tutorials/srpt_basics.ipynb`.

### Этап 6. Расширения (опционально, отдельные подзадачи)

1. **M/G/k SRPT**: bound из Grosof–Scully–Harchol-Balter 2018 (см. `1805.SRPT.pdf`, теорема 5.4) — добавить класс `MGkSrptBoundCalc` + sim для `num_of_channels > 1`.
2. **Динамические предсказания** (problem 5.5 из статьи Mitzenmacher–Shahout): SOAP-rank как функция накопленных наблюдений. Пока — отложено, заводим issue.
3. **Memory-aware SRPT** (problem 5.2): требует bin-packing в симуляторе. Отдельный roadmap.
4. **Limited preemption** (TRAIL) — параметр `max_preemptions` для PSJF/SRPT/SPRPT.

## 7. Критерии готовности (Definition of Done)

Для **каждой** реализованной дисциплины:

1. Класс в `most_queue/theory/srpt/` с тестами на численную устойчивость (минимум 2 распределения: H2, Gamma).
2. Поддержка в `SizeBasedQsSim` с прохождением FCFS-регрессии.
3. Тест `tests/test_mg1_<discipline>.py` зелёный при допусках `default_params.yaml`.
4. Доказана инваринта $E[T^{\text{SRPT}}] \le E[T^{\text{policy}}] \le E[T^{\text{FCFS}}]$ (для M/G/1).
5. README/docs обновлены.

## 8. Риски и open questions

- **Точность интегрирования при $\rho \to 1$**: знаменатель $(1 - \rho_x)^2$ становится почти-сингулярным. Потребуется либо адаптивная сетка, либо логарифмическая замена переменной.
- **CV симуляции для SRPT**: в SRPT длинные задачи имеют тяжёлые хвосты времени отклика ? дисперсия выборочного среднего велика. Нужно либо регрессионное сглаживание (variance reduction), либо явно поднимать `NUM_OF_JOBS` до 1M.
- **Предиктор для SPJF**: API должен быть достаточно гибким, чтобы поддерживать оба сценария — параметрический ($y = h(x) + \text{шум}$) и эмпирический (готовый сэмпл из $g(x, y)$).
- **Совместимость Server-API**: нужно решить — оставлять ли существующий `Server.start_service` нетронутым (создавать отдельный `SizeBasedServer`) или унифицировать. Предлагается: создать отдельный класс, чтобы не трогать стабильную часть.
- **Терминология «predicted size»**: есть риск пересечения с «vacation», «warm-up» и пр. Нужно убедиться, что новые поля Task не конфликтуют с существующими сценариями network/priority.

## 9. Список файлов для изменения

**Новые:**
- `most_queue/sim/size_based.py`
- `most_queue/sim/utils/queue_priority_size.py`
- `most_queue/sim/utils/predictor.py`
- `most_queue/theory/srpt/__init__.py`
- `most_queue/theory/srpt/mg1_srpt.py`
- `most_queue/theory/srpt/mg1_sjf.py`
- `most_queue/theory/srpt/mg1_psjf.py`
- `most_queue/theory/srpt/mg1_spjf.py`
- `most_queue/theory/srpt/utils/load_below.py`
- `tests/test_mg1_srpt.py`
- `tests/test_mg1_sjf.py`
- `tests/test_mg1_psjf.py`
- `tests/test_mg1_spjf.py`
- `tests/test_srpt_vs_fcfs.py`
- `tests/test_mg1_predictions_table.py`
- `examples/srpt_table.py`
- `tutorials/srpt_basics.ipynb`

**Модифицируемые:**
- `most_queue/sim/utils/tasks.py` (добавить `size`, `predicted_size`, `original_size`)
- `most_queue/sim/base_core.py` (хук `_sample_size`, опционально)
- `docs/calculation.md`, `docs/simulation.md`, `docs/concepts.md`, `docs/models.md`
- `docs/README.md` (новая секция со ссылкой на размер-зависимые дисциплины)
- `README.md` (добавить SRPT/SPJF в список features)

## 10. Оценка трудозатрат (грубо)

| Этап                         | Сложность | Срок (чел.-дней) |
|------------------------------|-----------|------------------|
| 1. Инфраструктура            | низкая    | 1–2              |
| 2. `SizeBasedQsSim`          | средняя   | 3–5              |
| 3. Теория M/G/1              | средняя   | 2–4              |
| 4. Тесты sim vs theory       | низкая    | 1–2              |
| 5. Документация и примеры    | низкая    | 1                |
| **Итого до релиза**          |           | **8–14**         |
| 6. Расширения (M/G/k и т.д.) | высокая   | +5–10            |

---

**Следующий шаг:** утвердить порядок дисциплин (предлагается: FCFS-регрессия ? SRPT ? SJF ? SPJF ? PSJF ? PSPJF/SPRPT) и начать с Этапа 1 (расширение `Task` + `queue_priority_size.py`).
