# Roadmap: SLA/deadline-violation probability слой в `most_queue`

> Источники-первоисточники:
> - QUARTZ: Quantile-Aware Routing and Queueing for TTFT SLOs in LLM Serving, ACL Findings 2026.
> - Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling, ACM TACO 2026,
>   doi:10.1145/3803390.
> - A Tail Latency SLO Guaranteed Task Scheduling Scheme for User-Facing Services, IEEE TPDS 2025,
>   doi:10.1109/tpds.2025.3542638.
> - Real-time queues in heavy traffic with earliest-deadline-first queue discipline, Ann. Appl.
>   Prob. 2001, doi:10.1214/aoap/1015345295.
> - Обзор и полный список источников — `docs/research/sla-deadline-queueing-2026.md`.

## 1. Цель

Добавить **горизонтальный** слой поверх уже существующего каталога моментных калькуляторов:
вероятность нарушения дедлайна `P(W > D)` (или `P(T > D)` для времени пребывания) и обратную
операцию — SLO-квантиль `D_p` такой, что `P(W > D_p) = p`. Слой не заменяет и не дублирует ни
один существующий калькулятор — он принимает уже посчитанные raw-моменты и достраивает хвостовую
вероятность через fit-распределение.

Продемонстрировать слой на композитном примере: **LLM-inference serving TTFT SLO** — очередь
запросов на GPU-инференс с батч-обслуживанием и (опционально) приоритетными SLO-уровнями.

Ключевое отличие от `srpt_spjf_roadmap.md`: там добавлялась новая дисциплина обслуживания (новая
теория + новый симулятор с нуля). Здесь новой очереди/дисциплины нет — это **обвязка** поверх
существующих `QueueResults`/`MulticlassResults`, аналогично по духу EPIC-013
(`prediction_degradation_curve` поверх существующего SPJF).

## 2. Что нужно от теории

### 2.1 Fit «моменты → распределение» (уже есть, нужно собрать в публичный API)

`most_queue.random.utils.fit` уже содержит всё нужное:

- `fit_h2(moments)` / `fit_h2_clx(moments, fitting_params)` — H2 по 2-3 моментам. Условие
  применимости: `cv = sqrt(m2 - m1^2) / m1 >= 1` (иначе `fit_h2` возвращает вырожденный
  `H2Params(p1=0, mu1=0, mu2=0)` — нужно ловить и не использовать).
- `fit_gamma(moments)` — Gamma по 2-3 моментам, работает для **любого** `cv > 0` (в т.ч. `cv < 1`,
  близкие к детерминированным распределения).
- `fit_erlang(moments)` — Erlang по 2 моментам, целый `r`, `cv <= 1`. Менее гибкий, чем Gamma
  (дискретный `r`), но точнее для истинно эрланговских хвостов (используется как опция, не
  default).

Новая функция `theory/utils/sla.py::fit_from_moments`:

```python
def fit_from_moments(moments: list[float], family: Literal["auto", "h2", "gamma", "erlang"] = "auto"):
    if family == "auto":
        cv = math.sqrt(moments[1] - moments[0] ** 2) / moments[0]
        family = "h2" if cv >= 1.0 else "gamma"
    if family == "h2":
        return fit_h2_clx(moments) if len(moments) >= 3 else fit_h2(moments)
    if family == "gamma":
        return fit_gamma(moments)
    if family == "erlang":
        return fit_erlang(moments)
    raise ValueError(f"unknown family: {family}")
```

Возвращаемый объект — один из `H2Params`/`GammaParams`/`ErlangParams` (уже определены в
`most_queue.random.utils.params`); дальше на нём вызывается `get_tail`/`get_cdf` из
`most_queue.random.distributions`.

### 2.2 Хвостовая вероятность и квантиль

`theory/utils/sla.py`:

```python
def deadline_violation_prob(moments: list[float], deadline: float, family="auto") -> float:
    params, dist_class = fit_from_moments(moments, family)
    if hasattr(dist_class, "get_tail"):
        return dist_class.get_tail(params, deadline)
    return 1.0 - dist_class.get_cdf(params, deadline)  # Gamma: нет get_tail, есть get_cdf


def slo_quantile(moments: list[float], p: float, family="auto") -> float:
    params, dist_class = fit_from_moments(moments, family)
    cdf_fn = lambda t: dist_class.get_cdf(params, t)
    return upper_bound(cdf_fn, p=p)  # из theory/utils/tail.py, см. §2.3
```

Точная сверка без fit (не аппроксимация) — для регрессионного теста: M/M/1,
`P(W > t) = rho * exp(-mu * (1 - rho) * t)` (Pollaczek–Khinchine для экспоненциального
обслуживания вырождается в чистую экспоненту хвоста ожидания). Это единственная модель в первой
итерации, где хвост считается в закрытой форме, а не через fit — она же служит эталоном точности
fit-подхода (H2/Gamma с `cv = 1` должны давать `mu1 = mu2` → выродиться в ту же экспоненту).

### 2.3 Общая квантильная бисекция (убрать дублирование с SRPT)

`most_queue/theory/srpt/utils/load_below.py::upper_bound` уже делает ровно то, что нужно для
`slo_quantile` — бисекция «найти x: `1 - CDF(x) < p`». Переносим в
`most_queue/theory/utils/tail.py::upper_bound` (общий, не size-based-специфичный модуль), SRPT
подключает его оттуда. Сигнатура не меняется, поведение не меняется — чистый перенос +
re-export для обратной совместимости импортов в SRPT-модуле.

## 3. Что нужно от симуляции

### 3.1 Онлайн-счётчики превышения дедлайна (без хранения сырых выборок)

`most_queue.sim.base.QsSim` уже накапливает `self.w`/`self.v` через `refresh_w_stat`/
`refresh_v_stat` без хранения списка сэмплов (см. `most_queue/sim/base.py:53-54, 542-564`). По
той же схеме:

- Конструктор `QsSim(..., deadline_thresholds: list[float] | None = None)`.
- `self.deadline_hits: dict[float, int] = {d: 0 for d in deadline_thresholds or []}`,
  `self.deadline_n = 0`.
- В том же месте, где вызывается `refresh_w_stat(tsk.wait_time)` (`most_queue/sim/base.py:173,
  305`), дополнительно: `self.deadline_n += 1; for d in self.deadline_hits: if tsk.wait_time > d:
  self.deadline_hits[d] += 1`.
- Геттер `get_empirical_violation_prob(d) -> float` = `deadline_hits[d] / deadline_n`.

Эта же схема прокидывается в симуляторы, которые наследуют/переиспользуют `QsSim`/
`BaseSimulationCore`: `most_queue/sim/priority.py`, `most_queue/sim/bulk_service.py`. Для
`priority.py` — счётчики по классам отдельно (аналогично тому, как там уже разделены
`w`/`v` по классам).

### 3.2 Не строим отдельный «SLA-симулятор»

Явно НЕ создаём новый файл симулятора — это была бы преждевременная абстракция: дедлайн-счётчики это тонкая добавка
к существующим `QsSim`/`PriorityQueueSimulator`/bulk-service симулятору, а не новая модель
поведения системы.

## 4. Структура изменений в коде

```
most_queue/
├── theory/
│   ├── utils/
│   │   ├── tail.py                 # NEW — upper_bound (перенесено из srpt/utils/load_below.py)
│   │   └── sla.py                  # NEW — fit_from_moments, deadline_violation_prob, slo_quantile
│   └── srpt/utils/load_below.py    # MOD — upper_bound теперь re-export из theory/utils/tail.py
├── sim/
│   ├── base.py                     # MOD — deadline_thresholds, deadline_hits, refresh-хук
│   ├── priority.py                 # MOD — то же, по классам
│   └── bulk_service.py             # MOD — то же
├── examples/
│   └── llm_serving_slo.py          # NEW — композитный пример
├── tutorials/
│   └── llm_serving_slo.ipynb       # NEW
└── docs/models/
    ├── sla.md                      # NEW
    └── sla.ru.md                   # NEW
```

## 5. Тесты

| Файл | Что проверяем |
|---|---|
| `tests/test_sla_tail.py` | `deadline_violation_prob`/`slo_quantile` на синтетических моментах: M/M/1 точная экспонента (regression, atol 1e-6); монотонность по `deadline`; round-trip `deadline_violation_prob(m, slo_quantile(m, p)) ≈ p` для H2 и Gamma семейств; вырожденный случай `cv < 1` → family="h2" должен явно упасть с понятной ошибкой, а не тихо дать мусор. |
| `tests/test_sla_vs_sim_mg1.py` | `MG1Calc` моменты → fit-хвост vs `QsSim(deadline_thresholds=[...])` эмпирическая частота, несколько `rho`/`cv`. |
| `tests/test_sla_vs_sim_mgn.py` | То же для `MGnCalc` (M/G/n, Такахаси). |
| `tests/test_sla_vs_sim_map_ph1.py` | То же для `MapPh1Calc` (бёрсти MAP-вход — самый релевантный для LLM-serving случай). |
| `tests/test_sla_vs_sim_priority.py` | То же для `MG1NonPreemptiveCalc`, по каждому классу отдельно (высокий/низкий приоритет — разные хвосты). |
| `tests/test_sla_vs_sim_bulk.py` | То же для `BulkServiceMM1Calc` (батч-инференс). |

Допуски — как у остальных тестов репозитория, `tests/default_params.yaml` (`MOMENTS_RTOL`);
fit-хвост — приближение по 2-3 моментам, поэтому сравнение с DES, а не «точное = точное» (кроме
M/M/1, где сравниваем с закрытой формой).

## 6. План работ по этапам

### Этап 1. Ядро `theory/utils` (без интеграции в sim, без примера)

1. Перенести `upper_bound` в `theory/utils/tail.py`, обновить импорт в
   `theory/srpt/utils/load_below.py` (re-export, регрессия существующих SRPT-тестов должна остаться зелёной).
2. `theory/utils/sla.py`: `fit_from_moments`, `deadline_violation_prob`, `slo_quantile`.
3. Точная формула M/M/1 — как отдельная маленькая функция-эталон (не через fit), используется
   только в тесте для проверки точности fit-подхода.
4. `tests/test_sla_tail.py`.

### Этап 2. Sim-сторона: онлайн-счётчики

1. `deadline_thresholds`/`deadline_hits` в `QsSim` (`most_queue/sim/base.py`).
2. То же в `PriorityQueueSimulator` (`most_queue/sim/priority.py`), по классам.
3. То же в bulk-service симуляторе (`most_queue/sim/bulk_service.py`).
4. Регрессия: существующие тесты sim не ломаются при `deadline_thresholds=None` (дефолт).

### Этап 3. Кросс-валидация поверх каталога моделей

1. `test_sla_vs_sim_mg1.py`, `test_sla_vs_sim_mgn.py` — базовые случаи.
2. `test_sla_vs_sim_map_ph1.py` — MAP/PH-вход (бёрсти арривалы, наиболее релевантно для
   LLM-serving трафика).
3. `test_sla_vs_sim_priority.py` — приоритетные SLO-уровни.
4. `test_sla_vs_sim_bulk.py` — батч-обслуживание.
5. Если fit-хвост систематически расходится с DES при высоких `rho` (ожидаемо — hеavy-traffic
   искажает 2-3-моментный fit) — задокументировать диапазон применимости в докстринге
   `deadline_violation_prob`, не пытаться «починить» точность сверх 3 моментов в этой итерации.

### Этап 4. Композитный пример LLM-serving TTFT SLO

1. `examples/llm_serving_slo.py`: `MapPh1Calc` (MAP-вход) + `BulkServiceMM1Calc`-подобная
   батч-логика (по образцу `bulk_service.py`) + опционально 2 приоритетных класса через
   `MG1NonPreemptiveCalc`/`map_ph_priority.py`. Строит кривую `P(TTFT > D)` от `rho`.
2. `tutorials/llm_serving_slo.ipynb` — то же с графиками (см. skill `dataviz` для палитры/осей),
   наложение sim-точек на аналитическую кривую.

### Этап 5. Документация

1. `docs/models/sla.md` + `.ru.md` — новая страница каталога.
2. Кросс-ссылки из `docs/models/priority.md`, `batch.md`, `map-ph.md`.
3. `README.md`/`README.ru.md` — пункт в списке возможностей.
4. `docs/epics/README.md` — EPIC-021 → done.

## 7. Критерии готовности (DoD)

1. M/M/1: `deadline_violation_prob` совпадает с точной формулой `rho * exp(-mu*(1-rho)*t)`,
   ошибка < 1e-6.
2. Для M/G/1, M/G/n, MAP/PH/1, приоритетного M/G/1 — fit-хвост в пределах `MOMENTS_RTOL` от
   эмпирической частоты DES при умеренной нагрузке (`rho <= 0.8`, как у остальных тестов
   репозитория).
3. `slo_quantile` и `deadline_violation_prob` взаимно согласованы (round-trip).
4. Композитный пример запускается и даёт содержательный график.
5. Документация EN+RU на месте, кросс-ссылки расставлены.

## 8. Риски и open questions

- **Точность 2-3-моментного fit в тяжёлом хвосте (`t` далеко в хвосте или `rho → 1`)**: H2/Gamma
  по первым моментам плохо восстанавливают экстремальные квантили (99.9%+). Документируем как
  известное ограничение первой итерации; более точные методы (например, подгонка по большему
  числу моментов, если калькулятор их отдаёт) — кандидат на будущее расширение, не блокирует
  DoD.
- **`cv < 1` и семейство `"h2"` явно запрошено пользователем**: `fit_h2` в этом случае возвращает
  вырожденный `H2Params(p1=0, mu1=0, mu2=0)` — нужно либо кидать `ValueError` с понятным
  сообщением, либо молча падать обратно на Gamma (решение — выбрать в Этапе 1, склоняемся к
  явной ошибке, чтобы не скрывать некорректный выбор семейства от пользователя).
  - **Выбор:** явный `ValueError`.
- **Наследование `deadline_thresholds` в производных симуляторах**: `PriorityQueueSimulator` и
  bulk-service симулятор не обязательно совпадают по внутренней структуре с `QsSim` в месте
  вызова `refresh_w_stat` — нужно проверить каждый файл индивидуально при реализации Этапа 2, а
  не полагаться на единый миксин.
- **MAP/PH/1 и предсказуемость TTFT**: для LLM-serving самое интересное — коррелированный
  (бёрсти) вход через MAP, а не Poisson. Это уже поддерживается `MapPh1Calc`/DES для MAP, но
  нужно явно проверить, что `deadline_thresholds` корректно прокидывается через MAP-специфичный
  путь симуляции (если он отличается от `QsSim`).

## 9. Список файлов для изменения

**Новые:**
- `most_queue/theory/utils/tail.py`
- `most_queue/theory/utils/sla.py`
- `tests/test_sla_tail.py`
- `tests/test_sla_vs_sim_mg1.py`
- `tests/test_sla_vs_sim_mgn.py`
- `tests/test_sla_vs_sim_map_ph1.py`
- `tests/test_sla_vs_sim_priority.py`
- `tests/test_sla_vs_sim_bulk.py`
- `examples/llm_serving_slo.py`
- `tutorials/llm_serving_slo.ipynb`
- `docs/models/sla.md`, `docs/models/sla.ru.md`

**Модифицируемые:**
- `most_queue/theory/srpt/utils/load_below.py` (re-export `upper_bound`)
- `most_queue/sim/base.py` (`deadline_thresholds`, `deadline_hits`)
- `most_queue/sim/priority.py` (то же, по классам)
- `most_queue/sim/bulk_service.py` (то же)
- `docs/models/priority.md`, `batch.md`, `map-ph.md` (кросс-ссылки)
- `README.md`, `README.ru.md`
- `docs/epics/README.md`

## 10. Оценка трудозатрат (грубо)

| Этап | Сложность | Срок (чел.-дней) |
|---|---|---|
| 1. Ядро `theory/utils` | низкая | 1–2 |
| 2. Sim-сторона (онлайн-счётчики) | низкая | 1–2 |
| 3. Кросс-валидация по каталогу | средняя | 2–3 |
| 4. Композитный LLM-serving пример | средняя | 2–3 |
| 5. Документация | низкая | 1 |
| **Итого** | | **7–11** |

---

## 11. Реализация (2026-09-29): итоги и отклонения от плана

Все 5 этапов выполнены. Два отклонения от изначального плана, оба обнаружены по ходу реализации:

1. **Bulk-service вне охвата.** `BulkServiceMM1Calc`/`BulkServiceSim` (§2.1, §3.1 плана) считают
   только среднее время ожидания (`w=[e_w, 0, 0, 0]`) — второго момента нет и не считается, поэтому
   SLA-слою (минимум 2 момента) там применять нечего. `test_sla_vs_sim_bulk.py` не написан;
   деadline-счётчики в `BulkServiceSim` не добавлены. Композитный пример LLM-serving использует
   PH-подгонку времени батч-инференса напрямую (через `MapPh1Calc`) вместо `BulkServiceMM1Calc`.
   Резерв на будущее: расширить `BulkServiceMM1Calc` до E[W²] из стационарного распределения CTMC.
2. **`MapPh1PriorityCalc` тоже mean-only.** Обнаружено при попытке собрать секцию с приоритетными
   SLO-классами в композитном примере через MAP-вход: калькулятор считает `w = [[mean0],[mean1]]`
   (E[N]/λ по CTMC), без старших моментов. Секция переделана на Poisson-сплит через
   `MG1NonPreemptiveCalc` (полные моменты по классам) — тот же вывод («приоритет защищает премиум
   при равной загрузке сервера»), другой вход (без бёрстинга).
3. **Найден краевой баг `fit_h2`** (Aliev's method, `most_queue/random/utils/fit.py`): когда третий
   момент лежит на границе H2-достижимости (`t_min` близко к `moments[2]`, включая чуть ниже),
   «one phase distribution»-ветка возвращала параметры, чьё восстановленное среднее расходится с
   целевым на порядки (воспроизведено дважды независимо: на моментах ожидания приоритетных классов
   M/G/1 NP и в композитном примере). На момент реализации EPIC-021 закрыт защитной проверкой
   внутри `theory/utils/sla.py::fit_from_moments` (сверка восстановленного среднего, откат на
   Gamma для `family="auto"`, явная ошибка для `family="h2"`), сам `fit_h2` не менялся.

**Корневой фикс `fit_h2` (2026-09-29, отдельная задача после EPIC-021).** Причина бага —
перепутанные переменные в буквенном виде ветки: код считал значение по формуле `t2` (у которой на
границе `q=q_max` аналитически `t2=0`), но присваивал его результату `mu1` (после инвертирования),
а `mu2` жёстко хардкодил в `1e6` — то есть обе величины были не на своих местах, а настоящая формула
`t1` (конечная, «медленная» фаза) вообще не считалась. Плюс отдельный баг: guard
`math.isclose(mu1, 0)` с `abs_tol=0` по умолчанию почти никогда не срабатывает для маленького
ненулевого float, поэтому деление на почти-ноль давало гигантские (но не бесконечные) ставки вместо
корректного распознавания «эта фаза вырождается в мгновенную».

Исправлено: ветка теперь считает `t1`/`t2` по тем же формулам, что и основной bisection-цикл (для
консистентности), корректно кладёт конечную фазу в `mu1`, а вырождающуюся — в `mu2` (со
масштабо-относительным sentinel `1e10/mean` вместо константы), с защитой от sqrt отрицательного
аргумента из-за fp-шума ровно на границе. Регрессионные тесты —
`tests/units/test_fit_h2.py` (интерьерные случаи, граничные случаи — сверка среднего/дисперсии
точно, третьего момента — с клэмпом к достижимому минимуму, невалидные `cv < 1`). Защитная проверка
в `sla.py` оставлена как defense-in-depth (теперь не должна срабатывать на легитимных входах).

**Следующий шаг:** нет — эпик и последующий фикс `fit_h2` реализованы; см.
`docs/epics/EPIC-021-slo-deadline-queueing.md` раздел «Результаты».
