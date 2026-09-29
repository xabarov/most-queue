# EPIC-021: SLA/deadline-violation probability (SLO-калькуляторы, LLM-serving TTFT)

- **Статус:** done (2026-09-29)
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/slo_deadline_roadmap.md](../roadmaps/slo_deadline_roadmap.md)

## Цель

Добавить в библиотеку горизонтальный слой **вероятность/квантиль нарушения дедлайна** поверх уже
существующих калькуляторов (любой калькулятор, возвращающий моменты ожидания/пребывания —
`QueueResults`/`MulticlassResults`/`PriorityResults`), и продемонстрировать его на композитном
примере LLM-inference serving (TTFT SLO), который сшивает уже готовые куски: MAP/PH-вход +
bulk-service (батч-инференс) + приоритетные SLO-уровни.

## Контекст

Обзор литературы: [../research/sla-deadline-queueing-2026.md](../research/sla-deadline-queueing-2026.md).
Все 20 предыдущих эпиков закрыты; это первое новое направление после итогового обзора трендов
2026-07. Самый горячий прикладной фронт 2025-2026 — SLO/deadline-aware LLM inference serving
(QUARTZ, Hermes, TailGuard/TPDS 2025) — математически сводится к вероятности/квантилю хвоста
времени ожидания, а не к RL/ML, поэтому хорошо ложится в нишу «точные моменты + DES-кросс-
валидация».

Ключевое наблюдение: **почти вся нужная инфраструктура уже есть**, но не собрана в публичный
слой:
- `most_queue.random.utils.fit`: `fit_h2`/`fit_h2_clx` (cv ≥ 1), `fit_gamma` (любой cv > 0),
  `fit_erlang` (cv ≤ 1) — моменты → параметры распределения.
- `most_queue.random.distributions.*Distribution.get_tail/get_cdf` — параметры → P(X > t).
- `most_queue.theory.srpt.utils.load_below.upper_bound` — бисекционный поиск квантиля по CDF,
  сейчас privately привязан к SRPT-модулю.
- `most_queue.sim.base.QsSim.refresh_w_stat/refresh_v_stat` — паттерн онлайн-накопления моментов
  без хранения сырых выборок — годная база для онлайн-счётчиков превышения дедлайна.

Эталон эпика с похожей структурой (обвязка поверх готовых расчётов, а не новая теория с нуля) —
[EPIC-013](EPIC-013-predictions-scheduling.md).

## Задачи

### Ядро: `theory/utils` — универсальный слой (не привязан к конкретной модели)

- [x] Вынести `upper_bound` из `theory/srpt/utils/load_below.py` в `theory/utils/tail.py` как
      переиспользуемую квантильную бисекцию (SRPT-модуль переходит на общую версию — убрать
      дублирование).
- [x] `theory/utils/sla.py`:
  - `fit_from_moments(moments: list[float], family: Literal["auto","h2","gamma","erlang"]="auto")`
    — авто-выбор по cv: cv ≥ 1 → H2 (`fit_h2`/`fit_h2_clx` по числу переданных моментов),
    cv < 1 → Gamma (`fit_gamma`, работает для любого cv > 0, включая близкие к детерминированным).
  - `deadline_violation_prob(moments, deadline, family="auto") -> float` — P(W/T > deadline) через
    fit + `get_tail`/`1 - get_cdf`.
  - `slo_quantile(moments, p, family="auto") -> float` — дедлайн `D_p`, при котором
    `P(W > D_p) = p` (через `theory/utils/tail.py`).
  - Точные частные случаи не через fit, а напрямую (без аппроксимации): M/M/1 —
    `P(W > t) = rho * exp(-mu*(1-rho)*t)` (замкнутая форма, используется и как regression-тест
    точности fit-подхода).
- [x] Юнит-тесты `tests/test_sla_tail.py`: сходимость fit-хвоста к точному для M/M/1; монотонность
      `deadline_violation_prob` по `deadline`; `slo_quantile` — обратная операция
      (`deadline_violation_prob(moments, slo_quantile(moments, p)) ≈ p`).
- [x] **Обнаружен и закрыт краевой баг `fit_h2`** (Aliev's method): когда третий момент лежит на
      границе H2-достижимости (или чуть ниже), «one phase distribution»-ветка возвращает параметры
      с грубо неверным восстановленным средним (расхождение на порядки — воспроизведено на
      моментах ожидания приоритетных классов M/G/1 NP). `fit_from_moments` теперь сверяет
      восстановленное среднее с целевым и при расхождении > 5% откатывается на Gamma для
      `family="auto"`, либо кидает `RuntimeError` для явного `family="h2"`. Сам `fit_h2` не
      трогали (общий код, используется в других местах) — фикс изолирован в SLA-слое.

### Sim-сторона: кросс-валидация без хранения сырых выборок

- [x] В `most_queue.sim.base_core.BaseSimulationCore` — `set_deadline_thresholds`/
      `_record_deadline_hit`/`get_empirical_violation_prob`; `QsSim.refresh_w_stat`
      (`most_queue/sim/base.py`) вызывает `_record_deadline_hit` при каждом обновлении статистики
      ожидания. Без накопления списка сэмплов, по аналогии с `refresh_w_stat`.
- [x] Прокинуто в приоритетный симулятор (`PriorityQueueSimulator`, `most_queue/sim/priority.py`)
      — по классам отдельно (`deadline_hits: list[dict]`, `get_empirical_violation_prob(k, d)`).
- [~] Bulk-service симулятор (`most_queue/sim/bulk_service.py`) — **не сделано, осознанно**:
      `BulkServiceMM1Calc`/`BulkServiceSim` считают только среднее (`w=[e_w,0,0,0]`), без второго
      момента — SLA-слою (нужно ≥2 момента) там применять пока нечего. Отдельная задача на будущее:
      расширить `BulkServiceMM1Calc` до вычисления E[W²] из стационарного распределения CTMC, тогда
      деadline-счётчики можно добавить тем же паттерном.

### Применение к каталогу моделей + кросс-валидация

- [x] Тесты `slo_quantile`/`deadline_violation_prob` поверх реальных калькуляторов:
      `tests/test_sla_vs_sim_mg1.py` (`MG1Calc`), `tests/test_sla_vs_sim_mgn.py` (`MGnCalc`),
      `tests/test_sla_vs_sim_map_ph1.py` (`MapPh1Calc`, MMPP-2 бёрсти вход),
      `tests/test_sla_vs_sim_priority.py` (`MG1NonPreemptiveCalc`, по каждому классу). Bulk-service
      кросс-валидация опущена — см. пункт выше. Допуск `atol=0.03` (0.05 для приоритетного теста —
      выше cv из-за residual-busy-period структуры, см. докстринг теста).

### Композитный пример: LLM-serving TTFT SLO

- [x] `examples/llm_serving_slo.py`: MAP(MMPP-2)/PH/1 (бёрсти запросы, `MapPh1Calc`) с
      PH-подогнанным временем батч-инференса. Секция с bulk-service (`bulk_service.py`) не вошла —
      та же причина (только среднее, нет момента для SLA); вместо неё вариативность обслуживания
      представлена через PH-подгонку сервисного времени напрямую. Второй раздел — 2 SLO-класса
      (premium/free) через `MG1NonPreemptiveCalc` (Poisson-сплит; `MapPh1PriorityCalc` тоже
      оказался mean-only, см. докстринг примера). Кривая P(TTFT > D) от ρ воспроизводит ожидаемую
      форму (резкий рост при приближении к границе устойчивости: 0.24 → 0.89 на ρ∈[0.5, 0.95]).
- [x] `tutorials/llm_serving_slo.ipynb` — то же в ноутбуке, выполнен (`jupyter nbconvert --execute`)
      и сохранён с выводом.

### Документация

- [x] `docs/models/sla.md` + `docs/models/sla.ru.md` — новая страница каталога моделей. Кросс-ссылки
      из `priority.md`, `batch.md`, `map-ph.md` (EN+RU).
- [x] `docs/models.md`/`docs/models.ru.md` — строка в таблице «Model families».
- [x] `README.md`/`README.ru.md` — строка в таблице возможностей.
- [x] `docs/epics/README.md` — реестр обновлён (EPIC-021 добавлен).

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. `deadline_violation_prob` для M/M/1 воспроизводит точную формулу `rho * exp(-mu*(1-rho)*t)` с
   ошибкой < 1e-6 (не аппроксимация — прямая проверка правильности слоя).
2. Для минимум 4 моделей каталога (M/G/1, M/G/n, MAP/PH/1, приоритетный M/G/1) fit-хвост совпадает
   с эмпирической частотой превышения дедлайна из DES в пределах допуска (`atol=0.03`, `0.05` для
   приоритетного случая). Bulk-service — вне охвата (калькулятор считает только среднее).
3. `slo_quantile` и `deadline_violation_prob` — взаимно согласованы (round-trip тест).
4. Композитный LLM-serving пример запускается, даёт содержательный график (вероятность нарушения
   SLO растёт с ростом ρ, стремится к 1 при ρ → 1).
5. Документация (EN+RU) на месте, кросс-ссылки расставлены.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/slo_deadline_roadmap.md), раздел 11 — итоги и
отклонения). Кратко:

- **Ядро:** `theory/utils/tail.py` (перенесённая `upper_bound`) + `theory/utils/sla.py`
  (`fit_from_moments`, `deadline_violation_prob`, `slo_quantile`, `mm1_deadline_violation_prob`).
  По ходу найден и закрыт краевой баг `fit_h2` (защитная проверка внутри SLA-слоя, сам `fit_h2` не
  трогали).
- **Sim:** онлайн-счётчики превышения дедлайна в `BaseSimulationCore`/`QsSim`/
  `PriorityQueueSimulator` (без хранения сырых выборок).
- **Кросс-валидация:** 4 новых тестовых файла (M/G/1, M/G/n, MAP/PH/1 с MMPP-2 входом,
  приоритетный M/G/1 NP) — fit-хвост сходится с эмпирической частотой DES. Bulk-service вне
  охвата (калькулятор считает только среднее — задокументировано как резерв на будущее).
- **Композитный пример:** `examples/llm_serving_slo.py` + исполненный `tutorials/
  llm_serving_slo.ipynb` — MAP(MMPP-2)/PH/1 LLM-serving TTFT SLO, кривая P(TTFT>D) от ρ (0.24→0.89
  на ρ∈[0.5,0.95]) + секция с двумя SLO-классами (premium/free).
- **Документация:** `docs/models/sla.md`+`.ru.md`, записи в `docs/models.md`/`.ru.md`,
  README/README.ru.md, кросс-ссылки из priority/batch/map-ph каталога.
- **Тесты:** все новые тесты зелёные (`test_sla_tail.py`, 4×`test_sla_vs_sim_*.py`), формат черным
  (`black --check`) чист. Полный прогон `tests/` — **472 passed, 0 failed** (867s), регрессий нет;
  16 предупреждений — все pre-existing (numpy `matrix`-deprecation, известная дивергенция в
  robustness-тесте Такахаси, известный divergent-integral в prediction-degradation), не связаны с
  изменениями этого эпика.
- **Постфактум (2026-09-29):** корневая причина найденного краевого бага `fit_h2` закрыта в самом
  `most_queue/random/utils/fit.py` (не только в защитной обвязке SLA-слоя) — см.
  `docs/roadmaps/slo_deadline_roadmap.md` §11 для разбора причины и фикса. Новый регрессионный тест
  `tests/units/test_fit_h2.py`.
