# EPIC-001: Исследование — модели теории очередей vs реализованные в Most-Queue

- **Статус:** done (2026-07-08); открытым остаётся бэклог рефакторинга (см. ниже)
- **Создан:** 2026-07-08
- **Roadmap:** будет создан по итогам (задача 4)

## Цель

Систематически сопоставить ландшафт моделей теории массового обслуживания (от базовых до
продвинутых) с тем, что уже реализовано в Most-Queue; получить приоритизированный список
кандидатов на реализацию с оценкой ценности и сложности. Итог — документ
`docs/models_gap_analysis.md` и roadmap(ы) на следующие эпики.

## Контекст: что уже реализовано (инвентаризация от 2026-07-08)

Подробный каталог с примерами — [../models.md](../models.md). Сводка по категориям:

**FIFO (theory/fifo):** M/M/n, M/M/n/r; M/G/1; GI/M/1; GI/M/n; M/D/n; E_k/D/n;
M/H2/n (Такахаси-Таками, complex-fit при CV<1); H2/M/n; H2/H2/n.

**Size-based дисциплины (theory/srpt):** M/G/1 SRPT (Schrage–Miller), SJF, PSJF,
SPJF с предикторами (Mitzenmacher).

**Приоритеты (theory/priority):** M/G/1 PR и NP (мультикласс); M/G/n PR/NP
(инвариантная аппроксимация); M/Ph/n PR (busy-period approx); M/M/2 3 класса,
M/M/n 2 класса PR (busy-period approx).

**Отпуска/прогрев (theory/vacations):** M/G/1 warm-up; M/H2/n + H2-прогрев;
M/M/n + H2-cold/H2-warm; M/Ph/n + прогрев/охлаждение/задержка охлаждения.

**Отрицательные заявки (theory/negative):** M/G/1 и M/G/n в вариантах RCS и disaster.

**Сети (theory/networks):** открытые сети декомпозицией (узлы — Такахаси-Таками),
с приоритетами, с отрицательными заявками; оптимизация матрицы маршрутизации.

**Прочее:** Fork-Join M/M/n (n,k) и Split-Join M/G/n; M^X/M/1 (batch arrivals);
M/M/1 с нетерпением; Engset (конечный источник).

**Симуляция (sim):** GI/G/n/r базовый движок + специализированные симуляторы для всех
категорий выше (приоритеты, batch, finite source, fork-join, negative, vacations,
size-based, суммирование потоков, сети — обычные/приоритетные/negative).

Методическое ядро: Такахаси-Таками (мультисерверные фазовые модели и всё, что наследует
`MGnCalc`), вложенные цепи Маркова (GI/M/·), метод моментов + busy-period LST (M/G/1-семейство),
декомпозиция сетей.

## Предварительный gap-список (верифицирован в задаче 2; итог — в [models_gap_analysis.md](../models_gap_analysis.md))

*Базовые пробелы:*
- M/M/∞ и M/G/∞ (бесконечное число приборов) — нет.
- Явные формулы Эрланга B (M/M/n/0, потери) и Erlang C как самостоятельные калькуляторы — частично покрыто M/M/n/r, но без insensitivity-варианта M/G/n/0.
- G/G/1 / GI/G/n аппроксимации (Kingman, Allen–Cunneen, Krämer–Langenbach-Belz, Whitt QNA) — симулятор есть, теории нет.
- M/G/1 Processor Sharing (PS) и LCFS-PR — нет (при том что size-based семейство уже развито).

*Средний уровень:*
- Erlang-A (M/M/n+M, нетерпение в многоканальной) — нетерпение только для M/M/1.
- Пакетное обслуживание M/M^Y/1 и общий M^X/G/1 — batch только на входе и только M^X/M/1.
- Retrial queues (M/M/1, M/G/1 с повторными вызовами) — нет.
- Классические vacation-модели M/G/1 (multiple/single vacations, декомпозиция Fuhrmann–Cooper) — есть только warm-up/cooling варианты.
- Гетерогенные приборы (разные μ по каналам) — нет.
- Дисциплины FB/LAS, Round-Robin — нет.

*Продвинутый уровень:*
- Матрично-аналитические методы: QBD, matrix-geometric (Neuts), MAP/PH/1, BMAP/G/1, PH/PH/n — нет (Такахаси-Таками покрывает часть ниши, но MAP-входа нет вообще).
- Коррелированные входные потоки (MAP/MMPP) — нет.
- Закрытые сети: Gordon–Newell, MVA, BCMP — нет (только Engset как закрытая СМО).
- Polling-системы — нет.
- Дискретные по времени очереди (Geo/Geo/1 и др.) — нет.
- Heavy-traffic / диффузионные аппроксимации, нестационарные (M_t) модели — нет.
- Надёжность приборов (breakdowns & repairs) как самостоятельная модель — нет (частично имитируется negative RCS).

## Задачи

- [x] 1. Инвентаризация зафиксирована (2026-07-08): все import-строки `docs/models.md`
      проверены исполнением; исправлено 11 неверных имён классов; добавлены отсутствовавшие
      в каталоге модели (MM2BusyApprox3Classes, MMnPR2ClsBusyApprox, MMnHyperExpWarmAndCold,
      MG1Disasters). Выявленные проблемы нейминга — в бэклоге рефакторинга ниже.
- [x] 2. Обзор ландшафта моделей ТМО по литературе и материалам `works/queueing_systems_review/`;
      gap-список верифицирован и дополнен (2026-07-08). Дополнительно проведён обзор
      экосистемы open-source инструментов (Ciw, BuTools, LINE, JMT, R queueing и др.)
      для оценки дифференциации.
- [x] 3. Оценка кандидатов (ценность/сложность/реюз ядра/дифференциация) оформлена
      в [../models_gap_analysis.md](../models_gap_analysis.md) с приоритизацией
      must/should/could/defer (2026-07-08).
- [x] 4. Заведены roadmaps и эпики (2026-07-08): [EPIC-002](EPIC-002-wave1-exact-models.md)
      (волна «must», [roadmap](../roadmaps/wave1_exact_models_roadmap.md)),
      [EPIC-003](EPIC-003-qbd-map-ph.md) (QBD/MAP/PH,
      [roadmap](../roadmaps/qbd_map_ph_roadmap.md)),
      [EPIC-004](EPIC-004-retrial-erlang-a.md) (retrial + Erlang-A,
      [roadmap](../roadmaps/retrial_erlang_a_roadmap.md)).

## Бэклог рефакторинга (выявлено при сверке каталога, задача 1)

Сверка `docs/models.md` с кодом (2026-07-08) выявила 11 расхождений в именах классов —
документация исправлена по факту кода. Первопричина — непоследовательный нейминг в
`most_queue.theory`; предложения (каждое — отдельная задача, делать с deprecation-алиасами
и фиксацией в changelog, т.к. ломает публичный API):

- [ ] Унифицировать суффикс `Calc`: сейчас `MG1Calc`, `MMnrCalc`, `HkHkNCalc`, но `Engset`,
      `MDn`, `EkDn`, `GiMn`, `MPhNPrty`, `MH2nH2Warm`. Предложение: все калькуляторы —
      с суффиксом `Calc` (`EngsetCalc`, `MDnCalc`, ...), старые имена — deprecated-алиасы.
- [ ] Единый порядок слов в negative-классах: `MG1NegativeCalcRCS` (Calc в середине)
      vs `MGnNegativeRCSCalc` (Calc в конце) → привести к `MG1NegativeRCSCalc`.
- [ ] Единый стиль имён vacation-классов: `MH2nH2Warm`, `MMnHyperExpWarmAndCold`,
      `MGnH2ServingColdWarmDelay` — три разных схемы для одного семейства.
- [ ] Единый стиль аббревиатур приоритетов: `MPhNPrty` (Prty) vs `MMnPR2ClsBusyApprox` (PR, Cls)
      vs `MGnInvarApproximation` (полное слово).
- [ ] Регистр «GI»: `GIM1Calc` vs `GiMn` → единообразно (`GIM1Calc`, `GIMnCalc`).
- [x] Реэкспортировать публичные классы в `__init__.py` подпакетов theory
      (**выполнено 2026-07-09** для всех 12 подпакетов) — стабильные пути импорта
      `from most_queue.theory.<sub> import <Calc>`, устойчивые к переименованию модулей.
- [x] Тест сверки каталога с кодом `tests/units/test_docs_imports.py` (**2026-07-09**):
      исполняет все 88 `from most_queue...import` строк из `models.md`/`models.ru.md` —
      каталог больше не разойдётся с кодом молча.
- [ ] **Полная унификация суффикса `Calc` — отложено в v3.0** (осознанное решение 2026-07-09).
      16 из 52 классов не оканчиваются на `Calc` (`Engset`, `MDn`, `EkDn`, `GiMn`, `BatchMM1`,
      `MM1Impatience`, `MPhNPrty`, `MG1Disasters`, `MH2nH2Warm`, `MGnH2ServingColdWarmDelay`,
      `MM2BusyApprox3Classes`, `MMnHyperExpWarmAndCold`, `MMnPR2ClsBusyApprox`,
      `MGnInvarApproximation`) + порядок слов (`MG1NegativeCalcRCS`, `OpenNetworkCalcPriorities`)
      + регистр (`GiMn` vs `GIM1Calc`). Ломающее изменение публичного API на 16 классов
      с deprecation-алиасами и правкой обоих каталогов/README — правильнее батчить в мажорный
      релиз с migration guide. Стабильные реэкспорты выше уже снимают основную боль.
- [x] **Баг GIM1Calc** (исправлен 2026-07-09): `get_w` считал ожидание вычитанием
      обслуживания из пребывания как независимых величин (`conv_moments_minus`) —
      среднее случайно верно, старшие моменты нет. Заменено точной замкнутой формой
      w_k = k!·σ/(μ(1−σ))^k; совпадение с QBD до 4 знаков, тест
      `test_phph1_matches_gim1` ужесточён до трёх моментов. `gi_m_n.py` проверен —
      чист (путь через LST корректен, GiMn(n=1) совпадает с QBD).
- [x] **sdist** (исправлено 2026-07-09): `[tool.hatch.build.targets.sdist] include`
      оставляет только `most_queue/`, README и LICENSE — с ~44 МБ до ~200 КБ. Заодно
      `__version__` теперь из `importlib.metadata` (был захардкожен и разошёлся 2.7↔2.8).
- [x] **Флаки-тест** (исправлено 2026-07-09): в `BaseSimulationCore`/`QsSim` добавлен
      опциональный `seed`; `test_qs_sim::test_sim` засиден — воспроизводимый прогон, больше
      не падает на невезучих сэмплах.

## Критерии готовности (DoD эпика)

- Документ `docs/models_gap_analysis.md` создан: таблица «модель — статус — ценность —
  сложность — приоритет» + рекомендации, с указанием источников (см. DoD исследовательской
  задачи в [../DOD.md](../DOD.md)).
- `docs/models.md` соответствует коду.
- Заведены roadmap/эпики минимум по 2–3 топ-кандидатам.

## Результаты

1. **Каталог сверен с кодом** — `docs/models.md` исправлен (11 ренеймов, 4 добавленные модели),
   способ проверки: исполнение всех import-строк каталога.
2. **Gap-анализ** — [../models_gap_analysis.md](../models_gap_analysis.md):
   19 кандидатов с приоритизацией must/should/could/defer, верифицированные источники (DOI),
   обзор экосистемы (уникальные ниши: size-based, vacations, negatives, retrial, MAP/PH-вакуум
   после заморозки BuTools).
3. **Следующие эпики**: EPIC-002 (волна точных базовых моделей), EPIC-003 (QBD/MAP/PH, флагман),
   EPIC-004 (retrial + Erlang-A) — с детальными roadmaps.
4. **Бэклог рефакторинга** нейминга классов (см. раздел выше) — вести в рамках отдельных задач
   с deprecation-алиасами.
