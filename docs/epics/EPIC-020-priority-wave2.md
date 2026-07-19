# EPIC-020: Приоритеты, волна 2 — accumulating priority, нетерпение, MAP-вход, retrial, preemptive-repeat

- **Статус:** done (2026-07-16)
- **Создан:** 2026-07-15
- **Roadmap:** обзор — [../research/priority-queues-2026.md](../research/priority-queues-2026.md)

## Цель

Вывести приоритетный стек за пределы статических классов: динамические
(accumulating) приоритеты для триажа и SLA, приоритеты с нетерпением для
call-центров, коррелированный (MAP) вход, приоритеты в retrial-моделях и
аналитика preemptive-repeat дисциплин, которые давно есть в симуляторе. Плюс
мелкие долги стека (экспорты, рассинхрон документации).

## Контекст

Приоритетный стек — один из самых сильных в библиотеке (EPIC-009: RDR-A, точные
CTMC-эталоны, дисперсия отклика), но весь построен на статических классах с
пуассоновским входом и без нетерпения. Ландшафт публикаций и ранжирование — в
[обзоре](../research/priority-queues-2026.md). Первоисточники: Stanford–Taylor–Ziedins
(Queueing Systems 2013), Sharif–Stanford (ORHC 2014), Iravani–Balcıoğlu (QS 2008),
Choi (QS 2001), Takine (JORSJ 1996), Horváth (Performance Evaluation 2012),
Operational Research 2015 (retrial PR), Gaver (JRSS-B 1962).

## Задачи

### П.1 Accumulating priority queue (APQ)

- [x] `MG1AccumulatingPriorityCalc` (`theory/priority/accumulating.py`): M/G/1 APQ,
      K классов, точные средние по рекурсии Клейнрока (формула сверена по первоисточникам);
      распределения (ЛСТ) — в резерв.
- [x] `AccumulatingPrioritySim` (`sim/accumulating_priority.py`): выбор по максимальному
      кредиту b_k·(ожидание), сидируемый.
- [x] Валидация (`tests/test_accumulating_priority.py`, 4 теста): равные b == FIFO
      (1e-12); экстремальные отношения == формула Кобхэма (1e-8; библиотечный
      `MG1NonPreemptiveCalc` совпадает в пределах своей ~1e-4 точности R-факторов);
      закон сохранения Σρ_kW_k (1e-12); vs sim (5%).

### П.2 Приоритеты + нетерпение (M/M/n + M, 2 класса)

- [x] `MMnPriorityImpatienceCalc` (`theory/priority/impatience.py`): 2 класса, NP,
      per-class θ_k; усечённая CTMC на общем решателе `ctmc_stationary`; средние w/v,
      вероятности ухода, occupancy. PR-вариант — в резерв (NP — стандарт колл-центров).
- [x] `MMnPriorityImpatienceSim` (`sim/priority_impatience.py`) — сидируемый.
- [x] Валидация (`tests/test_priority_impatience.py`, 4 теста): один класс == Erlang-A
      (1e-6); **точное тождество**: при θ_0=θ_1 суммарная очередь == агрегированный
      Erlang-A (1e-6); порядок ожиданий/уходов по классам; vs sim (5-6%).

### П.3 MAP/PH/1 с приоритетами (2 класса)

- [x] `MapPh1PriorityCalc` (`theory/priority/map_ph_priority.py`): MMAP[2] +
      PH per class, дисциплины **NP, PR (заморозка фазы) и RS** (бонус к плану);
      точная усечённая CTMC с автостом усечения и жёстким пределом.
- [x] Валидация (`tests/test_map_ph_priority.py`, 3 теста): Poisson+exp NP == Кобхэм
      (1e-6); PR == независимая замкнутая форма preemptive-resume (1e-6; попутная
      находка: `MG1PreemptiveCalc` даёт для низшего класса иное значение — 2.715 против
      2.451 замкнутой формы и точной CTMC — требует отдельного разбора); коррелированный
      MMAP (суперпозиция двух MMPP) + Cox-2 vs `PriorityQueueSimulator` c MAP-источниками (7%).

### П.4 M/M/1 retrial с приоритетами (2 класса)

- [x] `MM1RetrialPriorityCalc` (`theory/priority/retrial_priority.py`): очередь
      приоритетных × орбита × состояние прибора, class-dependent μ; усечённая CTMC.
- [x] `MM1RetrialPrioritySim` (`sim/retrial_priority.py`) — сидируемый.
- [x] Валидация (`tests/test_retrial_priority.py`, 3 теста): λ_0→0 == формула
      Falin–Templeton для орбиты (1e-6); γ→∞ == двухклассовый Кобхэм (1%); vs sim (5%).

### П.5 M/G/1 preemptive-repeat — аналитика (RS и RW)

- [x] `MG1PreemptiveRepeatCalc` (`theory/priority/preemptive/mg1_repeat.py`):
      **RS решён точно** (Cox-2 фит + RS-CTMC из П.3 — шире плана: доступен и
      MMAP/PH-вход) + замкнутые формы Гавера для среднего completion time обоих видов
      (RS и RW). Отклонение от плана: у RW-очереди нет конечного марковского
      представления (нужно помнить повторяемую длительность), а delay-cycle кандидаты,
      откалиброванные по точной RS-CTMC, дают систематическую ошибку O(1%) — ожидания
      RW сознательно НЕ публикуются, направление в резерв (честнее, чем приближение
      без оценки погрешности).
- [x] Валидация (`tests/test_mg1_repeat.py`, 4 теста): exp: RS == PR замкнутые формы
      (1e-5); RS vs `PriorityQueueSimulator("RS")` на общем Cox-2 (6%) — первый
      аналитический бенчмарк дисциплины; E[C_RS] exp == b(1+aE[B]) (1e-9); E[C_RW] ≥
      E[C_RS] (Йенсен); RW.run() — понятный NotImplementedError.

### Общее

- [x] Экспорты `theory/priority/__init__.py`: 5 новых классов + долги EPIC-009
      (`MMkPriorityExact`, `RDRAPriorityCalc`, `RDRAPriorityPH`, `MPhPhK2Class`,
      `PhaseType`) — 16 публичных классов.
- [x] Рассинхрон документации починен: `MG1Preemptive(num_of_classes=...)` и т.п. →
      фактические сигнатуры в `docs/priorities.md`(+ru) и `docs/models/priority.md`(+ru).
- [x] Каталог: раздел «Динамические и расширенные приоритетные модели» на странице
      семейства (EN+RU), схема `apq` (EN+RU), 5 строк в сравнительных таблицах хабов,
      строка Priorities в README (EN+RU).
- [x] Тесты сидированные (18 шт. в 5 файлах); black/isort/pylint; эпик done.

## Критерии готовности (DoD эпика)

Общий DoD ([../DOD.md](../DOD.md)). Специфично:

- Каждая модель заперта сводимостью к существующему решателю библиотеки с жёстким
  допуском (NP/FIFO для APQ, Erlang-A, M/PH/1-приоритеты, MM1RetrialCalc, PR для
  RS при exp).
- П.5 сверен с существующим симулятором RS/RW — это первый аналитический
  бенчмарк для этих дисциплин в библиотеке.
- Усечения CTMC — с контролем хвоста (переиспользовать
  `theory/reliability/utils.ctmc_stationary`).

## Результаты

Приоритетный стек вышел за пределы статических классов; все 5 направлений реализованы:

- **`MG1AccumulatingPriorityCalc` + `AccumulatingPrioritySim`** — первый динамический
  приоритет в библиотеке (Клейнрок/APQ): точные средние, спектр FIFO ↔ строгие
  приоритеты одной ручкой; закон сохранения выполняется до 1e-12.
- **`MMnPriorityImpatienceCalc` + `MMnPriorityImpatienceSim`** — приоритетный Erlang-A;
  точное тождество суммарной очереди с агрегированным Erlang-A при равных θ.
- **`MapPh1PriorityCalc`** — стык двух флагманских стеков: MMAP[2]/PH[2]/1 с NP/PR/RS,
  точная CTMC; сводимости к Кобхэму и preemptive-resume 1e-6, коррелированный кейс
  сверен с симулятором через суперпозицию независимых MAP.
- **`MM1RetrialPriorityCalc` + `MM1RetrialPrioritySim`** — приоритетная очередь + орбита;
  пределы: Falin–Templeton (1e-6) и Кобхэм при γ→∞.
- **`MG1PreemptiveRepeatCalc`** — RS решён точно (первый аналитический бенчмарк
  RS-дисциплины симулятора), completion-time формулы Гавера для RS и RW; RW-очередь —
  в резерв с доказанной причиной (нет конечного марковского представления; кандидаты
  формул дают O(1%) систематики против точной RS-CTMC).

Инфраструктура: общий CTMC-решатель `ctmc_stationary` ускорен (COO-сборка вместо
tolil — критично для MMAP/PH-цепей на ~10^5 состояний). Попутные находки, требующие
отдельного разбора (кандидаты в следующий эпик): `MG1PreemptiveCalc` расходится с
замкнутой формой и точной CTMC по низшему классу (2.715 vs 2.451 на тестовом кейсе);
`MG1NonPreemptiveCalc` имеет собственную точность ~1e-4 против точной формулы Кобхэма.
Резерв: распределения ожиданий APQ (ЛСТ), PR-вариант нетерпения, RW-очередь,
delayed APQ, нелинейные APQ.
