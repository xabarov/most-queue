# EPIC-024: Queueing-inventory systems (M/M/1, backorder, (0,S)-политика)

- **Статус:** done (2026-09-29)
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/queueing_inventory_roadmap.md](../roadmaps/queueing_inventory_roadmap.md)

## Цель

Добавить в библиотеку первую **queueing-inventory** модель: M/M/1 очередь, где обслуживание
заявки расходует единицу со склада; склад пополняется по политике `(0,S)` за случайное
(экспоненциальное) время поставки; при опустошении склада заявки **не теряются**, а ждут
(backorder) — классическая постановка Schwarz–Daduna (Queueing Systems 2006, Math. Methods of OR
2006). Точное решение через уже существующий в библиотеке QBD-солвер.

## Контекст

Обзор литературы: [../research/queueing-inventory-2026.md](../research/queueing-inventory-2026.md).
Единственное направление из резерва `sla-deadline-queueing-2026.md`, подтверждённое активным трижды
подряд (152 + 64 + 87 цитирований на трёх основополагающих статьях школы Schwarz–Daduna, плюс
устойчивый поток 2022–2026), но ещё не реализованное — в отличие от закрытых уже EPIC-022
(fork-join Pareto) и EPIC-023 (machine repair heterogeneous).

Ключевое наблюдение: модель — это **ровно QBD-процесс** (уровень = число заявок `n` в системе,
фаза = запас на складе `i ∈ {0,...,S}`), и в библиотеке уже есть полностью общий QBD-солвер
(`theory/matrix/qbd.py::QBDSolver`, logarithmic reduction, тот же инструмент, что использует
флагманский MAP/PH-стек) — не нужна новая численная машинерия, только новая матричная структура
блоков `A0/A1/A2/B00/B01/B10`, выведенная в roadmap.

## Задачи

### Ядро: точная QBD-модель

- [x] `most_queue/theory/inventory/mm1_inventory.py`: `MM1QueueingInventoryCalc(s_max)`,
      `set_sources(l)`, `set_servers(mu, theta)` — **отклонение от плана**: `theta` объединён с
      `set_servers` вместо отдельного `set_replenishment` (по образцу
      `MachineRepairCalc.set_sources(xi, eta, xi_s)` — избегает третьего флага
      `is_replenishment_set`). `run() -> QueueingInventoryResults` (E[N], `v`=[E[N]/λ] по
      Little's law, `w`=[v[0]-1/μ] — точная декомпозиция V=W+S для FCFS, без отдельного вывода
      E[Nq]), распределение запаса (точная сумма геометрического ряда `pi0 + pi1(I-R)^-1`, не
      усечение), `stockout_prob`, `fill_rate`. Строит блоки `A0/A1/A2/B00/B01/B10` по выводу из
      roadmap, решает через `most_queue.theory.matrix.qbd.QBDSolver`.
      `QueueingInventoryResults(QueueResults)` — новый dataclass в `structs.py`.
- [x] Юнит-тесты `tests/units/test_mm1_queueing_inventory.py` (6 тестов): редукция при большом
      `S`+`θ` к точному M/M/1 (`rtol=1e-3`); `QBDSolver.residual() < 1e-8`; валидность
      распределений (`sum=1`, `p>=0`); точная декомпозиция `V=W+1/μ`; монотонность ожидания по
      `S`/`θ`; отказ при `ρ>=1`.

### Sim и кросс-валидация

- [x] `most_queue/sim/inventory.py`: `MM1QueueingInventorySim` — explicit `(n, i)` состояние, три
      события (приход/обслуживание при `i≥1`/пополнение при `i=0`), прямой перенос правил из §2
      roadmap (проще, чем EPIC-023 — не нужно расщепление состояний).
- [x] `tests/test_inventory.py::test_mm1_queueing_inventory_vs_sim` — 400k событий, допуск как у
      соседних тестов репозитория.

### Документация

- [x] `docs/models/inventory.md`+`.ru.md` — новая страница каталога.
- [x] `docs/models.md`/`.ru.md` — новая строка «Queueing-inventory systems» в таблице семейств +
      строка в «Model comparison table».
- [x] README.md/README.ru.md — пункт в списке возможностей.
- [x] `docs/epics/README.md` — реестр.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. При `S` достаточно большом и `theta` достаточно большом (запас практически никогда не
   заканчивается) `E[N]`/`E[W]` сходятся к точным M/M/1-значениям.
2. `QBDSolver.residual()` < `1e-8` (баланс `A2 + A1 G + A0 G^2 = 0` решён корректно).
3. Против DES — в пределах стандартного допуска репозитория.
4. `P(stockout)` и fill rate — консистентные, корректно нормированные вероятности.
5. Документация (EN+RU) на месте.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/queueing_inventory_roadmap.md)). Кратко:

- **Ядро:** `MM1QueueingInventoryCalc` (`theory/inventory/mm1_inventory.py`) — прямое
  переиспользование `theory/matrix/qbd.py::QBDSolver` (та же машинерия, что у MAP/PH-стека) с
  новой матричной структурой (уровень = заявки, фаза = запас). Точная сумма геометрического ряда
  для распределения запаса (`pi0 + pi1(I-R)^-1`), точная декомпозиция `V=W+1/μ` вместо отдельного
  вывода E[Nq]. `QueueingInventoryResults(QueueResults)` — новый dataclass в `structs.py`.
- **Проверка:** точная редукция к M/M/1 при большом `S`/`θ` (residual QBD ~1e-14); DES-кросс-
  валидация (`MM1QueueingInventorySim`, `most_queue/sim/inventory.py`).
- **Документация:** `docs/models/inventory.md`+`.ru.md` с новой схемой, README/README.ru.md,
  `docs/models.md`/`.ru.md`.
- **Тесты:** полный прогон `tests/` — **526 passed, 0 failed** (913s), регрессий нет.
- **Резерв на будущее:** lost-sales вариант, общая `(s,S)`-политика (`s>0`), многоканальный
  случай, композиция с negative-customers/приоритетами — не реализовано.
