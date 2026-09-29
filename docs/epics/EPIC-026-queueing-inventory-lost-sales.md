# EPIC-026: Queueing-inventory lost-sales вариант

- **Статус:** done (2026-09-29)
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/queueing_inventory_lost_sales_roadmap.md](../roadmaps/queueing_inventory_lost_sales_roadmap.md)

## Цель

Добавить вариант **lost sales** к `MM1QueueingInventoryCalc` (EPIC-024): заявка, пришедшая при
пустом складе (`i=0`), теряется, а не встаёт в очередь (в отличие от текущего backorder-варианта).
Минимальное расширение — параметр политики, не новый класс.

## Контекст

Обзор литературы: [../research/queueing-inventory-lost-sales-2026.md](../research/queueing-inventory-lost-sales-2026.md).
Прямое продолжение EPIC-024, отложенное в резерв при его выборе. Первоисточник: Saffari, Haji,
Hassanzadeh, *The M/M/1 queue with inventory, lost sale, and general lead times*, Queueing
Systems, 2013 (87 цит.) — уже отмечен в research-доке EPIC-024.

Ключевое наблюдение: backorder и lost-sales отличаются **ровно одним переходом** — приходом при
`i=0`. В backorder он поднимает уровень (`n→n+1`); в lost-sales это вообще не переход (заявка
потеряна, состояние не меняется). В терминах QBD-блоков это обнуление строки фазы 0 в `A0`/`B01`
плюс поправка диагонали `A1[0,0]`/`B00[0,0]` (теряет слагаемое `λ`). Всё остальное (`A2`, `A1[0,S]`,
`B00[0,S]`, `B10`) не меняется. Реализуется как параметр `policy` на существующем классе, а не
дублированием.

## Задачи

### Ядро

- [x] `most_queue/theory/inventory/mm1_inventory.py`: `MM1QueueingInventoryCalc.__init__(s_max,
      policy: Literal["backorder","lost_sales"]="backorder", ...)`. `_build_solver()` строит
      `A0`/`B01` с обнулённой строкой фазы 0 и поправленной диагональю `A1[0,0]`/`B00[0,0]` при
      `policy=="lost_sales"`. Добавлена метрика `loss_prob` в `QueueingInventoryResults`.
      **Найдено по ходу (не в изначальном плане):** `get_v()` делил `E[N]` на номинальную `λ` —
      для lost-sales это неверно (закон Литтла требует throughput, а `level n` в lost-sales
      считает только принятых клиентов); добавлен `_effective_arrival_rate()` = `λ·(1-stockout_prob)`
      для lost-sales, `λ` без изменений для backorder.
- [x] Юнит-тесты (расширены `tests/units/test_mm1_queueing_inventory.py`, +7 тестов): backorder
      vs lost-sales дают разные результаты (`E[N]_lost_sales < E[N]_backorder`, подтверждено:
      2.67 vs 8.79 при `S=3,θ=0.5,ρ=0.5`); `S→∞`/`θ→∞` редукция к M/M/1 для обеих политик;
      `loss_prob == stockout_prob` (PASTA) для lost-sales; `loss_prob == 0` для backorder;
      `V=W+1/μ` для обеих политик; валидация `policy`-параметра.

### Sim и кросс-валидация

- [x] `most_queue/sim/inventory.py::MM1QueueingInventorySim` — тот же `policy` параметр; при
      `lost_sales` приход при `i=0` инкрементирует только счётчик потерь, не `n`; `loss_prob`
      считается напрямую (`lost/arrivals`), не через `stockout_prob` — независимая проверка
      PASTA-равенства, а не повторное использование той же величины.
- [x] `tests/test_inventory.py::test_mm1_queueing_inventory_lost_sales_vs_sim` — добавлен.

### Документация

- [x] `docs/models/inventory.md`+`.ru.md` — секция «Lost-sales вариант».
- [x] `docs/models.md`/`.ru.md`, README/README.ru.md — обновлены.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. При `policy="backorder"` поведение **не изменилось** (регрессия существующих EPIC-024 тестов).
2. `lost_sales` даёт меньшее `E[N]`, чем `backorder` при тех же параметрах (содержательная
   проверка направления эффекта, не только «что-то посчиталось»).
3. Обе политики сходятся к точному M/M/1 при `S`/`θ` → большие значения.
4. Против DES — в пределах стандартного допуска.
5. Документация (EN+RU) на месте.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/queueing_inventory_lost_sales_roadmap.md)). Кратко:

- **Ядро:** `policy` параметр на существующем `MM1QueueingInventoryCalc` — минимальное изменение
  `_build_solver()` (обнуление строки фазы 0 в `A0`/`B01`, поправка диагонали `A1[0,0]`/`B00[0,0]`).
  Новое поле `loss_prob` в `QueueingInventoryResults`.
- **Побочная находка:** `get_v()` делил `E[N]` на номинальную `λ`, что неверно для lost-sales
  (закон Литтла требует throughput, а не номинальный приход) — исправлено через
  `_effective_arrival_rate()`. Найдено при написании DES-кросс-валидации, не юнит-теста.
- **Проверка:** `E[N]_lost_sales < E[N]_backorder` при тех же параметрах (2.67 vs 8.79 на тестовом
  наборе); обе политики точно сводятся к M/M/1 при `S`/`θ`→∞; `loss_prob == stockout_prob` (PASTA)
  подтверждён и точной формулой, и независимо — прямым подсчётом в DES.
- **Тесты:** полный прогон `tests/` — **547 passed, 0 failed** (916s), регрессий нет (backorder-
  поведение по умолчанию не изменилось).
- **Резерв на будущее:** общая `(s,S)`-политика, многоканальный случай — не реализовано.
