# EPIC-028: Queueing-inventory, многоканальный случай (M/M/c)

- **Статус:** done
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/queueing_inventory_multiserver_roadmap.md](../roadmaps/queueing_inventory_multiserver_roadmap.md)

## Цель

Обобщить queueing-inventory семейство (EPIC-024/026/027, все `M/M/1`) на `c > 1` одинаковых
серверов: `MMcQueueingInventoryCalc`, точный QBD, совместимо с общей `(s,S)` и `backorder`/
`lost_sales`.

## Контекст

Обзор литературы: [../research/queueing-inventory-multiserver-2026.md](../research/queueing-inventory-multiserver-2026.md).
Число занятых серверов = `min(n,c)` — не требует отдельного бита состояния (детерминированная
функция числа клиентов `n`, как в обычном `M/M/c`). Но скорость обслуживания зависит от `n` при
`n=0,...,c-1` (`n·μ`), стабилизируясь на `c·μ` только при `n≥c` — то есть `c` разных граничных
уровней вместо одного. Решение: сложить их в один суперблок размерности `c·(S+1)` для
`QBDSolver` (уже поддерживает граничный блок произвольной, но единственной, размерности). При
`c=1` — точная редукция к уже реализованной модели.

## Задачи

### Ядро

- [x] `most_queue/theory/inventory/mmc_inventory.py` (новый файл): `MMcQueueingInventoryCalc`
      — параметры `c`, `s_max`, `s=0`, `policy="backorder"`. Граничный суперблок `B00/B01/B10`
      (roadmap §3), однородная часть `A0/A1/A2` (roadmap §4, `μ→c·μ`).
- [x] Производные метрики: `get_p()` (по подблокам `pi0` + повторяющаяся часть), `_phase_marginal()`,
      `_mean_in_system()` (со сдвигом `(c-1)`), `get_w()` (`V=W+1/μ`, без изменений),
      `_effective_arrival_rate()` (переиспользовать EPIC-026 логику).
- [x] Юнит-тесты: точная редукция при `c=1` к `MM1QueueingInventoryCalc`; больший `c` не
      увеличивает `E[W]`/`stockout_prob`; сумма `get_p()`=1; `residual()` мал; обе `policy`.

### Sim и кросс-валидация

- [x] `most_queue/sim/inventory.py` — новый класс `MMcQueueingInventorySim` — ставка обслуживания
      `min(n,c)·μ`.
- [x] `tests/test_inventory.py` — `MMcQueueingInventoryCalc` (c=2) vs DES.

### Документация

- [x] `docs/models/inventory.md`+`.ru.md` — секция «Многоканальный случай (M/M/c)».
- [x] `docs/models.md`/`.ru.md` — новая строка калькулятора.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. При `c=1` — точная регрессия к `MM1QueueingInventoryCalc` (не приближённое совпадение).
2. Больший `c` не увеличивает `E[W]`/`stockout_prob` при прочих равных.
3. Против DES — в пределах стандартного допуска.
4. Документация (EN+RU) на месте.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/queueing_inventory_multiserver_roadmap.md)). Кратко:

- **Ядро:** новый класс `MMcQueueingInventoryCalc` (`most_queue/theory/inventory/mmc_inventory.py`)
  — `c` граничных подуровней (`n=0,...,c-1`, ставка обслуживания `n·μ`) сложены в единый суперблок
  размерности `c·(S+1)` для `QBDSolver`; однородная часть (`n≥c`) идентична `M/M/1`-модели с
  `μ→c·μ`. `QBDSolver` не менялся — только новая логика построения блоков.
- **Проверка:** точная редукция при `c=1` к `MM1QueueingInventoryCalc` (расхождение ~1e-9); больший
  `c` не увеличивает `E[W]`/`stockout_prob`; малый residual QBD; кросс-валидация против DES
  (`MMcQueueingInventorySim`, новый класс в `most_queue/sim/inventory.py`) для обеих `policy`.
- **Находка при отладке:** `ρ=λ/(c·μ)<1` — необходимое, но не достаточное условие устойчивости
  (уже задокументировано в `_utilization()` с EPIC-024) — при медленном пополнении (`θ` мало
  относительно частоты обнуления склада) обслуживание блокируется достаточно часто, чтобы система
  была неустойчивой даже при низкой номинальной загрузке; подтверждено через drift-условие
  стационарного распределения генератора A0+A1+A2.
- **Тесты:** `tests/units/test_mmc_queueing_inventory.py` — 20 passed;
  `tests/test_inventory.py` — 2 новых теста (DES кросс-валидация). Полный прогон `tests/` —
  **573 passed, 0 failed** (960s), регрессий нет.
- **Документация:** новая секция «Многоканальный случай (M/M/c)» в `docs/models/inventory.md`/
  `.ru.md` (EN+RU), новая строка в `docs/models.md`/`.ru.md`.
- **Резерв на будущее:** гетерогенные серверы (комбинация с EPIC-023/025 state-splitting),
  многотоварный (multi-commodity) склад.
