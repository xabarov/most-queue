# Roadmap: M/M/1 queueing-inventory system, (0,S)-политика, backorder

> Источник постановки: Schwarz M., Wichelhaus C., Daduna H., *Queueing systems with inventory
> management with random lead times and with backordering*, Mathematical Methods of Operations
> Research, 2006, doi:10.1007/s00186-006-0085-1; Schwarz M., Daduna H., *M/M/1 Queueing systems
> with inventory*, Queueing Systems, 2006, doi:10.1007/s11134-006-8710-5.
> Полный список источников — `docs/research/queueing-inventory-2026.md`.

## 1. Цель

Реализовать первую queueing-inventory модель в библиотеке: M/M/1 очередь, где обслуживание
расходует единицу со склада; склад пополняется по политике `(0,S)` (заказ на `S` единиц
оформляется сразу при опустошении) за экспоненциальное время поставки `Exp(θ)`; при опустошении
склада заявки **ждут** (backorder — не теряются). Решение — точное, через уже существующий
`theory/matrix/qbd.py::QBDSolver`.

## 2. Постановка и вывод QBD-блоков

**Состояние:** `(n, i)` — `n ≥ 0` число заявок в системе (в очереди + обслуживаемая), `i ∈
{0,...,S}` — запас на складе. Уровень QBD = `n` (неограничен), фаза = `i` (размер `m = S+1`).

**Переходы:**

| Событие | Ставка | Условие | Результат |
|---|---|---|---|
| Приход заявки | `λ` | всегда | `(n,i) → (n+1,i)` — вверх, фаза не меняется |
| Завершение обслуживания | `μ` | `n≥1` **и** `i≥1` | `(n,i) → (n-1,i-1)` — вниз, расход 1 единицы склада |
| Пополнение склада | `θ` | `i=0` | `(n,0) → (n,S)` — тот же уровень, фаза скачком в `S` |

Обслуживание блокировано при `i=0` (сервер простаивает, даже если `n≥1`, — ждёт пополнения). Это
классический **backorder**-вариант (в отличие от lost-sales, где приход при `i=0` терялся бы, не
увеличивая `n`).

**Блоки для внутренних уровней (`n≥1`), индексация фаз `0..S`:**

```
A0 = λ · I_{S+1}                                   (вверх: приход, фаза не меняется)

A2[i, i-1] = μ  для i = 1..S,  A2[0, :] = 0         (вниз: обслуживание, фаза -1;
                                                      из фазы 0 вниз идти нельзя — заблокировано)

A1[0, S] = θ                                        (пополнение: фаза 0 → S, тот же уровень)
A1[i, i] = -(λ + μ)   для i = 1..S                  (диагональ: суммарный отток из (n,i), i≥1)
A1[0, 0] = -(λ + θ)                                 (диагональ для i=0: приход + пополнение,
                                                      обслуживания нет)
```

**Граничный уровень `n=0`** (нечего обслуживать — `μ`-переходов нет):

```
B00[0, S] = θ                                       (пополнение, как в A1)
B00[i, i] = -λ            для i = 1..S               (только отток по приходу)
B00[0, 0] = -(λ + θ)

B01 = λ · I_{S+1}   = A0                             (вверх из уровня 0, идентично A0)
B10 = A2                                             (вниз с уровня 1 на уровень 0 — та же
                                                       мю-структура, что и между внутренними
                                                       уровнями: фазовое пространство одинаковое)
```

Устойчивость (`sp(R) < 1`) эквивалентна `λ < μ · P(i≥1)` в установившемся режиме — `QBDSolver.solve()`
сам бросает `ValueError`, если это не так (не нужно проверять отдельно).

## 3. API

```python
class MM1QueueingInventoryCalc:
    def __init__(self, s_max: int):
        """:param s_max: S -- maximum stock level, (0, S) reorder policy."""

    def set_sources(self, l: float): ...          # arrival rate lambda
    def set_servers(self, mu: float): ...          # service rate mu
    def set_replenishment(self, theta: float): ...  # lead-time rate theta

    def run(self, num_levels: int = 200) -> QueueingInventoryResults:
        """
        Build A0/A1/A2/B00/B01/B10 per sec. 2, solve via QBDSolver, derive:
          - mean_in_system (E[N], matrix-geometric formula, same pattern as
            MapPhcCalc._mean_in_system)
          - mean_wait / mean_sojourn (Little's law: E[N]/lambda, (E[N]-E[Nq])/lambda)
          - stock distribution (marginal over n, sum of phase-i mass across levels)
          - stockout_prob = P(i=0)
          - fill_rate = 1 - stockout_prob (fraction of time an item is available)
        """
```

`QueueingInventoryResults` — новый dataclass (`most_queue/structs.py` или локальный в
`theory/inventory/`, по аналогии с `MachineRepairResults`): `mean_in_system`, `mean_wait`,
`mean_sojourn`, `stock_distribution: list[float]`, `stockout_prob`, `fill_rate`, `duration`.

`num_levels` — сколько уровней `n` суммировать для маргинального распределения запаса (хвост по
`n` затухает геометрически через `R`, как и в `MapPhcCalc.p` — тот же паттерн усечения).

## 4. Sim

`most_queue/sim/inventory.py::MM1QueueingInventorySim` — тактовый CTMC-симулятор по образцу
`most_queue/sim/reliability.py::MachineRepairHeterogeneousSim` (explicit `(n, i)` state,
`rate = λ + [μ if n>=1 and i>=1] + [θ if i==0]`, три исхода события). Проще, чем heterogeneous
machine repair (не нужно расщепление состояний) — прямой перенос трёх правил из таблицы §2.

## 5. Тесты

| Файл | Что проверяем |
|---|---|
| `tests/units/test_mm1_queueing_inventory.py` | `S` большой + `θ` большой (запас практически никогда не пуст) → `E[N]`/`E[W]` сходятся к точному M/M/1 (`MG1Calc`/`ErlangCCalc`); `QBDSolver.residual() < 1e-8`; `stock_distribution` суммируется в 1; `stockout_prob` согласуется с прямым суммированием маргиналов по уровням. |
| `tests/test_inventory.py` | `MM1QueueingInventoryCalc` vs `MM1QueueingInventorySim`, допуск как у соседних QBD-тестов репозитория. |

## 6. Резерв (не в этой волне)

- **Lost-sales вариант** (Saffari–Haji–Hassanzadeh 2013) — приход при `i=0` не увеличивает `n`
  (теряется), а не блокируется; другая структура блока `A0`/`B01` для фазы `i=0`. Естественное
  следующее расширение — тот же `QBDSolver`, другая матрица.
- **Общая `(s,S)`-политика** (`s>0`, а не только `(0,S)`) — требует отслеживать, размещён ли уже
  заказ, отдельным битом состояния (аналогично тому, как в EPIC-023 пришлось расщеплять состояние
  «кто занят»); при `(0,S)` этот бит избыточен (i=0 однозначно означает «заказ размещён»).
- **Многоканальный** (`c` серверов) — фазовое пространство растёт (нужно отслеживать занятость
  серверов отдельно), либо приближение по образцу existing `c`-серверных QBD-моделей библиотеки.
- **Queueing-inventory + negative customers/catastrophes** и **+ приоритеты** — композиции с уже
  существующими фирменными темами библиотеки; отдельные эпики после базовой модели.

## 7. Оценка трудозатрат

| Этап | Сложность | Срок (чел.-дней) |
|---|---|---|
| 1. Ядро (блоки + `QBDSolver` + метрики) | низкая-средняя | 1–2 |
| 2. Sim + кросс-валидация | низкая | 1 |
| 3. Документация | низкая | 0.5 |
| **Итого** | | **2.5–3.5** |

---

**Следующий шаг:** реализовать `MM1QueueingInventoryCalc` в
`most_queue/theory/inventory/mm1_inventory.py`, начиная с блоков `A0/A1/A2/B00/B01/B10` из §2.
