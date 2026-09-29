# Roadmap: Machine repair с двумя гетерогенными ремонтниками

> Источник техники: Krishnamoorthi B., *On Poisson Queue with Two Heterogeneous Servers*,
> Operations Research, 1963, doi:10.1287/opre.11.3.321. Первоисточник для machine repair:
> *Performance analysis and optimization of a machine repair problem with warm spares and two
> heterogeneous repairmen*, Optimization and Engineering, 2012, doi:10.1007/s11081-012-9195-1.
> Полный список — `docs/research/machine-repair-heterogeneous-2026.md`.

## 1. Цель

Точное (не приближённое) расширение `MachineRepairCalc` на R=2 ремонтника с разными скоростями
ремонта `eta_a ≠ eta_b`, при сохранении тёплого резерва (`xi_s`) из существующей модели.

## 2. Состояния CTMC

Существующая модель: `j` = число неисправных единиц, `0..n_top` (`n_top = M+S`), birth-death.

При `eta_a ≠ eta_b` состояние `j=1` неоднозначно — нужно знать, какой ремонтник занят. Политика
«быстрый первым» (`eta_a = max(eta_a, eta_b)` после нормализации внутри `set_sources`):

- `j=0`: оба свободны — 1 состояние.
- `j=1`: **два** состояния — «занят A» (единственный путь входа: `j=0→1`, всегда быстрый) и «занят
  B» (единственный путь входа: `j=2→1` через завершение A первым).
- `j=2..n_top`: оба заняты — 1 состояние на каждое `j` (кто именно чем занят, не важно, т.к. на
  выходе не важно "кто освободится" — очередь тут же подхватывает следующего, а при `j` падении
  до 1 restore-от-CDF разбирается отдельно, см. переходы ниже).

Итого `n_top+2` состояний вместо `n_top+1`.

Индексация: `idx(0)=0`, `idx(1,A)=1`, `idx(1,B)=2`, `idx(j)=j+1` для `j>=2`.

**Переходы:**

| Откуда | Рождение (`b(j)`, см. существующую формулу) | Смерть |
|---|---|---|
| `idx(0)` | → `idx(1,A)`, ставка `b(0)` | — |
| `idx(1,A)` | → `idx(2)`, ставка `b(1)` | → `idx(0)`, ставка `eta_a` |
| `idx(1,B)` | → `idx(2)`, ставка `b(1)` | → `idx(0)`, ставка `eta_b` |
| `idx(2)` | → `idx(3)`, ставка `b(2)` (если `n_top>=3`) | → `idx(1,B)`, ставка `eta_a` (A закончил первым, B продолжает); → `idx(1,A)`, ставка `eta_b` |
| `idx(j)`, `j=3..n_top-1` | → `idx(j+1)`, ставка `b(j)` | → `idx(j-1)`, ставка `eta_a+eta_b` (очередь тут же подхватывает — кто именно освободился, не важно) |
| `idx(n_top)` (верх, `j>=3`) | — (birth=0 по построению `b`) | → `idx(n_top-1)`, ставка `eta_a+eta_b` |

Формула `b(j)` — та же, что в существующем `MachineRepairCalc._birth`:
`b(j) = xi * min(M, M+S-j) + xi_s * max(0, S-j)`.

При `n_top=1` состояние `idx(1,B)` недостижимо (нулевая стационарная вероятность), но остаётся в
системе с корректным исходящим переходом (`death -> idx(0)` at rate `eta_b`) — `ctmc_stationary` из
`theory/reliability/utils.py` решает это без особого случая (не изолированное состояние, просто с
нулевым входящим потоком).

**Редукция-регрессия:** при `eta_a = eta_b = eta` цепь эквивалентна существующему birth-death
`MachineRepairCalc` (по построению — переходы `idx(2)→idx(1,A)`/`idx(1,B)` симметричны, суммарная
вероятность `p[idx(1,A)]+p[idx(1,B)]` должна точно совпасть с `p[1]` гомогенной модели).

## 3. API

```python
@dataclass
class MachineRepairResults:  # переиспользуем существующий dataclass, добавим поля
    ...
    utilization_a: float = 0.0  # P(repairman A busy)
    utilization_b: float = 0.0  # P(repairman B busy)


class MachineRepairHeterogeneousCalc:
    def __init__(self, n_machines: int, n_spares: int = 0): ...
    def set_sources(self, xi: float, eta_a: float, eta_b: float, xi_s: float | None = None):
        """eta_a, eta_b in any order -- internally normalised to fast/slow."""
    def run(self) -> MachineRepairResults: ...
```

Решение через `most_queue.theory.reliability.utils.ctmc_stationary(transitions, n_states)` —
строим список `(from, to, rate)` по таблице переходов выше, никакой отдельной линейной алгебры не
пишем (переиспользуем готовую утилиту, как `priority/impatience.py`/`priority/map_ph_priority.py`).

`p` (маргинальное распределение по `j`) получаем суммированием `pi[idx(1,A)]+pi[idx(1,B)]` в
позицию `j=1`.

## 4. Sim

`most_queue/sim/reliability.py::MachineRepairHeterogeneousSim` — расширение тактового цикла
`MachineRepairSim`: вместо одного счётчика `failed` держим explicit `(failed, busy_a: bool,
busy_b: bool)`. При рождении — если оба свободны, занять A; если один свободен — занять его; если
оба заняты — просто `failed += 1` (очередь растёт). При смерти — случайно выбрать, кто из занятых
освобождается (по ставке `eta_a`/`eta_b`, конкурирующие экспоненты), затем, если `failed > текущее
число занятых`, немедленно занять освободившегося следующим из очереди.

## 5. Тесты

| Файл | Что проверяем |
|---|---|
| `tests/units/test_machine_repair_heterogeneous.py` | `eta_a=eta_b` → точная редукция к `MachineRepairCalc`; `eta_b→0` (числовой предел, очень малое значение) → приближается к `MachineRepairCalc(n_repairmen=1)`; `utilization_a+utilization_b` эквивалентно суммарной занятости; консистентность `p` (сумма=1, `mean_failed` через `p`). |
| `tests/test_reliability.py` (доп. функции) | `MachineRepairHeterogeneousCalc` vs `MachineRepairHeterogeneousSim`, допуск как у соседних тестов в этом файле (`rtol=0.02`). |

## 6. Резерв

- **R>2 гетерогенных ремонтника** — комбинаторный взрыв состояний (какое именно подмножество
  занято, не только сколько); для точного решения нужна другая структура состояния (например,
  вектор занятости), для приближённого — методы Computers & OR 2013. Не в этой волне.

## 7. Оценка трудозатрат

| Этап | Сложность | Срок (чел.-дней) |
|---|---|---|
| 1. Ядро (CTMC + API) | низкая-средняя | 1 |
| 2. Sim + кросс-валидация | низкая | 1 |
| 3. Документация | низкая | 0.5 |
| **Итого** | | **2.5** |

---

**Следующий шаг:** реализовать `MachineRepairHeterogeneousCalc` в
`most_queue/theory/reliability/machine_repair_heterogeneous.py`.
