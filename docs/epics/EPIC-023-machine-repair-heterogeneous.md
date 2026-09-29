# EPIC-023: Machine repair с двумя гетерогенными ремонтниками

- **Статус:** done (2026-09-29)
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/machine_repair_heterogeneous_roadmap.md](../roadmaps/machine_repair_heterogeneous_roadmap.md)

## Цель

Расширить `MachineRepairCalc` (EPIC-019) на случай **двух ремонтников с разной скоростью ремонта**
(`eta_a ≠ eta_b`) — точной малой CTMC, не приближением, по классической технике Krishnamoorthi
(1963) для гетерогенных серверов, перенесённой на конечно-источниковую (finite-source) систему с
тёплым резервом.

## Контекст

Обзор литературы: [../research/machine-repair-heterogeneous-2026.md](../research/machine-repair-heterogeneous-2026.md).
Направление из резерва `unreliable-queues-2026.md`, подтверждено дважды при повторных проходах
lit-search (свежий survey 2026 «Machine Repair Problems with Standby Systems» +
«Performance analysis and optimization of a machine repair problem with warm spares and two
heterogeneous repairmen», Optimization and Engineering 2012 — прямой первоисточник для этого
эпика).

Ключевое наблюдение: текущий `MachineRepairCalc` — чистый birth-death по числу неисправных `j`,
что верно только при одинаковой скорости ремонта (не важно, какой конкретно ремонтник занят). При
`eta_a ≠ eta_b` состояние `j=1` нужно расщепить на «занят A» / «занят B» (второе достижимо только
из `j=2`, когда A освобождается первым) — при `j=0` и `j>=2` состояние по-прежнему однозначно
описывается `j`. Итог — малая точная CTMC (`M+S+2` состояния вместо `M+S+1`), решаемая уже готовой
`theory/reliability/utils.py::ctmc_stationary`.

## Задачи

### Ядро: точная CTMC

- [x] `most_queue/theory/reliability/machine_repair_heterogeneous.py`:
      `MachineRepairHeterogeneousCalc(n_machines, n_spares=0)`, `set_sources(xi, eta_a, eta_b,
      xi_s=None)` (без требования упорядоченности от вызывающего — быстрый определяется внутри как
      `max(eta_a, eta_b)`), `run() -> MachineRepairResults` (общий dataclass с `MachineRepairCalc`,
      расширен полями `utilization_a`/`utilization_b`). Политика назначения:
      "fastest-available-first" — при `j=0→1` всегда берёт быстрый.
- [x] Юнит-тесты `tests/units/test_machine_repair_heterogeneous.py` (8 тестов): редукция при
      `eta_a=eta_b` к существующему `MachineRepairCalc` (точное совпадение, разные `(M,S,ξ,η,ξ_s)`
      наборы); **инвариант потокового баланса** `failure_throughput == eta_a*utilization_a +
      eta_b*utilization_b` для генуинно гетерогенных наборов (сильная model-independent проверка
      корректности CTMC — план изначально предполагал проверку через предел `eta_b→0`, но это
      оказалось математически некорректным тестом: при `eta_b→0` система не сходится к `R=1`,
      т.к. «застрявший» медленный ремонтник перманентно выводит из строя половину мощности —
      качественно другой режим, не редукция; заменено на инвариант баланса); нормализация порядка
      `eta_a`/`eta_b` в `set_sources`; валидность распределения (`sum(p)=1`, `p>=0`).

### Sim и кросс-валидация

- [x] `most_queue/sim/reliability.py`: `MachineRepairHeterogeneousSim` — явно отслеживает
      `(failed, busy_a, busy_b)`; при рождении — fastest-available-first; при смерти — repairman
      немедленно перезанимается, только если очередь ещё не покрыта оставшимся занятым (иначе
      просто освобождается) — выведено и проверено по таблице переходов из roadmap.
- [x] `tests/test_reliability.py::test_machine_repair_heterogeneous_vs_sim` —
      `MachineRepairHeterogeneousCalc` vs `MachineRepairHeterogeneousSim` (400k событий,
      `rtol=0.03` по mean_failed/availability/utilization_a/utilization_b).

### Документация

- [x] `docs/models/reliability.md`+`.ru.md`: секция «Machine repair, два гетерогенных
      ремонтника» — техника расщепления состояния, ссылка на Krishnamoorthi 1963.
- [x] `docs/models.md`/`.ru.md`, README/README.ru.md — строка Reliability обновлена.
- [x] `docs/epics/README.md` — реестр.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. При `eta_a = eta_b` результат точно (не приближённо) совпадает с существующим
   `MachineRepairCalc(n_repairmen=2, ...)`.
2. Против DES-симулятора — в пределах стандартного допуска репозитория.
3. `utilization_a + utilization_b` согласуется с `repairmen_utilization`-эквивалентом (суммарная
   загрузка ремонтников).
4. Документация (EN+RU) на месте.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/machine_repair_heterogeneous_roadmap.md)). Кратко:

- **Ядро:** `MachineRepairHeterogeneousCalc` (`theory/reliability/machine_repair_heterogeneous.py`)
  — малая точная не-birth-death CTMC (`M+S+2` состояния), решается уже готовой
  `theory/reliability/utils.py::ctmc_stationary`. `MachineRepairResults` расширен полями
  `utilization_a`/`utilization_b`.
- **Проверка:** точная редукция к `MachineRepairCalc` при `eta_a=eta_b` (совпадение до ~1e-16);
  инвариант потокового баланса `failure_throughput == eta_a*utilization_a + eta_b*utilization_b`
  (тоже ~1e-16) — заменил изначально запланированный, но математически некорректный тест на предел
  `eta_b→0` (см. roadmap §5 и памятку в epic-задачах).
- **Sim + кросс-валидация:** `MachineRepairHeterogeneousSim`
  (`most_queue/sim/reliability.py`) + `test_machine_repair_heterogeneous_vs_sim`.
- **Документация:** `docs/models/reliability.md`+`.ru.md` (новая секция), README/README.ru.md,
  `docs/models.md`/`.ru.md`.
- **Тесты:** полный прогон `tests/` — **517 passed, 0 failed** (862s), регрессий нет, все
  предупреждения pre-existing. `black --check` чист.
- **Резерв на будущее:** R>2 гетерогенных ремонтника (комбинаторный взрыв состояний, нужна другая
  структура) — не реализовано.
