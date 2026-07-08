# EPIC-004: Retrial-очереди и Erlang-A

- **Статус:** proposed
- **Создан:** 2026-07-08
- **Roadmap:** [../roadmaps/retrial_erlang_a_roadmap.md](../roadmaps/retrial_erlang_a_roadmap.md)

## Цель

Модели поведения абонента: повторные попытки (retrial, орбита) и уход из очереди
(abandonment, Erlang-A). Оба направления — приоритет **should** по
[gap-анализу](../models_gap_analysis.md), высокий прикладной спрос (телеком, колл-центры),
retrial дополнительно — научный фокус проекта.

## Контекст

Retrial: точные решения M/M/1 и M/G/1 (Falin–Templeton) в стиле имеющейся моментной
машинерии; open-source реализаций нет. Erlang-A: обобщение готового `impatience/mm1.py`
до M/M/n+M. Синергия с EPIC-003: после QBD открываются многоканальный retrial и MAP-retrial
(кандидат в публикацию).

## Задачи

- [ ] A1. `MMnImpatienceCalc` (Erlang-A): вероятности состояний, P(wait), P(abandon), моменты W
- [ ] A2. Staffing-задачи (мин. n при целевых метриках)
- [ ] B1. `MM1RetrialCalc` (точное стационарное распределение, орбита)
- [ ] B2. `MG1RetrialCalc` (PGF/LST Falin–Templeton, стохастическая декомпозиция)
- [ ] B3. Механизм орбиты в симуляторе (`RetrialQsSim`)
- [ ] B4. Валидация на сетке параметров + предельные случаи (γ→∞, θ→0)

## Критерии готовности (DoD эпика)

DoD моделей из [../DOD.md](../DOD.md); отдельно: тесты предельных случаев (retrial→M/G/1,
Erlang-A→M/M/n) и запись обоих семейств в `docs/models.md`.

## Результаты

*(заполняется по завершении)*
