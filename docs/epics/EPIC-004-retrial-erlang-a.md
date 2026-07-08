# EPIC-004: Retrial-очереди и Erlang-A

- **Статус:** done (2026-07-08)
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

- [x] A1. `MMnImpatienceCalc` (`theory/impatience/mmn.py`): birth-death, P(wait), P(abandon),
      средние w/v по Литтлу — 2026-07-08
- [x] A2. Staffing: `find_min_servers(target_abandonment)` — 2026-07-08
- [x] B1. `MM1RetrialCalc` (`theory/retrial/mm1.py`): точное решение level-dependent цепи
      адаптивным усечением — 2026-07-08
- [x] B2. `MG1RetrialCalc`: формула среднего размера орбиты Falin–Templeton, верифицирована
      численно (совпадение 1e-8 с точным M/M/1-решением) и симуляцией с Gamma-обслуживанием.
      Старшие моменты W не реализованы (нет надёжной замкнутой формы) — средние по Литтлу — 2026-07-08
- [x] B3. `RetrialQueueSim` (`sim/retrial.py`): орбита с классической линейной политикой — 2026-07-08
- [x] B4. Валидация: n=1 = MM1Impatience (1e-10), θ→0 = Erlang C, γ→∞ = M/G/1,
      сверки с симуляцией по p/w/v; 8 тестов `test_retrial_erlang_a.py` — 2026-07-08

## Критерии готовности (DoD эпика)

DoD моделей из [../DOD.md](../DOD.md); отдельно: тесты предельных случаев (retrial→M/G/1,
Erlang-A→M/M/n) и запись обоих семейств в `docs/models.md`.

## Результаты

- Erlang-A: `MMnImpatienceCalc` + staffing-помощник; сверка с `ImpatientQueueSim` и точными
  вырожденными случаями (MM1Impatience, Erlang C).
- Retrial: точный численный `MM1RetrialCalc` (усечение level-dependent цепи), формульный
  `MG1RetrialCalc` (Falin–Templeton, верифицирован двумя независимыми путями),
  симулятор `RetrialQueueSim`.
- Записи в каталоге моделей (EN и RU) со схемой орбиты (`figures/retrial.png`).
- Многоканальный retrial (через LDQBD/Neuts–Rao поверх QBD-ядра EPIC-003) — в будущий MAP-эпик.
