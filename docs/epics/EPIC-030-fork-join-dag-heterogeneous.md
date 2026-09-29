# EPIC-030: Fork-Join с гетерогенными ветвями и series-parallel DAG

- **Статус:** done
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/fork_join_dag_heterogeneous_roadmap.md](../roadmaps/fork_join_dag_heterogeneous_roadmap.md)

## Цель

Снять два ограничения EPIC-022 (`SplitJoinCalc`/`MaxDistribution`, только плоский `n`-way
i.i.d. fork-join): (1) `n` независимых, но НЕ одинаково распределённых ветвей (гетерогенный
максимум); (2) произвольная **series-parallel** структура задач (DAG), а не только один уровень
fork→join.

## Контекст

Обзор литературы: [../research/fork-join-dag-heterogeneous-2026.md](../research/fork-join-dag-heterogeneous-2026.md).
Максимум независимых, но разнораспределённых величин не имеет замкнутой формы в общем случае
(в отличие от i.i.d. Pareto из EPIC-022), но численно точен (при заданном семействе каждой ветви)
через стандартную интегральную формулу `E[max^k] = k∫t^(k-1)P(max>t)dt`. Series-parallel графы
задач (в отличие от произвольных precedence-DAG, доказанно #P-hard, Dodin 1985) сводятся точно
через рекурсивную композицию: `series` = свёртка моментов (`conv_moments`, уже есть в
библиотеке), `parallel` = гетерогенный максимум (новый блок п.1). Композитные (не raw-leaf) узлы
внутри `parallel` требуют шага подгонки распределения по моментам — тот же fit-based стандарт,
что `SplitJoinCalc` уже использует для не-Pareto случая, не новый источник неточности.

## Задачи

### Ядро

- [x] `most_queue/theory/utils/max_dist.py`: `heterogeneous_max_moments(branches, num)` —
      гетерогенный максимум через `scipy.integrate.quad`; регрессия к `pareto_max_moments` при
      одинаковых Pareto-ветвях. Попутно найден и исправлен краевой случай: `ParetoDistribution.get_tail`
      не был определён/валиден при `t < K`, ломая интегрирование от 0 (см. «Результаты»).
- [x] `most_queue/theory/fork_join/dag.py` (новый файл): `ForkJoinDAGCalc` — рекурсивная
      series/parallel композиция по спецификации DAG, обёртка `MG1Calc` поверх моментов корня
      (Split-Join блокирующая семантика EPIC-022).

### Проверка

- [x] Лёгкий Monte Carlo сэмплер DAG (инлайн в тестах, не полноценный DES — развязка
      очереди/DAG через блокирующую семантику уже проверена ранее) для валидации
      композиционной математики.
- [x] Юнит-тесты: гетерогенный максимум vs MC и vs `pareto_max_moments`; плоский и двухуровневый
      DAG vs MC; `ForkJoinDAGCalc.run()` корректен (`v`/`w`/`utilization`).

### Документация

- [x] `docs/models/fork-join.md`+`.ru.md` — новая секция: гетерогенный максимум,
      series-parallel DAG, честная граница точности.
- [x] `docs/models.md`/`.ru.md` — обновлено описание + новая строка калькулятора.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. Гетерогенный максимум с одинаковыми Pareto-ветвями точно (в пределах `quad`) совпадает с
   `pareto_max_moments`.
2. Композиция DAG проверена против независимого Monte Carlo сэмплера той же структуры.
3. Документация честно разграничивает exact (raw-leaf уровень) и fit-based (составные узлы)
   случаи — без overclaiming.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/fork_join_dag_heterogeneous_roadmap.md)). Кратко:

- **Ядро:** `heterogeneous_max_moments` (`most_queue/theory/utils/max_dist.py`) — момент
  максимума `n` независимых, но разнораспределённых ветвей через `scipy.integrate.quad`;
  `ForkJoinDAGCalc` (`most_queue/theory/fork_join/dag.py`) — рекурсивная `leaf`/`series`/`parallel`
  композиция series-parallel DAG, обёртка `MG1Calc` поверх моментов корня (блокирующая семантика
  EPIC-022).
- **Найденный по ходу баг:** `ParetoDistribution.get_tail`/`get_cdf` не был определён корректно
  при `t < K` (минимум носителя) — все существующие вызовы этого никогда не задевали, но новое
  интегрирование от 0 до ∞ задело. Исправлено локально в новом `branch_tail()` (без изменения
  самого `ParetoDistribution`, чтобы не трогать уже корректный для существующих вызывающих код).
- **Проверка:** точная регрессия к `pareto_max_moments` при одинаковых Pareto-ветвях (rel diff
  ~1e-7); Monte Carlo для смешанных семейств (<0.15% на 3 моментах, 2М сэмплов); плоский и
  двухуровневый DAG проверены против независимого MC-сэмплера (включая fit-шаг для составных
  узлов внутри `parallel`, ~1% ошибка на 3-м моменте — ожидаемо и задокументировано).
- **Тесты:** `tests/units/test_heterogeneous_max.py` — 5 passed; `tests/units/test_fork_join_dag.py`
  — 6 passed. Полный прогон `tests/` — **605 passed, 0 failed** (861s), регрессий нет.
- **Документация:** новая секция в `docs/models/fork-join.md`/`.ru.md` (EN+RU), обновлены
  `docs/models.md`/`.ru.md`.
- **Резерв на будущее:** общие (не series-parallel) precedence-DAG — доказанно #P-hard (Dodin
  1985); (n,k)-fork-join поверх гетерогенных/DAG-ветвей — взято следующим эпиком (EPIC-031).
