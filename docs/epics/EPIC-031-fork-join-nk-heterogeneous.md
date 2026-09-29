# EPIC-031: (n,k)-Fork-Join поверх гетерогенных/DAG-ветвей

- **Статус:** done
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/fork_join_nk_heterogeneous_roadmap.md](../roadmaps/fork_join_nk_heterogeneous_roadmap.md)

## Цель

Обобщить EPIC-030 (гетерогенный максимум = n-из-n) на **(n,k)-fork-join**: заявка завершена, как
только любые `k` из `n` (возможно, разнораспределённых) ветвей закончились — не только все `n`.

## Контекст

Обзор литературы: [../research/fork-join-nk-heterogeneous-2026.md](../research/fork-join-nk-heterogeneous-2026.md).
Гетерогенный случай сводится к классической задаче надёжности k-out-of-n:G систем: `P(X_(k)<=x)`
= «хотя бы `n-k+1` из `n` независимых событий `{X_i<=x}` произошли» — точно считается через O(n²)
DP по Poisson-Binomial распределению (Rushdi 1985), дальше та же квадратурная формула момента,
что уже даёт `heterogeneous_max_moments` (EPIC-030). Бонус: i.i.d. Pareto-случай k-й порядковой
статистики имеет точную замкнутую форму через Beta-функцию, обобщающую уже реализованный
`pareto_max_moments` (EPIC-022) — даёт точный регрессионный якорь.

## Задачи

### Ядро

- [x] `most_queue/theory/utils/max_dist.py`: `pareto_kth_order_moments(params, n, k, num)` —
      точная замкнутая форма (Beta-функция), регрессия к `pareto_max_moments` при `k=n`.
- [x] `heterogeneous_kth_order_moments(branches, k, num)` — Rushdi/Poisson-Binomial DP +
      `scipy.integrate.quad`; регрессия к `heterogeneous_max_moments` при `k=n` и к
      `pareto_kth_order_moments` при одинаковых Pareto-ветвях. При реализации найдена и
      исправлена ошибка в собственном выводе (roadmap/research использовали `n-k+1` вместо `k`
      в DP — перепутаны местами min/max); поймана самим же регрессионным тестом на `k=n` до того,
      как код был зафиксирован (см. «Результаты»).
- [x] `most_queue/theory/fork_join/dag.py`: узел `("parallel", children, k)` — необязательный
      `k` (по умолчанию `len(children)`, сохраняет поведение EPIC-030).

### Проверка

- [x] Точные регрессии (к `pareto_max_moments`, к `heterogeneous_max_moments`, друг к другу).
- [x] Monte Carlo для смешанных семейств и `k<n`.
- [x] Монотонность `E[X_(k)]` по `k`.

### Документация

- [x] `docs/models/fork-join.md`+`.ru.md` — секция «(n,k)-fork-join поверх гетерогенных ветвей».
- [x] `docs/models.md`/`.ru.md` — обновить строку/описание.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. `heterogeneous_kth_order_moments` при `k=n` точно (в пределах `quad`) совпадает с
   `heterogeneous_max_moments`.
2. `pareto_kth_order_moments`/`heterogeneous_kth_order_moments` при одинаковых Pareto-ветвях
   совпадают друг с другом при любом `k`.
3. `("parallel", children, k)` в DAG проверено против Monte Carlo.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/fork_join_nk_heterogeneous_roadmap.md)). Кратко:

- **Ядро:** `pareto_kth_order_moments` — новая точная замкнутая форма (не было в репозитории
  раньше) для k-й порядковой статистики i.i.d. Pareto через `U_(k)~Beta(k,n-k+1)`, обобщает
  `pareto_max_moments` (EPIC-022). `heterogeneous_kth_order_moments` — численно точный (при
  заданном семействе) гетерогенный случай через классический DP Rushdi (1985) для
  Poisson-Binomial распределения (та же математика, что и в теории надёжности k-out-of-n:G систем
  с гетерогенными компонентами) + квадратура момента хвоста (переиспользует технику
  `heterogeneous_max_moments`, EPIC-030). `ForkJoinDAGCalc`: узел `("parallel", children, k)` —
  необязательный `k`.
- **Найденная и исправленная ошибка в собственном выводе:** первая версия DP-вызова использовала
  `_poisson_binomial_at_least(probs, n-k+1)` вместо корректного `k` — перепутаны местами
  min/max (при `k=n` код по ошибке считал «хотя бы 1» вместо «хотя бы n», отдавая значение
  минимума под видом максимума). Пойман немедленно собственным же регрессионным тестом
  `k=n` против уже проверенного `heterogeneous_max_moments`/`pareto_max_moments` — несовпадение
  было точным и однозначным (значения `k=1` и `k=n` оказались переставлены местами).
- **Проверка:** точные регрессии (`k=n` → max, одинаковые Pareto-ветви → `pareto_kth_order_moments`
  при любом `k`); Monte Carlo для смешанных семейств и `k<n` (<0.1% на 2 моментах, 2М сэмплов);
  монотонность `E[X_(k)]` по `k`; `("parallel",...,k)` в DAG проверено против MC.
- **Тесты:** `tests/units/test_kth_order.py` — 7 passed; `tests/units/test_fork_join_dag.py`
  расширен на 2 новых теста. Полный прогон `tests/` — **618 passed, 0 failed** (884s), регрессий
  нет.
- **Документация:** новая секция «(n,k)-fork-join поверх гетерогенных ветвей» в
  `docs/models/fork-join.md`/`.ru.md` (EN+RU), обновлены `docs/models.md`/`.ru.md`.
- **Резерв на будущее:** purging vs non-purging семантика после k-го завершения (что происходит
  с оставшимися `n-k` ветвями) — теория здесь агностична, DES для однородного случая уже различает
  варианты (`ForkJoinSim`).
