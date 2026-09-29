# EPIC-022: Fork-Join с тяжёлыми хвостами (точный max n·Pareto)

- **Статус:** done (2026-09-29)
- **Создан:** 2026-09-29
- **Roadmap:** [../roadmaps/fork_join_heavy_tail_roadmap.md](../roadmaps/fork_join_heavy_tail_roadmap.md)

## Цель

Расширить fork-join/split-join стек библиотеки точным (не аппроксимационным) расчётом для
**тяжёлохвостых** (Pareto) времён обслуживания подзадач: максимум n независимых Pareto(α,K) имеет
замкнутую форму CDF и моментов (через Beta-функцию), которой сейчас в `MaxDistribution` нет — там
только light-tailed (H2/Gamma/Erlang) численная квадратура.

## Контекст

Обзор литературы: [../research/fork-join-heavy-tail-2026.md](../research/fork-join-heavy-tail-2026.md).
Направление — из резерва `queueing-trends-2026.md`/`sla-deadline-queueing-2026.md`, подтверждено
активным дважды (июль и сентябрь 2026): Boxma et al., *Fork–join and redundancy systems with
heavy-tailed job sizes*, Queueing Systems 2022; *Extreme values for the waiting time in large
fork-join queues*, Queueing Systems 2025.

Ключевое наблюдение: максимум n iid Pareto(α, K) не нужно аппроксимировать. Заменой `U=F(X)`
(максимум Pareto ↔ максимум Uniform(0,1), плотность `n·u^{n-1}`) получаем точный результат:

- `P(max > x) = 1 - (1 - (K/x)^α)^n`, точно, для любых n, α, x ≥ K.
- `E[max^k] = K^k · n · B(n, 1 - k/α)` (Beta-функция), корректно при `k < α`.

Композиция с уже существующим `SplitJoinCalc` (max → точный M/G/1 P-K) даёт **точный** калькулятор
sojourn-time для split-join при α > 2 (конечная дисперсия) — не новая теория, а сборка уже готовых
точных кусков. При α ≤ 2 P-K неприменим (нужна дисперсия), но точная CDF/хвост максимума остаётся
валидной и полезной сама по себе (хвост «времени ответа худшей подзадачи»).

Инфраструктура симуляции уже готова: `ForkJoinSim`/`QsSim.set_servers(params, "Pa")` уже
поддерживает Pareto через `most_queue.random.distributions.ParetoDistribution` — кросс-валидация
не требует нового симулятора.

## Задачи

### Ядро: точный max n·Pareto

- [x] `most_queue/theory/utils/max_dist.py`: `pareto_max_moments(params, n, num)` (моменты через
      `math.lgamma`-based Beta-функцию, лог-пространство против overflow при больших n) и
      `pareto_max_tail(params, n, x)` (точная CDF/хвост). `ValueError` при запрошенном порядке
      момента `>= α`.
- [x] Юнит-тесты `tests/units/test_max_dist_pareto.py` (13 тестов): сверка Beta-формулы с
      численным интегрированием на области Uniform(0,1) (там, где `quad` численно устойчив —
      исходная попытка через хвост на `[K,∞)` показала расхождения интеграции, не формулы: см.
      разбор ниже) для `n∈{1,2,5,20}`; `n=1` сводится к `ParetoDistribution`; `ValueError` при
      `k>=α`; монотонность хвоста по x и по n; границы `n<1`.

### Интеграция в `SplitJoinCalc`

- [x] `theory/fork_join/split_join.py`: `approximation="pareto"` — допустимое значение
      `calc_params.approx_distr`; `set_servers` в этом режиме принимает `ParetoParams` (типовая
      проверка в обе стороны: `ParetoParams` только с `approximation="pareto"` и наоборот). Метод
      `_pareto_b_max()` запрашивает `min(num, floor(α-ε))` моментов (не жёстко 4 — иначе валидный
      `α=3.5` ломался бы на запросе несуществующего 4-го момента). При `α ≤ 2` — явная
      `ValueError` с указанием использовать `pareto_max_tail`. `get_v_delta` (warm-up-задержка) —
      явный `NotImplementedError` для pareto (не в этой волне).

### Кросс-валидация

- [x] `tests/test_fork_join_heavy_tail_vs_sim.py`: `SplitJoinCalc(approximation="pareto")` при
      `α=4.5` против `ForkJoinSim(..., is_sj=True)` с `set_servers(pareto_params, "Pa")` — моменты
      sojourn в пределах допуска. Плюс независимая проверка `pareto_max_tail` против эмпирического
      хвоста максимума, сэмплированного напрямую (inverse-CDF), без прохождения через M/G/1-слой.

### Документация

- [x] `docs/models/fork-join.md`+`.ru.md`: секция «Тяжёлые хвосты подзадач (Pareto)» — точные
      формулы, ограничение `k<α`, ссылка на резерв (one-big-jump асимптотика очереди при α≤2 — не
      реализована).
- [x] `docs/models/sla.md`+`.ru.md`: примечание, что fit-based `deadline_violation_prob`
      предполагает конечную дисперсию и неприменима при α≤2 — кросс-ссылка на `pareto_max_tail`.
- [x] `docs/models.md`/`.ru.md`, `README.md`/`README.ru.md` — обновлены строки Fork-Join.
- [x] `docs/epics/README.md` — реестр обновлён.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. `pareto_max_moments`/`pareto_max_tail` при `n=1` точно сводятся к исходному Pareto
   (`ParetoDistribution.calc_theory_moments`/`get_tail`).
2. Формула момента (Beta-функция) сверена с независимым численным интегрированием для нескольких
   (n, α) — расхождение < 1e-6.
3. `SplitJoinCalc(approximation="pareto")` при α>2 сверяется с DES (`ForkJoinSim`) в пределах
   стандартного допуска репозитория.
4. Явная, понятная ошибка при `k ≥ α` (момент не существует) — не NaN/inf.
5. Документация (EN+RU) на месте.

## Результаты

Реализовано по плану ([roadmap](../roadmaps/fork_join_heavy_tail_roadmap.md)). Кратко:

- **Ядро:** `most_queue/theory/utils/max_dist.py::pareto_max_moments`/`pareto_max_tail` — точные
  замкнутые формулы (Beta-функция для моментов, элементарная CDF для хвоста), выведены и
  перепроверены численным интегрированием (13 юнит-тестов, `tests/units/test_max_dist_pareto.py`).
- **Интеграция:** `SplitJoinCalc(calc_params=CalcParams(approx_distr="pareto"))` — точная
  композиция max-of-n-Pareto с точной M/G/1 P-K формулой при α>2; понятная `ValueError` при α≤2
  (вместо тихого NaN/inf), явный `NotImplementedError` для пока не поддержанного `get_v_delta`.
- **Кросс-валидация:** `tests/test_fork_join_heavy_tail_vs_sim.py` — `SplitJoinCalc` (α=4.5) против
  `ForkJoinSim` (DES с Pareto-обслуживанием), плюс независимая проверка `pareto_max_tail` против
  эмпирического максимума (inverse-CDF sampling, 200k прогонов).
- **Документация:** `docs/models/fork-join.md`+`.ru.md` (новая секция), кросс-ссылка из
  `docs/models/sla.md`+`.ru.md` (fit-based SLA неприменим при α≤2), README/README.ru.md,
  `docs/models.md`/`.ru.md`.
- **Тесты:** полный прогон `tests/` — **485 passed, 0 failed** (895s; 472 было до EPIC-022, +13
  новых юнит-тестов), регрессий нет, все предупреждения pre-existing. `black --check` чист.
- **Резерв на будущее** (см. research doc): «one big jump» асимптотика полного времени отклика
  очереди при α≤2 (Boxma et al. 2022) и (n,k)-порядковые статистики Pareto за пределами максимума —
  оба требуют более аккуратного вывода, не реализованы в этой волне.
