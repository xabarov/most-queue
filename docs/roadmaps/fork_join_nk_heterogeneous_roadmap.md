# Roadmap: (n,k)-Fork-Join поверх гетерогенных/DAG-ветвей

> Источники — `docs/research/fork-join-nk-heterogeneous-2026.md`. База: EPIC-022, EPIC-030
> (`theory/utils/max_dist.py::heterogeneous_max_moments`, `theory/fork_join/dag.py`).

## 1. Точная замкнутая форма для i.i.d. Pareto (обобщение максимума)

`X_(k)` — k-я порядковая статистика (по возрастанию, `k=1` — минимум, `k=n` — максимум) n i.i.d.
Pareto(α,K). Через `U=F(X)~Uniform(0,1)`: `X_(k) = K/(1-U_(k))^(1/α)`, `U_(k)~Beta(k,n-k+1)`,
`1-U_(k)~Beta(n-k+1,k)`:

```
E[X_(k)^m] = K^m · B(n-k+1 - m/α, k) / B(n-k+1, k)
```

Сходится при `m < α·(n-k+1)` (при `k=n` — ровно `m<α`, как у `pareto_max_moments`). Точная
регрессия: `pareto_kth_order_moments(params, n, k=n, num)` == `pareto_max_moments(params, n, num)`.

## 2. Гетерогенный случай — Rushdi/Poisson-Binomial DP + квадратура

Для `n` независимых, но разнораспределённых ветвей: `X_(k)<=x` ⟺ хотя бы `k` из `n` событий
`{X_i<=x}` произошли (стандартный факт теории порядковых статистик: `k`-я по возрастанию величина
`<=x` тогда и только тогда, когда как минимум `k` из `n` величин `<=x`). При фиксированном `x`,
`p_i(x)=P(X_i<=x)` — вероятности независимых Bernoulli. DP (Rushdi 1985, O(n²) на одну точку `x`):

```
Q_0(0)=1, Q_0(m)=0 (m>0)
Q_j(m) = Q_{j-1}(m)·(1-p_j) + Q_{j-1}(m-1)·p_j     (j=1..n, m=0..j)

P(X_(k) <= x) = Σ_{m=k}^{n} Q_n(m)
```

**Проверка на `k=n`:** сводится к `P(X_(n)<=x)=Π_i p_i(x)` (нужны ВСЕ `n` успехов) — совпадает с
`heterogeneous_max_moments`. При `k=1`: `P(X_(1)<=x)=1-Π_i(1-p_i(x))` (хотя бы 1 успех) — минимум.
(Ранняя версия этого документа по ошибке писала `n-k+1` вместо `k` — перепутаны местами min/max;
исправлено и проверено численно при реализации, см. «Результаты» эпика.)

Дальше — та же интегральная формула момента, что и в `heterogeneous_max_moments`:
`E[X_(k)^m] = m·∫_0^∞ x^(m-1)·(1-P(X_(k)<=x))dx` через `scipy.integrate.quad`, с тем же
разбиением на Pareto-кинки (`branch_tail`, EPIC-030) и увеличенным бюджетом подразбиений для
хвоста. При `k=n` DP-формула упрощается ровно до `P(X_(n)<=x)=Π_i p_i(x)` — совпадает с
`heterogeneous_max_moments` (регрессия).

```python
def heterogeneous_kth_order_moments(branches: list[BranchSpec], k: int, num: int) -> list[float]:
    """k=n reduces exactly to heterogeneous_max_moments; k=1 is the min."""
```

## 3. (n,k)-узел в DAG

`ForkJoinDAGCalc`: узел `("parallel", children)` уже означает "все `len(children)` обязательны"
(EPIC-030). Добавляем необязательный третий элемент:

```
("parallel", children, k)   # k из len(children) обязательны; k по умолчанию = len(children)
```

`dag_moments()` при `kind=="parallel"` читает `k = node[2] if len(node)>2 else len(children)`,
вызывает `heterogeneous_kth_order_moments` вместо `heterogeneous_max_moments` (либо последняя
остаётся частным случаем при `k==len(children)` — переиспользуем без дублирования логики, вызывая
новую функцию всегда, т.к. `k=n` даёт идентичный результат).

## 4. Проверка

- `pareto_kth_order_moments` при `k=n` == `pareto_max_moments` (точная регрессия).
- `heterogeneous_kth_order_moments` при одинаковых Pareto-ветвях == `pareto_kth_order_moments`
  (точная регрессия, любые `k`).
- `heterogeneous_kth_order_moments` при `k=n` == `heterogeneous_max_moments` (регрессия к
  EPIC-030).
- Monte Carlo сэмплер (сортировка n гетерогенных сэмплов, взять k-й) — для смешанных семейств,
  `k<n`.
- Монотонность: `E[X_(k)]` не убывает по `k` (при фиксированных ветвях).
- `("parallel", children, k)` в `ForkJoinDAGCalc` — регрессия к `heterogeneous_kth_order_moments`
  напрямую и к предыдущему поведению (`k` не указан ⇒ `k=len(children)`, EPIC-030 без изменений).

## 5. Тесты

| Файл | Что проверяем |
|---|---|
| `tests/units/test_heterogeneous_max.py` (расширить) или новый `test_kth_order.py` | Точные регрессии §1/§2 (к `pareto_max_moments`, к `heterogeneous_max_moments`); MC для смешанных семейств и `k<n`; монотонность по `k`. |
| `tests/units/test_fork_join_dag.py` (расширить) | `("parallel", children, k)` — регрессия и MC для составного (n,k)-узла внутри DAG. |

## 6. Документация

- `docs/models/fork-join.md`+`.ru.md` — новая подсекция «(n,k)-fork-join поверх гетерогенных
  ветвей», пример `("parallel", children, k)`.
- `docs/models.md`/`.ru.md` — обновить описание/строку.

## 7. Резерв

- Purging vs non-purging семантика после k-го завершения (что происходит с оставшимися `n-k`
  ветвями) — теоретическая сторона здесь агностична (считает только время join), только DES уже
  различает варианты для однородного случая (`ForkJoinSim`).

## 8. Оценка трудозатрат

| Этап | Сложность | Срок (чел.-дней) |
|---|---|---|
| 1. `pareto_kth_order_moments` (замкнутая форма) | низкая-средняя | 0.5 |
| 2. `heterogeneous_kth_order_moments` (DP + квадратура) | средняя | 1 |
| 3. `("parallel", ..., k)` в `ForkJoinDAGCalc` | низкая | 0.5 |
| 4. Тесты (регрессии + MC) | низкая-средняя | 1 |
| 5. Документация | низкая | 0.25 |
| **Итого** | | **3.25–3.5** |

---

**Следующий шаг:** `pareto_kth_order_moments` в `most_queue/theory/utils/max_dist.py` по §1.
