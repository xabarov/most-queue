# Roadmap: точный max n·Pareto для fork-join/split-join

> Источники:
> - Boxma O. и др., *Fork–join and redundancy systems with heavy-tailed job sizes*, Queueing
>   Systems, 2022, doi:10.1007/s11134-022-09856-6.
> - *Extreme values for the waiting time in large fork-join queues*, Queueing Systems, 2025,
>   doi:10.1007/s11134-025-09937-2.
> - Полный список — `docs/research/fork-join-heavy-tail-2026.md`.

## 1. Цель

Добавить точный (не аппроксимационный) расчёт максимума n независимых Pareto(α, K)-распределённых
времён обслуживания подзадач и подключить его к уже существующему `SplitJoinCalc` +
DES-кросс-валидации (`ForkJoinSim`, который уже умеет `"Pa"`). Никакой новой теории поверх очереди
не выводим — только точная замена light-tailed аппроксимации `MaxDistribution` на закрытую форму
там, где она существует.

## 2. Математика (вывод, не из вторички)

Pareto(α, K): `F(x) = 1 - (K/x)^α`, `x ≥ K`. Замена `U = F(X) ~ Uniform(0,1)` даёт
`X = K·(1-U)^{-1/α}`. Максимум n iid Pareto соответствует максимуму n iid Uniform(0,1) =: `U_(n)`,
плотность `f_{U_(n)}(u) = n·u^{n-1}` на `(0,1)`.

**Точная CDF максимума:**
```
P(max_n > x) = 1 - F(x)^n = 1 - (1 - (K/x)^α)^n,  x ≥ K
```
(прямо из независимости — не требует замены переменных, но замена нужна для моментов).

**Точные сырые моменты:**
```
E[max_n^k] = E[K^k · (1-U_(n))^{-k/α}]
           = K^k · ∫_0^1 (1-u)^{-k/α} · n·u^{n-1} du
           = K^k · n · B(n, 1 - k/α)
```
где `B(a,b) = Γ(a)Γ(b)/Γ(a+b)` — Beta-функция. Интеграл (и момент) существует только при
`1 - k/α > 0`, т.е. **`k < α`** — то же условие существования моментов, что и у исходного Pareto,
не зависит от n. Проверка при `n=1`: `B(1, 1-k/α) = Γ(1)Γ(1-k/α)/Γ(2-k/α) = 1/(1-k/α)`, значит
`E[X^k] = K^k·1·1/(1-k/α) = K^k·α/(α-k)` — совпадает с формулой `ParetoDistribution.calc_theory_moments`
(`a * k^i / (a-i)`, с `i=k`). Сходится.

## 3. API

`most_queue/theory/utils/max_dist.py`, новые функции (не методы `MaxDistribution` — она заточена
под аппроксимации по моментам произвольного распределения, Pareto же не нужно приближать):

```python
def pareto_max_moments(params: ParetoParams, n: int, num: int) -> list[float]:
    """E[max_n^k] for k=1..num, exact via the Beta function. Raises ValueError
    at the first k >= alpha (the moment does not exist), rather than the
    silent truncation ParetoDistribution.calc_theory_moments uses -- this
    function must return exactly `num` moments for its callers (MaxDistribution-
    style consumers expect a fixed-length list), so a clear failure beats a
    silently short list."""

def pareto_max_tail(params: ParetoParams, n: int, x: float) -> float:
    """P(max_n > x), exact closed form, valid for any x >= K, any n, any alpha
    (unlike the moments, the tail/CDF is always well-defined)."""
```

Реализация через `math.gamma`/`math.lgamma` (избегаем overflow для больших n через
`math.exp(math.lgamma(...))`-комбинацию, а не `math.gamma` напрямую — n может быть десятки/сотни).

## 4. Интеграция в `SplitJoinCalc`

`theory/fork_join/split_join.py`:
- `calc_params.approx_distr` получает третье значение `"pareto"`.
- В режиме `"pareto"` `set_servers` принимает `ParetoParams` напрямую (не список моментов — в
  отличие от gamma/h2/erlang, где сервер задаётся моментами; Pareto лучше задавать параметрами,
  т.к. это точный, а не приближённый путь).
- `get_v()`: если `approximation == "pareto"`, считать `b_max = pareto_max_moments(params, n,
  num=4)`, дальше как сейчас — `MG1Calc.set_servers(b_max)` (точный P-K поверх точного максимума).
  Если `params.alpha <= 2` — **не пытаться** тихо продолжить с усечённым списком моментов (P-K
  нужна минимум дисперсия, т.е. `num=2`); поднять понятный `ValueError`, объясняющий, что M/G/1
  P-K неприменим при бесконечной дисперсии, и указать на `pareto_max_tail` как точную альтернативу
  для хвоста максимума подзадач (без сквозного времени отклика очереди — это резерв, см. §6).

## 5. Тесты

| Файл | Что проверяем |
|---|---|
| `tests/units/test_max_dist_pareto.py` | `pareto_max_moments`/`pareto_max_tail`: `n=1` сводится к `ParetoDistribution`; Beta-формула сверена с `scipy.integrate.quad` численным интегрированием тех же моментов для нескольких `(n, α)`; `ValueError` при `k >= α`; тривиальные свойства CDF (монотонность, `P(max>K)=1`, `P(max>∞)→0`). |
| `tests/test_fork_join_heavy_tail_vs_sim.py` | `SplitJoinCalc(approximation="pareto")` при `α>2` против `ForkJoinSim(..., is_sj=True)` с `set_servers(pareto_params, "Pa")` — моменты sojourn в пределах `MOMENTS_RTOL`. Плюс: `pareto_max_tail` против эмпирического хвоста максимума одного «раунда» подзадач через прямую выборку (не через полную очередь) — санити-чек независимо от M/G/1-слоя. |

## 6. Резерв (не в этой волне)

- **«One big jump» асимптотика полного времени отклика очереди при α≤2** (Boxma et al. 2022) —
  требует аккуратного воспроизведения асимптотических формул из первоисточника (не выводится с
  нуля без риска тонкой ошибки); отдельная будущая задача.
- **(n,k)-fork-join порядковые статистики для Pareto** (не только максимум = (n,n)) — тот же
  Beta-подход обобщается на `E[X_(k)^m]` через неполную Beta-функцию `I_{1-u}`, но требует
  отдельного вывода и не является先 priority этой волны.
- **Extreme-value предельные распределения при n→∞** (QS 2025) — асимптотическая теория, не точный
  калькулятор; не соответствует нише проекта.

## 7. Оценка трудозатрат

| Этап | Сложность | Срок (чел.-дней) |
|---|---|---|
| 1. `pareto_max_moments`/`pareto_max_tail` + юнит-тесты | низкая | 1 |
| 2. Интеграция в `SplitJoinCalc` | низкая-средняя | 1 |
| 3. DES-кросс-валидация | низкая | 1 |
| 4. Документация | низкая | 0.5 |
| **Итого** | | **3.5** |

---

**Следующий шаг:** реализовать `pareto_max_moments`/`pareto_max_tail` в
`most_queue/theory/utils/max_dist.py` + юнит-тесты.
