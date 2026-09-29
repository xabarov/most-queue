# Roadmap: Fork-Join с гетерогенными ветвями и series-parallel DAG

> Источники — `docs/research/fork-join-dag-heterogeneous-2026.md`. База: EPIC-022
> (`docs/research/fork-join-heavy-tail-2026.md`, `theory/fork_join/split_join.py`,
> `theory/utils/max_dist.py`).

## 1. Гетерогенный максимум (новый строительный блок)

`n` независимых, но НЕ обязательно одинаково распределённых ветвей. Для ветви `i` — хвост
(survival function) `tail_i(t) = P(X_i > t)`. Тогда:

```
P(max > t) = 1 - Π_i (1 - tail_i(t))
E[max^k]   = k · ∫_0^∞ t^(k-1) · P(max > t) dt    (стандартная формула для X>=0)
```

Численно точно (не аппроксимация распределения максимума) при заданном семействе каждой ветви —
через `scipy.integrate.quad` (уже используется в библиотеке, напр. `theory/srpt/utils/predictor.py`).
При `n` одинаковых Pareto(α,K)-ветвей должно точно (в пределах точности `quad`) совпадать с
`pareto_max_moments` — основной regression-тест.

```python
def heterogeneous_max_moments(branches: list[tuple[str, Any]], num: int) -> list[float]:
    """branches[i] = (family, spec): family in {'pareto','gamma','h2','erlang'};
    spec = ParetoParams for 'pareto', raw moments (list[float]) otherwise (fit internally,
    как в MaxDistribution)."""
```

## 2. Series-parallel DAG композиция

Спецификация узла графа (рекурсивные вложенные кортежи):

```
("leaf", family, spec)              # один подзадача, распределение как в §1
("series", [node1, node2, ...])     # последовательно: sum-композиция через conv_moments
("parallel", [node1, node2, ...])   # параллельно (fork-join): max-композиция через §1
```

Композиция:

- `series`: `conv_moments(m1, m2, num)`, свёрнутый по всем детям (существующая утилита, точная
  формула для суммы независимых по моментам).
- `parallel`: если ВСЕ дети — узлы `leaf` (одноуровневая гетерогенная fork-join) —
  `heterogeneous_max_moments` напрямую по их `(family, spec)`, точно как в §1. Если хотя бы один
  ребёнок — составное поддерево (`series`/`parallel`), его уже посчитанные моменты нужно сначала
  подогнать (fit) под семейство (`gamma`/`h2`/`erlang`, как уже делает `MaxDistribution` для
  негетерогенного случая) — получаем `tail_i(t)` для промежуточного узла, дальше та же формула §1.
  **Это тот же fit-based шаг, что уже принят в `SplitJoinCalc` для не-Pareto случая** — не новый
  источник неточности, а перенос уже принятого в библиотеке стандарта на составные узлы.

Итоговые моменты корня графа — это моменты времени обслуживания ОДНОЙ заявки (Split-Join,
блокирующая семантика EPIC-022: новая заявка не входит, пока текущая не завершится полностью).
Дальше — существующая точная M/G/1 (Pollaczek–Khinchine) формула (`MG1Calc`) поверх этих моментов
— без нового численного метода на уровне очереди.

## 3. API

```python
class ForkJoinDAGCalc(BaseQueue):
    def __init__(self, dag: DAGNode, calc_params: CalcParams | None = None): ...
    def set_sources(self, l: float): ...
    def run(self) -> QueueResults: ...  # v, w, utilization — через MG1Calc поверх моментов DAG
```

`DAGNode` — вложенный кортеж/dataclass, как в §2.

## 4. Проверка (без нового DES-движка)

Split-Join семантика развязывает «распределение времени обслуживания DAG» и «динамику очереди»
— очередь уже проверена (M/G/1 против DES в других тестах). Поэтому:

- **Monte Carlo сэмплер DAG** (лёгкая функция, не полноценный DES): рекурсивно сэмплирует
  реализации по той же структуре узлов (`leaf` — сэмпл из распределения; `series` — сумма
  сэмплов детей; `parallel` — максимум сэмплов детей), считает эмпирические моменты по многим
  прогонам — валидирует ТОЛЬКО композиционную математику (`conv_moments`/
  `heterogeneous_max_moments`/fit-шаг), не очередь.
- Против этого MC-сэмплера тестируем: (а) плоский гетерогенный fork-join (2-3 разные Pareto/Gamma
  ветви); (б) двухуровневый DAG (`series` из `parallel` из `parallel`) — композиция через
  составные поддеревья, включая fit-шаг.
- `ForkJoinDAGCalc.run()` против `pareto_max_moments`-эквивалентного случая (когда DAG вырождается
  в плоский i.i.d. Pareto fork-join) — точная регрессия к EPIC-022.

## 5. Тесты

| Файл | Что проверяем |
|---|---|
| `tests/units/test_heterogeneous_max.py` (новый) | `heterogeneous_max_moments` с одинаковыми Pareto-ветвями == `pareto_max_moments` (регрессия); против MC-сэмплера для 2-3 разных семейств/параметров; монотонность (больше `n` → не меньше `E[max]`). |
| `tests/units/test_fork_join_dag.py` (новый) | Плоский DAG (`parallel` из `leaf`) сводится к `SplitJoinCalc`/`heterogeneous_max_moments`; двухуровневый DAG против MC-сэмплера; `ForkJoinDAGCalc.run()` даёт корректные `v`/`w`/`utilization` (через `MG1Calc`). |

## 6. Документация

- `docs/models/fork-join.md` (или расширение существующей секции) — новые подсекции: гетерогенный
  максимум, series-parallel DAG, честная граница точности (composite-узлы — fit-based).
- `docs/models.md`/`.ru.md` — обновить описание fork-join строки.

## 7. Резерв

- Общие (не series-parallel) DAG с произвольными precedence-ограничениями — доказанно #P-hard в
  общем случае (Dodin 1985); нужны bounding-техники или чистый MC, не точный/квази-точный
  калькулятор.
- (n,k)-fork-join (не все n ветвей обязательны) поверх гетерогенных/DAG-ветвей — сейчас
  `ForkJoinMarkovianCalc`/`SplitJoinCalc` это не покрывают для гетерогенного случая.

## 8. Оценка трудозатрат

| Этап | Сложность | Срок (чел.-дней) |
|---|---|---|
| 1. `heterogeneous_max_moments` (+ regression к Pareto) | средняя | 1 |
| 2. `ForkJoinDAGCalc` (рекурсивная композиция + fit-шаг) | средняя-высокая | 1.5–2 |
| 3. MC-сэмплер + тесты | низкая-средняя | 1 |
| 4. Документация | низкая | 0.25 |
| **Итого** | | **3.75–4.25** |

---

**Следующий шаг:** `heterogeneous_max_moments` в `most_queue/theory/utils/max_dist.py` по §1.
