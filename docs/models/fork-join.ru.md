# Fork-Join системы

[🇬🇧 English version](fork-join.md) · [← Каталог моделей](../models.ru.md)

![Схема Fork-Join](../figures/fork_join.ru.png)

**Простыми словами:** заявка при входе разделяется (fork) на несколько частей, части
обслуживаются параллельно на разных приборах, а результат готов только когда собраны все
части (join). Так устроены параллельные вычисления, RAID-массивы, распределённые запросы
(map-reduce). Время ответа определяет *самая медленная* часть — поэтому среднее время
пребывания растёт с числом частей даже при той же суммарной работе.

### M/M/c Fork-Join

**Описание:** Система, где заявка разделяется на несколько частей, обслуживаемых параллельно, и затем объединяется.

**Класс расчета:** `ForkJoinMarkovianCalc`

**Пример:**

```python
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc

calc = ForkJoinMarkovianCalc(n=5, k=2)  # 5 каналов, требуется 2
calc.set_sources(l=1.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/G/c Split-Join

**Описание:** Система Split-Join с произвольным распределением времени обслуживания.

**Суть:** строгий вариант Fork-Join — следующая заявка не начинает обслуживаться, пока
полностью не собрана предыдущая (синхронный конвейер партий).

**Класс расчета:** `SplitJoinCalc`

**Пример:** См. тест `test_fj_sim.py`

### Тяжёлые хвосты подзадач (Pareto) — точно, без аппроксимации

**Простыми словами:** light-tailed путь выше приближает максимум n времён обслуживания подзадач
подгонкой H2/Gamma/Erlang по моментам и численным интегрированием (квадратура Гаусса–Лагерра). Для
**Pareto**-распределённых подзадач эта аппроксимация и не нужна, и неверна — максимум n
независимых Pareto(α, K) имеет точную замкнутую форму (замена `U = F(X) ~ Uniform(0,1)` превращает
максимум n Pareto в максимум n Uniform, чьи моменты — интеграл через Beta-функцию):

```
P(max_n > x) = 1 - (1 - (K/x)^α)^n                (точный хвост/CDF, любой α > 0)
E[max_n^k]   = K^k · n · B(n, 1 - k/α)             (точные моменты, только при k < α)
```

Моменты порядка ≥ α не существуют (как и у одного Pareto) — `pareto_max_moments` кидает понятную
ошибку, а не тихо возвращает `NaN`/`inf`. `SplitJoinCalc(approximation="pareto")` сшивает этот
точный максимум с точной формулой Поллачека–Хинчина M/G/1 для полного времени пребывания —
корректно при α > 2 (конечная дисперсия, нужна для P-K). При α ≤ 2 используйте `pareto_max_tail`
напрямую для точного хвоста самой медленной подзадачи; асимптотика полного времени отклика очереди
в режиме бесконечной дисперсии («one big jump» — Boxma и др., Queueing Systems 2022) пока не
реализована, см. [`docs/research/fork-join-heavy-tail-2026.md`](../research/fork-join-heavy-tail-2026.md).

**Функции/классы:** `pareto_max_moments`, `pareto_max_tail`
(`most_queue.theory.utils.max_dist`); `SplitJoinCalc(calc_params=CalcParams(approx_distr="pareto"))`

**Пример:**

```python
from most_queue.random.utils.params import ParetoParams
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fork_join.split_join import SplitJoinCalc

calc = SplitJoinCalc(n=4, calc_params=CalcParams(approx_distr="pareto"))
calc.set_sources(l=0.3)
calc.set_servers(ParetoParams(alpha=3.5, K=1.0))   # alpha > 2: конечная дисперсия
results = calc.run()   # точные моменты времени пребывания
```

**См. также:** [SLA / вероятность нарушения дедлайна](sla.ru.md) — её подгонка H2/Gamma
предполагает конечную дисперсию и **неверна** при α ≤ 2; используйте вместо неё точный
`pareto_max_tail` для дедлайнов в тяжёлохвостом fork-join.

### Гетерогенные ветви и series-parallel DAG задач

**Простыми словами:** все модели выше предполагают ОДИНАКОВОЕ распределение времени обслуживания
у каждой ветви. Реальные графы задач редко такие — mapper'ы MapReduce работают на шардах разного
размера, вызов микросервиса веерно расходится по эндпоинтам с разным профилем задержки. Этот
раздел снимает предположение «ветви одинаковы» в двух независимых направлениях: разные
распределения ветвей на одном уровне fork-join, и граф задач (вложенная series/parallel
композиция, а не один плоский уровень).

**Описание:** `n` независимых ветвей, каждая со своим семейством/параметрами (любая комбинация
Pareto/Gamma/H2/Erlang из библиотеки). Замкнутой формы в общем случае нет (в отличие от i.i.d.
Pareto выше), но моменты всё равно численно точны при заданном семействе каждой ветви — через
стандартное тождество `E[max^k] = k∫t^(k-1)P(max>t)dt`, `P(max>t) = 1 - Π(1-tail_i(t))`,
вычисляемое через `scipy.integrate.quad`. Точно сводится к `pareto_max_moments` при одинаковых
Pareto-ветвях (регрессионный тест).

**Функции:** `heterogeneous_max_moments`, `branch_tail` (`most_queue.theory.utils.max_dist`)

```python
from most_queue.theory.utils.max_dist import heterogeneous_max_moments
from most_queue.random.utils.params import ParetoParams

branches = [
    ("pareto", ParetoParams(alpha=5.0, K=1.0)),
    ("pareto", ParetoParams(alpha=6.0, K=1.5)),
    ("gamma", [2.0, 6.0, 24.0]),   # моменты -- подгоняются внутри
]
moments = heterogeneous_max_moments(branches, num=3)  # E[max], E[max^2], E[max^3]
```

**Композиция DAG:** `ForkJoinDAGCalc` рекурсивно собирает небольшую спецификацию из вложенных
кортежей — `("leaf", family, spec)`, `("series", [node, ...])` (последовательно, сумма через
`conv_moments`), `("parallel", [node, ...])` (fork-join, максимум через
`heterogeneous_max_moments`) — покрывает известный класс **series-parallel-reducible** графов
задач (стадии MapReduce, цепочки вызовов микросервисов, CI/CD-пайплайны — подавляющее большинство
реальных графов задач; см.
[`docs/research/fork-join-dag-heterogeneous-2026.md`](../research/fork-join-dag-heterogeneous-2026.md)).
Точно — для узла `parallel`, чьи дети являются «сырыми» листьями; узел `parallel`, чьи дети сами
являются составными поддеревьями, требует один дополнительный шаг подгонки распределения по
моментам (тот же стандарт, который `SplitJoinCalc` уже использует для не-Pareto i.i.d. случая — не
новый источник неточности, проверено против прямого Monte Carlo сэмплирования графа). Моменты
корня — время обслуживания одной заявки в Split-Join (блокирующей) очереди, обёрнутое в точную
формулу M/G/1 (Поллачек–Хинчин).

```python
from most_queue.theory.fork_join.dag import ForkJoinDAGCalc
from most_queue.random.utils.params import ParetoParams

dag = (
    "series",
    [
        ("parallel", [("leaf", "pareto", ParetoParams(alpha=6.0, K=1.0)), ("leaf", "gamma", [2.0, 5.0, 14.0])]),
        ("parallel", [("leaf", "pareto", ParetoParams(alpha=7.0, K=1.2)), ("leaf", "pareto", ParetoParams(alpha=6.0, K=0.8))]),
    ],
)
calc = ForkJoinDAGCalc(dag)
calc.set_sources(l=0.1)
res = calc.run()   # res.v, res.w -- моменты пребывания/ожидания всего DAG
```

**Не покрыто (резерв):** общие (не series-parallel) precedence-DAG доказанно #P-hard для точного
решения в общем случае (Dodin, Operations Research, 1985) — нужны bounding-техники или Monte
Carlo, не точный/квази-точный калькулятор.

### (n,k)-fork-join поверх гетерогенных ветвей

**Простыми словами:** все модели выше требуют завершения КАЖДОЙ ветви (n-из-n). На практике часто
достаточно ЧАСТИ: реплицированному чтению достаточно ответа самого быстрого кворума, записи с
RAID/erasure coding достаточно, когда записано достаточно шардов, фаза reduce в MapReduce со
спекулятивным исполнением ждёт не все избыточные копии mapper'ов, а достаточное число. Это
k-out-of-n join, здесь обобщённый на ветви, которые не обязаны быть одинаковыми.

**Описание:** k-я порядковая статистика (по возрастанию: `k=1` — самая быстрая/минимум, `k=n` —
самая медленная/максимум, т.е. модели выше) `n` независимых ветвей. `P(X_(k) <= x) = P(хотя бы k
из n ветвей завершились к x)` — стандартный факт теории порядковых статистик, и численно та же
задача, что надёжность k-out-of-n:G систем с гетерогенными компонентами: считается через
классический DP Rushdi (1985), O(n²), по Poisson-Binomial распределению от CDF каждой ветви в
точке `x`, дальше та же квадратура момента хвоста, что и у гетерогенного максимума выше (`k=n`
точно сводится к нему). Для i.i.d. Pareto-ветвей снова есть точная замкнутая форма
(`pareto_kth_order_moments`), обобщающая `pareto_max_moments` через то, что k-я порядковая
статистика Uniform(0,1) распределена как `Beta(k, n-k+1)` — используется здесь как регрессионный
якорь для численного гетерогенного пути. См.
[`docs/research/fork-join-nk-heterogeneous-2026.md`](../research/fork-join-nk-heterogeneous-2026.md).

**Функции:** `pareto_kth_order_moments`, `heterogeneous_kth_order_moments`
(`most_queue.theory.utils.max_dist`)

```python
from most_queue.theory.utils.max_dist import heterogeneous_kth_order_moments
from most_queue.random.utils.params import ParetoParams

branches = [
    ("pareto", ParetoParams(alpha=6.0, K=1.0)),
    ("pareto", ParetoParams(alpha=7.0, K=1.2)),
    ("gamma", [2.0, 6.0, 24.0]),
]
moments = heterogeneous_kth_order_moments(branches, k=2, num=2)  # 2-я из 3 завершившихся
```

Любой узел `("parallel", ...)` в DAG `ForkJoinDAGCalc` может принять необязательный третий
элемент — требуемое число `k` (по умолчанию `len(children)`, т.е. обычный fork-join):

```python
dag = ("parallel", [("leaf", "pareto", p1), ("leaf", "pareto", p2), ("leaf", "pareto", p3)], 2)
```
