# Справочник по распределениям

[🇬🇧 English version](distributions.md)

Библиотека Most-Queue поддерживает различные распределения для моделирования времени поступления заявок и времени обслуживания. Этот справочник описывает все поддерживаемые распределения и способы их использования.

## Поддерживаемые распределения

| Распределение | Обозначение Кендалла | Параметры |
|--------------|---------------------|-----------|
| Экспоненциальное | M | μ (интенсивность) |
| Гиперэкспоненциальное 2-го порядка | H | H2Params |
| Эрланга | E | ErlangParams |
| Гамма | Gamma | GammaParams |
| Кокса 2-го порядка | C | Cox2Params |
| Парето | Pa | ParetoParams |
| Детерминированное | D | b (постоянное значение) |
| Равномерное | Uniform | UniformParams |
| Нормальное (Гауссово) | Norm | GaussianParams |

## Экспоненциальное распределение (M)

**Обозначение Кендалла:** M (Markovian)

Экспоненциальное распределение используется для моделирования пуассоновского потока и экспоненциального обслуживания.

### Параметры

- **`mu`** (float) — интенсивность (параметр экспоненциального распределения)

### Использование в симуляции

```python
from most_queue.sim.base import QsSim

qs = QsSim(num_of_channels=1)

# Поток поступления с интенсивностью λ = 0.5
qs.set_sources(0.5, "M")

# Обслуживание с интенсивностью μ = 1.0
qs.set_servers(1.0, "M")
```

### Использование в расчетах

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

calc = MMnrCalc(n=1)
calc.set_sources(l=0.5)  # интенсивность потока
calc.set_servers(mu=1.0)  # интенсивность обслуживания
```

### Характеристики

- Среднее: 1/μ
- Дисперсия: 1/μ²
- Коэффициент вариации: 1.0

## Гиперэкспоненциальное распределение 2-го порядка (H)

**Обозначение Кендалла:** H

Гиперэкспоненциальное распределение представляет собой смесь двух экспоненциальных распределений. Полезно для моделирования распределений с коэффициентом вариации > 1.

### Параметры (H2Params)

```python
from most_queue.random.utils.params import H2Params

h2_params = H2Params(
    p1=0.3,    # вероятность первой компоненты (0 < p1 < 1)
    mu1=1.0,   # интенсивность первой компоненты
    mu2=2.0    # интенсивность второй компоненты
)
```

### Создание по среднему и CV

```python
from most_queue.random.distributions import H2Distribution

# Создание параметров по среднему и коэффициенту вариации
h2_params = H2Distribution.get_params_by_mean_and_cv(
    mean=2.0,   # среднее значение
    cv=1.5      # коэффициент вариации (должен быть >= 1)
)
```

### Использование

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import H2Distribution, H2Params

qs = QsSim(num_of_channels=1)

# Создание параметров
h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=1.2)

# Использование в симуляции
qs.set_sources(0.5, "M")
qs.set_servers(h2_params, "H")

# Для расчетов нужны моменты
b = H2Distribution.calc_theory_moments(h2_params, num=5)
```

## Гамма-распределение (Gamma)

**Обозначение Кендалла:** Gamma

Гамма-распределение — гибкое распределение, которое может моделировать различные формы (включая экспоненциальное как частный случай).

### Параметры (GammaParams)

```python
from most_queue.random.utils.params import GammaParams

gamma_params = GammaParams(
    alpha=2.0,  # параметр формы (shape)
    mu=1.0      # параметр масштаба (rate)
)
```

### Создание по среднему и CV

```python
from most_queue.random.distributions import GammaDistribution

gamma_params = GammaDistribution.get_params_by_mean_and_cv(
    mean=2.0,   # среднее значение
    cv=0.6      # коэффициент вариации
)
```

### Использование

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import GammaDistribution

qs = QsSim(num_of_channels=1)

# Создание параметров
gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.7)

# Симуляция
qs.set_sources(0.5, "M")
qs.set_servers(gamma_params, "Gamma")

# Для расчетов
b = GammaDistribution.calc_theory_moments(gamma_params, num=5)
```

## Распределение Эрланга (E)

**Обозначение Кендалла:** E

Распределение Эрланга — это сумма k независимых экспоненциальных распределений с одинаковым параметром. Полезно для моделирования распределений с CV < 1.

### Параметры (ErlangParams)

```python
from most_queue.random.utils.params import ErlangParams

erlang_params = ErlangParams(
    k=3,        # число фаз (целое число >= 1)
    mu=1.0      # интенсивность каждой фазы
)
```

### Создание по среднему и CV

```python
from most_queue.random.distributions import ErlangDistribution

erlang_params = ErlangDistribution.get_params_by_mean_and_cv(
    mean=2.0,   # среднее значение
    cv=0.5      # коэффициент вариации (должен быть <= 1)
)
```

### Использование

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import ErlangDistribution

qs = QsSim(num_of_channels=1)

erlang_params = ErlangDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
qs.set_sources(0.5, "M")
qs.set_servers(erlang_params, "E")
```

## Распределение Кокса 2-го порядка (C)

**Обозначение Кендалла:** C

Распределение Кокса — это двухфазное распределение, которое может моделировать широкий диапазон коэффициентов вариации.

### Параметры (Cox2Params)

```python
from most_queue.random.utils.params import Cox2Params

cox_params = Cox2Params(
    p1=0.4,     # вероятность перехода на вторую фазу
    mu1=1.0,    # интенсивность первой фазы
    mu2=2.0     # интенсивность второй фазы
)
```

### Использование

```python
from most_queue.sim.base import QsSim
from most_queue.random.utils.params import Cox2Params

qs = QsSim(num_of_channels=1)

cox_params = Cox2Params(p1=0.4, mu1=1.0, mu2=2.0)
qs.set_sources(0.5, "M")
qs.set_servers(cox_params, "C")
```

## Распределение Парето (Pa)

**Обозначение Кендалла:** Pa

Распределение Парето используется для моделирования тяжелых хвостов распределений.

### Параметры (ParetoParams)

```python
from most_queue.random.utils.params import ParetoParams

pareto_params = ParetoParams(
    alpha=2.0,   # параметр формы
    K=1.0        # минимальное значение
)
```

### Создание по среднему и CV

```python
from most_queue.random.distributions import ParetoDistribution

pareto_params = ParetoDistribution.get_params_by_mean_and_cv(
    mean=2.0,
    cv=1.5
)
```

## Детерминированное распределение (D)

**Обозначение Кендалла:** D

Детерминированное распределение — постоянное значение без случайности.

### Параметры

- **`b`** (float) — постоянное значение

### Использование

```python
from most_queue.sim.base import QsSim

qs = QsSim(num_of_channels=1)

# Постоянный интервал между заявками
qs.set_sources(2.0, "D")  # интервал = 2.0

# Постоянное время обслуживания
qs.set_servers(3.0, "D")  # время обслуживания = 3.0
```

## Равномерное распределение (Uniform)

**Обозначение Кендалла:** Uniform

Равномерное распределение на интервале [a, b].

### Параметры (UniformParams)

```python
from most_queue.random.utils.params import UniformParams

uniform_params = UniformParams(
    mean=2.0,        # среднее значение
    half_interval=1.0  # половина интервала (b - a) / 2
)
```

### Использование

```python
from most_queue.sim.base import QsSim
from most_queue.random.utils.params import UniformParams

qs = QsSim(num_of_channels=1)

uniform_params = UniformParams(mean=2.0, half_interval=1.0)
qs.set_sources(0.5, "M")
qs.set_servers(uniform_params, "Uniform")
```

## Нормальное распределение (Norm)

**Обозначение Кендалла:** Norm

Нормальное (Гауссово) распределение. **Внимание:** может давать отрицательные значения, что не имеет физического смысла для времени.

### Параметры (GaussianParams)

```python
from most_queue.random.utils.params import GaussianParams

gaussian_params = GaussianParams(
    mean=2.0,    # среднее значение
    std=0.5      # стандартное отклонение
)
```

### Использование

```python
from most_queue.sim.base import QsSim
from most_queue.random.utils.params import GaussianParams

qs = QsSim(num_of_channels=1)

gaussian_params = GaussianParams(mean=2.0, std=0.5)
qs.set_sources(0.5, "M")
qs.set_servers(gaussian_params, "Norm")
```

## Фазовые распределения (PH)

**Нотация Кендалла:** PH

![Знакомые распределения как фазовые цепочки](figures/ph_gallery.ru.png)

**Простыми словами:** если вы работали с экспоненциальным, Эрланга, H₂ или Кокса — вы уже
знаете фазовые распределения: всё это их частные случаи. PH-величина — это время блуждания
«фишки» по цепочке экспоненциальных **фаз** до выхода: стартовая фаза разыгрывается по вектору
**α**, переходы между фазами задаёт суб-генератор **T** (интенсивности выхода — `t0 = -T @ 1`).
Выстроите фазы в цепочку — получите Эрланга; поставите параллельно — H₂; разрешите досрочный
выход — Кокса. Произвольная пара (α, T) даёт семейство, которым можно приблизить любое
положительное распределение.

| Знакомое распределение | Структура PH | Конвертер |
|---|---|---|
| Экспоненциальное(μ) | одна фаза | `PHDistribution.from_exp(mu)` |
| Эрланга-r | цепочка из r одинаковых фаз | `PHDistribution.from_erlang(params)` |
| H₂ | две параллельные фазы | `PHDistribution.from_h2(params)` |
| Кокса-2 | цепочка с досрочным выходом | `PHDistribution.from_cox(params)` |

Начальные моменты — в замкнутой форме: `m_k = k! · α (−T)⁻ᵏ 1`.

### Как читать матрицу T

В T записаны **интенсивности** (размерность 1/время), а не вероятности — поэтому строки
НЕ суммируются в 1:

- **Вне диагонали** `T[i][j] ≥ 0` — интенсивность перехода из фазы i в фазу j.
- **Диагональ** `T[i][i] < 0` — минус СУММАРНАЯ скорость ухода из фазы i (включая выход
  из распределения), поэтому время пребывания в фазе i ~ Exp(−T[i][i]).
- **Суммы строк ≤ 0**, а «недостача до нуля» — это интенсивность выхода наружу:
  `t0 = −T @ 1`. Нулевая сумма строки означает «прямого выхода из этой фазы нет».

Просидев в фазе i время Exp(−T[i][i]), фишка выбирает направление с вероятностями
`T[i][j] / (−T[i][i])` (в фазу j) или `t0[i] / (−T[i][i])` (выход) — вероятности
*вычисляются* из интенсивностей, в T они не хранятся. Знакомые структуры в матричном виде:

```text
H2(p1, mu1, mu2):  alpha = [p1, 1-p1]   T = [[-mu1,   0 ],    t0 = [mu1]
                                             [  0,  -mu2]]         [mu2]

Эрланг-3(mu):      alpha = [1, 0, 0]    T = [[-mu,  mu,   0]   t0 = [0]
                                             [  0, -mu,  mu]        [0]
                                             [  0,   0, -mu]]       [mu]

Кокса-2:           alpha = [1, 0]       T = [[-mu1, p1*mu1]    t0 = [(1-p1)*mu1]
                                             [  0,   -mu2 ]]        [   mu2    ]
```

`PHDistribution` проверяет всё это при создании (отрицательная диагональ, неотрицательные
внедиагональные элементы, суммы строк ≤ 0, α суммируется в 1). Матрицы MAP ниже устроены
по той же логике: роль T играет D₀, а пара должна удовлетворять условию «выхода нет вообще» —
у (D₀ + D₁) нулевые суммы строк, потому что MAP никогда не останавливается,
а только порождает заявки.

### Использование

```python
import numpy as np
from most_queue.random.map_ph import PHDistribution, PHParams
from most_queue.random.utils.params import ErlangParams

# готовый конвертер...
ph = PHDistribution.from_erlang(ErlangParams(r=3, mu=2.0))

# ...или произвольная пара (alpha, T): Эрланг-2 с замедленной второй фазой
custom = PHParams(alpha=np.array([1.0, 0.0]), T=np.array([[-2.0, 2.0], [0.0, -0.7]]))

moments = PHDistribution.calc_theory_moments(custom, 4)
f_at_1 = PHDistribution.get_pdf(custom, 1.0)

# в симуляции PH работает и как источник, и как обслуживание
from most_queue.sim.base import QsSim
qs = QsSim(1)
qs.set_sources(1.0, "M")
qs.set_servers(custom, "PH")
```

**Зачем, если есть H₂/Эрланг?** PH — «язык обслуживания» матрично-аналитических решателей:
[калькулятор MAP/PH/1](models/map-ph.ru.md) принимает любое PH, так что одна модель покрывает всё —
от почти детерминированного (длинные цепочки Эрланга) до высоковариативного (H₂) обслуживания
одним точным методом.

## Марковский входной поток (MAP)

**Нотация Кендалла:** MAP

![Механика MMPP](figures/mmpp.ru.png)

**Простыми словами:** все потоки выше — *рекуррентные* (renewal): интервалы между приходами
независимы, монетка без памяти. Реальный трафик обычно **пульсирует**: за коротким интервалом
чаще следует короткий. MAP добавляет ровно эту память: фоновая марковская цепь блуждает по
фазам, а заявки генерируются с интенсивностью, зависящей от фазы. Задают его две матрицы:
**D₀** — переходы фаз *без* заявки, **D₁** — переходы, *порождающие* заявку.

Простейший содержательный пример — **MMPP** (Markov-modulated Poisson process) на схеме:
быстрая и медленная фазы с редкими переключениями — поток чередует всплески и затишья
при неизменной средней интенсивности.

Частные случаи: однофазный MAP — пуассоновский поток; `D1 = t0 @ alpha` — рекуррентный поток
с PH-интервалами (без корреляции). Моменты интервалов и **лаг-k автокорреляция** — в замкнутой
форме.

### Использование

```python
import numpy as np
from most_queue.random.map_ph import MAP, PHDistribution

# MMPP-2: интенсивность 2.0 в фазе 1, 0.4 в фазе 2, редкие переключения
mmpp = MAP.mmpp([2.0, 0.4], np.array([[-0.2, 0.2], [0.3, -0.3]]))

print(MAP.arrival_rate(mmpp))            # средняя интенсивность
print(MAP.calc_theory_moments(mmpp, 3))  # моменты интервалов
print(MAP.lag_correlation(mmpp, 1))      # пульсация, которую не видят renewal-модели

# другие фабрики
poisson = MAP.poisson(1.0)
renewal_h2 = MAP.from_ph_renewal(PHDistribution.from_exp(1.0))

# в симуляции MAP — источник с состоянием
from most_queue.sim.base import QsSim
qs = QsSim(1)
qs.set_sources(mmpp, "MAP")
qs.set_servers(1.5, "M")
```

**Почему это важно:** при тех же загрузке, среднем и CV интервалов положительная корреляция
может умножить среднее ожидание в разы — см. точную модель
[MAP/PH/1](models/map-ph.ru.md) и демо-ноутбук
[`tutorials/map_ph_correlation.ipynb`](../tutorials/map_ph_correlation.ipynb).

### Подгонка MAP по данным

Обычно у вас нет готового MAP — есть поток измеренных интервалов между приходами. Хелпер
`most_queue.random.map_fit` подгоняет двухфазный MMPP к трём статистикам: интенсивности,
квадрату коэффициента вариации (SCV) и лаг-1 автокорреляции. MMPP всегда переразбросан и
положительно коррелирован, поэтому требует **SCV ≥ 1** и **лаг-1 ≥ 0** (иначе — понятная
ошибка).

```python
from most_queue.random.map_fit import fit_mmpp2, fit_map_from_trace, map_statistics

# подгонка к целевым статистикам...
mmpp = fit_mmpp2(rate=1.0, scv=3.0, lag1=0.2)

# ...или прямо из трейса интервалов
mmpp = fit_map_from_trace(interarrival_samples)

print(map_statistics(mmpp))  # (rate, scv, lag1) — что фактически получилось
```

Полученный `MAPParams` напрямую подставляется в модели MAP/PH/1, MAP/M/c и симулятор выше.

## Получение моментов распределений

Для использования в численных методах расчета нужны моменты распределений:

```python
from most_queue.random.distributions import (
    H2Distribution,
    GammaDistribution,
    ErlangDistribution
)

# H2-распределение
h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=1.2)
b = H2Distribution.calc_theory_moments(h2_params, num=5)
# b[0] - среднее, b[1] - второй момент, b[2] - третий момент, и т.д.

# Гамма-распределение
gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.7)
b = GammaDistribution.calc_theory_moments(gamma_params, num=5)

# Распределение Эрланга
erlang_params = ErlangDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.5)
b = ErlangDistribution.calc_theory_moments(erlang_params, num=5)
```

## Выбор распределения

### По коэффициенту вариации

- **CV < 1**: Распределение Эрланга (E) или Гамма (Gamma)
- **CV = 1**: Экспоненциальное (M)
- **CV > 1**: Гиперэкспоненциальное (H) или Гамма (Gamma)

### По характеру данных

- **Пуассоновский поток**: Экспоненциальное (M)
- **Регулярное поступление**: Детерминированное (D) или Эрланга (E)
- **Высокая вариативность**: Гиперэкспоненциальное (H) или Парето (Pa)
- **Универсальное**: Гамма (Gamma) или Кокса (C)

## Пример: сравнение распределений

```python
from most_queue.sim.base import QsSim
from most_queue.random.distributions import (
    H2Distribution,
    GammaDistribution,
    ErlangDistribution
)

arrival_rate = 0.4
service_mean = 2.5
service_cv = 0.8
num_jobs = 30000

# H2-распределение
h2_params = H2Distribution.get_params_by_mean_and_cv(service_mean, service_cv)
qs_h2 = QsSim(1)
qs_h2.set_sources(arrival_rate, "M")
qs_h2.set_servers(h2_params, "H")
results_h2 = qs_h2.run(num_jobs)

# Гамма-распределение
gamma_params = GammaDistribution.get_params_by_mean_and_cv(service_mean, service_cv)
qs_gamma = QsSim(1)
qs_gamma.set_sources(arrival_rate, "M")
qs_gamma.set_servers(gamma_params, "Gamma")
results_gamma = qs_gamma.run(num_jobs)

# Сравнение
print(f"H2: среднее время ожидания = {results_h2.w[0]:.4f}")
print(f"Gamma: среднее время ожидания = {results_gamma.w[0]:.4f}")
```

## Список поддерживаемых распределений

Для получения списка всех поддерживаемых распределений:

```python
from most_queue.random.distributions import print_supported_distributions

print_supported_distributions()
```

---

**См. также:**
- [Симуляция СМО](simulation.ru.md) — использование распределений в симуляции
- [Численные методы](calculation.ru.md) — работа с моментами распределений
- [Примеры использования](examples.ru.md) — практические примеры

