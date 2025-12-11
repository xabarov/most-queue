# Справочник по распределениям

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
- [Симуляция СМО](simulation.md) — использование распределений в симуляции
- [Численные методы](calculation.md) — работа с моментами распределений
- [Примеры использования](examples.md) — практические примеры

