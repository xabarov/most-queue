# Каталог поддерживаемых моделей СМО

Библиотека Most-Queue поддерживает широкий спектр моделей систем массового обслуживания. Этот каталог содержит описание всех доступных моделей с примерами использования.

## FIFO системы (дисциплина First In First Out)

### M/M/c

**Описание:** Многоканальная система с пуассоновским потоком и экспоненциальным обслуживанием.

**Класс расчета:** `MMnrCalc`

**Пример:**

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

calc = MMnrCalc(n=3)  # 3 канала
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/M/c/r

**Описание:** M/M/c с ограниченной очередью (максимум r мест в очереди).

**Класс расчета:** `MMnrCalc`

**Пример:**

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

calc = MMnrCalc(n=3, r=20)  # 3 канала, очередь до 20
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/G/1

**Описание:** Одноканальная система с пуассоновским потоком и произвольным распределением времени обслуживания.

**Класс расчета:** `MG1Calc`

**Пример:**

```python
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.random.distributions import H2Distribution

calc = MG1Calc()
calc.set_sources(l=0.5)

h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=0.8)
b = H2Distribution.calc_theory_moments(h2_params, 5)
calc.set_servers(b)

results = calc.run()
```

### GI/M/1

**Описание:** Одноканальная система с общим потоком поступления и экспоненциальным обслуживанием.

**Класс расчета:** `GIM1Calc`

**Пример:**

```python
from most_queue.theory.fifo.gi_m_1 import GIM1Calc
from most_queue.random.distributions import GammaDistribution

calc = GIM1Calc()

gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
a = GammaDistribution.calc_theory_moments(gamma_params)
calc.set_sources(a)

calc.set_servers(mu=0.6)
results = calc.run()
```

### GI/M/c

**Описание:** Многоканальная система с общим потоком поступления и экспоненциальным обслуживанием.

**Класс расчета:** `GiMn`

**Пример:**

```python
from most_queue.theory.fifo.gi_m_n import GiMn
from most_queue.random.distributions import GammaDistribution

calc = GiMn(n=3)  # 3 канала

gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean=2.0, cv=0.6)
a = GammaDistribution.calc_theory_moments(gamma_params)
calc.set_sources(a)

calc.set_servers(mu=0.6)
results = calc.run()
```

### M/D/c

**Описание:** Многоканальная система с пуассоновским потоком и детерминированным временем обслуживания.

**Класс расчета:** `MDnCalc`

**Пример:**

```python
from most_queue.theory.fifo.m_d_n import MDnCalc

calc = MDnCalc(n=3)
calc.set_sources(l=2.0)
calc.set_servers(b=1.0)  # постоянное время обслуживания
results = calc.run()
```

### E_k/D/c

**Описание:** Многоканальная система с распределением Эрланга межприходных времен и детерминированным обслуживанием.

**Класс расчета:** `EkDnCalc`

**Пример:**

```python
from most_queue.theory.fifo.ek_d_n import EkDnCalc

calc = EkDnCalc(n=3, k=2)  # 3 канала, Эрланга порядка 2
calc.set_sources(l=2.0)
calc.set_servers(b=1.0)
results = calc.run()
```

### M/H₂/c (метод Такахаси-Таками)

**Описание:** Многоканальная система с пуассоновским потоком и гиперэкспоненциальным обслуживанием. Использует численный метод Такахаси-Таками с комплексными параметрами.

**Класс расчета:** `MGnCalc`

**Пример:**

```python
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.random.distributions import H2Distribution

calc = MGnCalc(n=5)

calc.set_sources(l=2.0)

h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=1.2)
b = H2Distribution.calc_theory_moments(h2_params, 4)
calc.set_servers(b)

results = calc.run()
```

## Системы с приоритетами

### M/G/1/PR (прерываемый приоритет)

**Описание:** Одноканальная система с несколькими классами приоритетов. Приоритетные заявки могут прерывать обслуживание низкоприоритетных.

**Класс расчета:** `MG1Preemptive`

**Пример:**

```python
from most_queue.theory.priority.preemptive.mg1 import MG1Preemptive

calc = MG1Preemptive(num_of_classes=3)
calc.set_sources([0.1, 0.2, 0.3])  # интенсивности для каждого класса

# Моменты времени обслуживания для каждого класса
b = [
    [2.0, 4.0, 8.0],  # класс 1
    [3.0, 9.0, 27.0], # класс 2
    [4.0, 16.0, 64.0] # класс 3
]
calc.set_servers(b)

results = calc.run()
```

### M/G/1/NP (непрерываемый приоритет)

**Описание:** Одноканальная система с приоритетами, где начатое обслуживание не прерывается.

**Класс расчета:** `MG1NonPreemptive`

**Пример:**

```python
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptive

calc = MG1NonPreemptive(num_of_classes=3)
calc.set_sources([0.1, 0.2, 0.3])
calc.set_servers(b)  # моменты для каждого класса
results = calc.run()
```

### M/G/c/PR и M/G/c/NP

**Описание:** Многоканальные системы с приоритетами (прерываемый и непрерываемый).

**Класс расчета:** `MGnInvarApproximation`

**Пример:**

```python
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation

calc = MGnInvarApproximation(n=5, priority="PR")  # или "NP"
calc.set_sources([0.1, 0.2, 0.3])
calc.set_servers(b)
results = calc.run()
```

### M/Ph/c/PR

**Описание:** Многоканальная система с фазовым распределением времени обслуживания и приоритетами.

**Класс расчета:** `MPhNPrtyBusyApprox`

**Пример:** См. тест `test_m_ph_n_prty.py`

## Системы с отпусками (Vacations)

### M/H₂/c с прогревом

**Описание:** Многоканальная система с гиперэкспоненциальным обслуживанием и прогревом каналов.

**Класс расчета:** `MH2H2WarmCalc`

**Пример:**

```python
from most_queue.theory.vacations.m_h2_h2warm import MH2H2WarmCalc

calc = MH2H2WarmCalc(n=3)
# Настройка параметров прогрева и обслуживания
# (см. тест test_m_h2_h2warm.py)
```

### M/G/1 с прогревом

**Описание:** Одноканальная система с прогревом.

**Класс расчета:** `MG1WarmCalc`

### M/Ph/c с прогревом, задержкой и отпусками

**Описание:** Сложная система с H₂-обслуживанием, H₂-прогревом, H₂-задержкой и H₂-отпусками.

**Класс расчета:** `MGnWithH2DelayColdWarm`

**Пример:** См. тест `test_mgn_with_h2_delay_cold_warm.py`

## Системы с отрицательными заявками

### M/G/1 RCS (Remove Customer from Service)

**Описание:** Система, где отрицательные заявки удаляют заявку из обслуживания.

**Класс расчета:** `MG1RCS`

**Пример:**

```python
from most_queue.theory.negative.mg1_rcs import MG1RCS

calc = MG1RCS()
calc.set_sources(l=0.5, l_neg=0.1)  # l_neg - интенсивность отрицательных заявок
calc.set_servers(b)
results = calc.run()
```

### M/G/c RCS

**Описание:** Многоканальная система с отрицательными заявками типа RCS.

**Класс расчета:** `MGnRCS`

### M/G/c Disaster

**Описание:** Система, где отрицательные заявки удаляют все заявки из системы (и из очереди, и из обслуживания).

**Класс расчета:** `MGnDisaster`

**Пример:** См. тесты `test_mgn_disaster.py` и `test_mg1_disaster.py`

## Fork-Join системы

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

**Класс расчета:** `SplitJoinCalc`

**Пример:** См. тест `test_fj_sim.py`

## Системы с пакетным поступлением

### M^x/M/1

**Описание:** Система, где заявки поступают пакетами случайного размера.

**Класс расчета:** `BatchMM1`

**Пример:**

```python
from most_queue.theory.batch.mm1 import BatchMM1

calc = BatchMM1()
# Вероятности размеров пакетов: [p(1), p(2), p(3), ...]
batch_probs = [0.2, 0.3, 0.1, 0.2, 0.2]
calc.set_sources(l=0.5, batch_probs=batch_probs)
calc.set_servers(mu=1.0)
results = calc.run()
```

## Системы с нетерпеливыми заявками

### M/M/1/D (с экспоненциальным нетерпением)

**Описание:** Система, где заявки могут покинуть очередь, если ожидание слишком долгое.

**Класс расчета:** `MM1Impatience`

**Пример:** См. тест `test_impatience.py`

## Закрытые системы

### Engset (M/M/1/N)

**Описание:** Система с конечным числом источников заявок.

**Класс расчета:** `EngsetCalc`

**Пример:**

```python
from most_queue.theory.closed.engset import EngsetCalc

calc = EngsetCalc()
calc.set_sources(N=10, lambda_source=0.5)  # 10 источников
calc.set_servers(mu=1.0)
results = calc.run()
```

## Сравнительная таблица моделей

| Модель | Класс расчета | Симуляция | Приоритеты | Особенности |
|--------|--------------|-----------|------------|-------------|
| M/M/c | MMnrCalc | QsSim | - | Базовая модель |
| M/G/1 | MG1Calc | QsSim | - | Произвольное обслуживание |
| GI/M/1 | GIM1Calc | QsSim | - | Общий поток |
| M/G/c/PR | MGnInvarApproximation | PriorityQueueSimulator | Да | Прерываемый приоритет |
| M/G/c/NP | MGnInvarApproximation | PriorityQueueSimulator | Да | Непрерываемый приоритет |
| Fork-Join | ForkJoinMarkovianCalc | ForkJoinSim | - | Параллельное обслуживание |
| M^x/M/1 | BatchMM1 | QueueingSystemBatchSim | - | Пакетное поступление |
| Engset | EngsetCalc | QueueingFiniteSourceSim | - | Конечное число источников |

## Рекомендации по выбору модели

1. **Начните с простой модели** — M/M/c для базового понимания
2. **Учитывайте реальные данные** — выберите распределения, соответствующие вашим данным
3. **Используйте симуляцию для проверки** — сравните результаты расчета и симуляции
4. **Учитывайте особенности системы** — приоритеты, отпуска, ограничения

## Примеры использования

Все модели имеют примеры использования в папке `tests/`. Рекомендуется изучить соответствующие тесты для понимания деталей использования.

---

**См. также:**
- [Симуляция СМО](simulation.md) — имитационное моделирование
- [Численные методы](calculation.md) — аналитические расчеты
- [Приоритетные системы](priorities.md) — детали работы с приоритетами
- [Сети очередей](networks.md) — моделирование сетей

