# Системы с отпусками (Vacations)

[🇬🇧 English version](vacations.md) · [← Каталог моделей](../models.ru.md)

![Жизненный цикл прибора в vacation-моделях](../figures/vacations.ru.png)

**Простыми словами:** прибор не всегда готов работать мгновенно. После простоя ему нужен
**прогрев** (warm-up — прогрев станка, холодный старт сервера), после опустошения очереди он
может уйти в **охлаждение/отпуск** (cooling/vacation — энергосбережение, регламентные работы),
иногда с **задержкой** (delay — ждём, не придёт ли ещё заявка, прежде чем выключаться).
Заявкам, пришедшим «не вовремя», приходится ждать дольше — модели этого раздела считают,
насколько.

### M/G/1 с многократными отпусками (multiple vacations)

**Описание:** Классическая vacation-модель: опустошив очередь, прибор уходит в отпуск; вернувшись в пустую систему — сразу в следующий. Точное решение через декомпозицию Fuhrmann–Cooper: ожидание = ожидание M/G/1 + остаточное время отпуска.

**Суть:** сервер, «досыпающий» пока нет работы (энергосбережение, фоновые задачи).
Плата заявок за отпуска — в среднем половина «длины» отпуска с поправкой на разброс,
независимо от загрузки.

**Класс расчета:** `MG1MultipleVacationsCalc` (`most_queue.theory.vacations.mg1_vacations`)
**Симуляция:** `VacationQueueingSystemSimulator(1, is_multiple_vacations=True)` + `set_cold(...)`

**Пример:**

```python
from most_queue.theory.vacations.mg1_vacations import MG1MultipleVacationsCalc
from most_queue.random.distributions import GammaDistribution

b = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(0.7, 1.2), 5)
vac = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(1.5, 1.2), 4)

calc = MG1MultipleVacationsCalc()
calc.set_sources(l=1.0)
calc.set_servers(b)
calc.set_vacations(vac)
results = calc.run()  # для k моментов W нужно k+1 моментов отпуска
```

### M/G/1 под N-policy

![Схема N-policy](../figures/n_policy.ru.png)

**Описание:** Прибор выключается при опустошении системы и включается, только когда накопится N заявок; далее обслуживает до опустошения. Точное решение: добавка к ожиданию M/G/1 — эрланговская смесь, в среднем (N−1)/(2λ).

**Суть:** экономия на «включениях»: чем больше N, тем реже прибор запускается, но тем дольше
ждут первые накопившиеся заявки. Модель для выбора порога N (batch-запуск оборудования,
редкие рейсы). N=1 — обычная M/G/1.

**Класс расчета:** `MG1NPolicyCalc` (`most_queue.theory.vacations.mg1_vacations`)
**Симуляция:** `NPolicyQueueSim(1, big_n=N)`

### M/G/1 с ненадёжным прибором (breakdowns & repairs)

![Схема ненадёжного прибора](../figures/unreliable.ru.png)

**Описание:** Прибор отказывает с пуассоновской интенсивностью ξ во время обслуживания; ремонт — произвольное распределение; прерванная заявка дообслуживается с места остановки. Точное сведение к M/G/1 с «completion time» (обслуживание + свои ремонты) — Avi-Itzhak–Naor (1963).

**Суть:** станок, который ломается под нагрузкой: заявка занимает прибор своё время
обслуживания плюс все случившиеся за это время ремонты. Кумулянты completion time
считаются в замкнутой форме, дальше работает обычная формула Полячека–Хинчина.

**Класс расчета:** `MG1UnreliableCalc` (`most_queue.theory.vacations.mg1_unreliable`)
**Симуляция:** `UnreliableQueueSim` (`most_queue.sim.unreliable`)

**Пример:**

```python
from most_queue.theory.vacations.mg1_unreliable import MG1UnreliableCalc
from most_queue.random.distributions import GammaDistribution

b = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(0.5, 1.2), 5)
r = GammaDistribution.calc_theory_moments(
    GammaDistribution.get_params_by_mean_and_cv(0.4, 1.2), 5)

calc = MG1UnreliableCalc()
calc.set_sources(l=1.0)
calc.set_servers(b)
calc.set_breakdowns(xi=0.3, repair=r)
results = calc.run()
```

### M/H₂/c с прогревом

**Описание:** Многоканальная система с гиперэкспоненциальным обслуживанием и прогревом каналов.

**Класс расчета:** `MH2nH2Warm`

**Пример:**

```python
from most_queue.theory.vacations.m_h2_h2warm import MH2nH2Warm

calc = MH2nH2Warm(n=3)
# Настройка параметров прогрева и обслуживания
# (см. тест test_m_h2_h2warm.py)
```

### M/M/n с H₂-охлаждением и H₂-прогревом

**Описание:** Многоканальная система с экспоненциальным обслуживанием, гиперэкспоненциальными охлаждением и прогревом.

**Класс расчета:** `MMnHyperExpWarmAndCold` (`most_queue.theory.vacations.mmn_with_h2_cold_and_h2_warmup`)

**Пример:** См. тест `test_mmn_h2cold_h2warm.py`

### M/G/1 с прогревом

**Описание:** Одноканальная система с прогревом.

**Класс расчета:** `MG1WarmCalc`

### M/Ph/c с прогревом, задержкой и отпусками

**Описание:** Сложная система с H₂-обслуживанием, H₂-прогревом, H₂-задержкой и H₂-отпусками.

**Класс расчета:** `MGnH2ServingColdWarmDelay`

**Пример:** См. тест `test_mgn_with_h2_delay_cold_warm.py`
