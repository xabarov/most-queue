# FIFO системы (дисциплина First In First Out)

[🇬🇧 English version](fifo.md) · [← Каталог моделей](../models.ru.md)

![Схема M/M/c](../figures/fifo_mmn.ru.png)

**Простыми словами:** заявки (клиенты, задачи, пакеты) приходят в случайные моменты, встают
в общую очередь и обслуживаются в порядке прихода первым освободившимся прибором. Разница
между моделями этого раздела — только в том, насколько «случайны» поток и обслуживание:
от полностью бес­памятных M/M/c до произвольных распределений, приближаемых гиперэкспонентой
(метод Такахаси-Таками).

### M/M/c

**Описание:** Многоканальная система с пуассоновским потоком и экспоненциальным обслуживанием.

**Суть:** «идеальный колл-центр» — и промежутки между звонками, и длительности разговоров
случайны и не зависят от прошлого. Простейшая многоканальная модель, все характеристики
считаются точно; с неё стоит начинать любой анализ.

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

**Суть:** то же, что M/M/c, но мест в «зале ожидания» всего r: заявка, пришедшая в полную
систему, получает отказ и теряется. Модель для систем с конечным буфером (телефония,
сетевое оборудование).

**Класс расчета:** `MMnrCalc`

**Пример:**

```python
from most_queue.theory.fifo.mmnr import MMnrCalc

calc = MMnrCalc(n=3, r=20)  # 3 канала, очередь до 20
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
```

### M/M/n/0 — Erlang B (система с потерями)

![Система с потерями Erlang B](../figures/loss.ru.png)

**Описание:** Классическая система с потерями: очереди нет, заявка, заставшая все n приборов занятыми, теряется. Вероятность блокировки — формула Эрланга B (устойчивая рекурсия).

**Суть:** сколько нужно телефонных линий (коек, парковочных мест), чтобы терять не больше
заданной доли клиентов. По теореме Севастьянова блокировка не зависит от формы распределения
обслуживания — только от его среднего, поэтому результат верен и для M/G/n/0.

**Класс расчета:** `ErlangBCalc` (`most_queue.theory.fifo.erlang`)

**Пример:**

```python
from most_queue.theory.fifo.erlang import ErlangBCalc

calc = ErlangBCalc(n=3)
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
blocking = calc.get_blocking_probability()
```

### M/M/n — Erlang C (система с ожиданием)

**Описание:** Многоканальная система с бесконечной очередью. Вероятность ожидания — формула Эрланга C; моменты времени ожидания в замкнутой форме.

**Суть:** базовая модель staffing: какова вероятность, что клиенту придётся ждать, и сколько.
Ожидание либо нулевое (есть свободный прибор), либо экспоненциальное — отсюда все моменты
одной формулой.

**Класс расчета:** `ErlangCCalc` (`most_queue.theory.fifo.erlang`)

**Пример:**

```python
from most_queue.theory.fifo.erlang import ErlangCCalc

calc = ErlangCCalc(n=3)
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
results = calc.run()
p_wait = calc.get_waiting_probability()
```

### M/G/∞ (бесконечное число приборов)

![Схема M/G/∞](../figures/m_g_inf.ru.png)

**Описание:** Каждой заявке мгновенно достаётся свой прибор: ожидания нет, число занятых приборов имеет пуассоновское распределение со средним λ·b₁ независимо от формы распределения обслуживания (нечувствительность).

**Суть:** модель «изобильного» ресурса — активные сессии, звонки в большой сети, машины
на трассе. Ответ на вопрос «сколько ресурса реально занято одновременно» и строительный
блок для staffing-аппроксимаций.

**Класс расчета:** `MGInfCalc` (`most_queue.theory.fifo.m_g_inf`)

**Пример:**

```python
from most_queue.theory.fifo.m_g_inf import MGInfCalc
from most_queue.random.distributions import GammaDistribution

calc = MGInfCalc()
calc.set_sources(l=1.0)

gamma_params = GammaDistribution.get_params_by_mean_and_cv(2.0, 1.2)
b = GammaDistribution.calc_theory_moments(gamma_params, 4)
calc.set_servers(b=b)

results = calc.run()
busy_mean = calc.get_offered_load()  # среднее число занятых приборов
```

### M/G/1

**Описание:** Одноканальная система с пуассоновским потоком и произвольным распределением времени обслуживания.

**Суть:** один прибор, время обслуживания — любое (задаётся начальными моментами). Классика
Полячека–Хинчина: очередь растёт не только от загрузки, но и от *разброса* времени
обслуживания — при одинаковом среднем система с редкими «тяжёлыми» заявками ждёт гораздо
дольше, чем с одинаковыми.

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

Следующие четыре модели — **size-based дисциплины**: прибор выбирает, кого обслуживать,
глядя на *размер* заявки (известный или предсказанный), а не на порядок прихода. Вот как
одни и те же заявки проходят через один прибор при разных дисциплинах:

![Сравнение дисциплин FCFS/SJF/SRPT](../figures/disciplines_timeline.ru.png)

### M/G/1 SRPT

**Описание:** Одноканальная M/G/1 с дисциплиной **Shortest Remaining Processing Time** (прерывание по остатку работы). Численно: формула Schrage–Miller (1966).

**Суть:** прибор всегда занят заявкой, которой *осталось* меньше всего работы; если пришла
более короткая — текущая прерывается и ждёт (см. на схеме выше, как заявка A уступает место
и дообслуживается в конце). SRPT доказуемо минимизирует среднее время пребывания среди всех
дисциплин.

**Класс расчета:** `MG1SrptCalc`  
**Симуляция:** `SizeBasedQsSim(discipline="SRPT")` — размер заявки сэмплируется при приходе.

**Пример:**

```python
from most_queue.theory.srpt import MG1SrptCalc
from most_queue.random.distributions import H2Distribution

calc = MG1SrptCalc()
calc.set_sources(1.0)
h2 = H2Distribution.get_params_by_mean_and_cv(0.7, 1.2)
calc.set_servers(h2, "H")
results = calc.run()
```

### M/G/1 SJF (SPT)

**Описание:** Непрерываемое обслуживание по **наименьшему истинному размеру** (Shortest Job First / Shortest Processing Time).

**Суть:** без прерываний — в момент освобождения прибора из очереди берётся самая короткая
заявка, но начатое обслуживание всегда доводится до конца.

**Класс расчета:** `MG1SjfCalc`  
**Симуляция:** `SizeBasedQsSim(discipline="SJF")`

### M/G/1 PSJF

**Описание:** Прерываемое обслуживание по **исходному** размеру заявки (отличается от SRPT).

**Суть:** как SRPT, но сравнивается *полный исходный* размер, а не остаток: почти
дообслуженная длинная заявка всё равно уступит новой короткой.

**Класс расчета:** `MG1PsjfCalc`  
**Симуляция:** `SizeBasedQsSim(discipline="PSJF")`

### M/G/1 SPJF (с предсказаниями)

**Описание:** Непрерываемое обслуживание по **предсказанному** размеру \(Y\) (Mitzenmacher, 2020). Совместное распределение \((X,Y)\) задаётся объектом предиктора (`PerfectPredictor`, `ExpNoisePredictor`, …).

**Суть:** истинный размер заявки неизвестен, но есть его *предсказание* (например, от
ML-модели) — обслуживаем короткие «по прогнозу». Модель отвечает на вопрос, сколько
выигрыша от SJF сохраняется при неточных предсказаниях. При идеальном предикторе
переходит в SJF.

**Класс расчета:** `MG1SpjfCalc`  
**Симуляция:** `SizeBasedQsSim(discipline="SPJF")` + `set_predictor(...)`.

**Пример:**

```python
from most_queue.theory.srpt import MG1SpjfCalc
from most_queue.theory.srpt.utils.predictor import ExpNoisePredictor

calc = MG1SpjfCalc()
calc.set_sources(0.5)
calc.set_servers(1.0, "M")
calc.set_predictor(ExpNoisePredictor())
results = calc.run()
```

#### Graceful degradation предсказаний (learning-augmented scheduling)

**Описание:** Как меняется среднее время отклика SPJF по мере ухудшения предсказаний? Хелпер
`prediction_degradation_curve` проходит по уровню лог-нормального шума σ и возвращает среднее SPJF в
рамке SRPT (size-aware оптимум), SJF (идеальные предсказания) и слепой FB/LAS. Возвращает
**точку перелома** — уровень шума, при котором SPJF начинает проигрывать *слепой* политике,
воспроизводя центральную открытую задачу survey SIGMETRICS 2025 «Queueing, Predictions, and LLMs»
(гарантии graceful degradation нет «бесплатно»).

```python
from most_queue.theory.srpt import prediction_degradation_curve

curve = prediction_degradation_curve(0.7, service_h2_params, "H")
# curve.spjf[i] при curve.sigmas[i]; ссылки curve.srpt / curve.sjf / curve.blind_fb;
# curve.breakeven_sigma — шум, при котором SPJF становится хуже слепой политики
```

Следующие три дисциплины (FB, PS, LCFS-PR) дополняют size-based семейство. Кто из них
как обращается с заявками разного размера — считают сами калькуляторы библиотеки:

![Замедление по размерам заявки для FCFS/PS/FB/SRPT](../figures/slowdown.ru.png)

### M/G/1 FB (Foreground-Background / LAS)

**Описание:** Прерывающая **blind**-дисциплина: прибор всегда обслуживает заявку с наименьшим *полученным* обслуживанием (least attained service); при равенстве — делится поровну. Размеры заявок знать не нужно.

**Суть:** «дадим шанс новичкам»: свежая заявка сразу получает прибор и держит его, пока не
догонит по обслуженному объёму остальных. Если короткие заявки часты (убывающий hazard rate,
CV > 1) — FB приближается к SRPT, не зная размеров; если время обслуживания почти постоянное —
FB проигрывает даже FCFS. Экспоненциальное обслуживание — граница: FB совпадает с PS.

**Класс расчета:** `MG1FbCalc` (`most_queue.theory.srpt`)
**Симуляция:** `FBSim` (`most_queue.sim.single_server_disciplines`)

**Пример:**

```python
from most_queue.theory.srpt import MG1FbCalc
from most_queue.random.distributions import GammaDistribution

calc = MG1FbCalc()
calc.set_sources(1.0)
calc.set_servers(GammaDistribution.get_params_by_mean_and_cv(0.7, 1.2), "Gamma")
results = calc.run()
```

### M/G/1 PS (Processor Sharing)

![Схема Processor Sharing](../figures/ps.ru.png)

**Описание:** Прибор делится поровну между всеми находящимися заявками (каждая из k заявок обслуживается со скоростью 1/k). Вероятности состояний — геометрические, нечувствительные к форме распределения обслуживания; условное среднее время пребывания заявки размера x — ровно x/(1−ρ).

**Суть:** модель процессора, веб-сервера, разделяемого канала: никто не ждёт «в очереди»,
но все замедляются в одинаковое число раз 1/(1−ρ). Идеально справедливая дисциплина —
baseline для сравнения с SRPT/SJF (которые быстрее в среднем, но за счёт длинных заявок).
Пока считаются только средние (старшие моменты — методы Яшкова/Отта, отложено).

**Класс расчета:** `MG1PSCalc` (`most_queue.theory.fifo.mg1_ps`)
**Симуляция:** `ProcessorSharingSim` (`most_queue.sim.single_server_disciplines`)

**Пример:**

```python
from most_queue.theory.fifo.mg1_ps import MG1PSCalc

calc = MG1PSCalc()
calc.set_sources(l=1.0)
calc.set_servers([0.7, 1.2])  # моменты времени обслуживания
results = calc.run()
slowdown = calc.get_mean_slowdown()          # 1/(1-rho)
t_x = calc.get_conditional_sojourn_mean(2.0)  # x/(1-rho)
```

### M/G/1 LCFS-PR

![Схема LCFS-PR](../figures/lcfs_pr.ru.png)

**Описание:** Прерывающий стек: новая заявка вытесняет обслуживаемую, вытесненные дообслуживаются с места прерывания. Время пребывания распределено как период занятости M/G/1 — все моменты по рекурсиям Такача; вероятности состояний — те же геометрические (BCMP).

**Суть:** «последний пришёл — первый обслужен»: свежая заявка получает прибор сразу,
но рискует быть вытесненной. Среднее время пребывания то же, что у PS (b₁/(1−ρ),
нечувствительность к форме распределения), но разброс гораздо больше — хвосты как у
периода занятости. У FCFS среднее другое: оно зависит ещё и от b₂ (Полячек–Хинчин).

**Класс расчета:** `MG1LcfsPrCalc` (`most_queue.theory.fifo.mg1_lcfs_pr`)
**Симуляция:** `LcfsPRSim` (`most_queue.sim.single_server_disciplines`)

### GI/M/1

**Описание:** Одноканальная система с общим потоком поступления и экспоненциальным обслуживанием.

**Суть:** зеркальная к M/G/1 ситуация — теперь «произвольная» сторона не обслуживание,
а входящий поток: промежутки между приходами имеют любое распределение (задаётся моментами),
обслуживание — экспоненциальное.

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

### GI/G/1 и GI/G/m (двухмоментные аппроксимации)

**Описание:** Приближённый расчёт среднего времени ожидания по первым двум моментам потока и обслуживания: Kingman (верхняя граница), Krämer–Langenbach-Belz для GI/G/1 (точно для M/G/1), Allen–Cunneen для GI/G/m (точно для M/M/m).

**Суть:** «формулы на салфетке» для capacity planning: когда известны только средние и разбросы,
а точного решения нет. Возвращается только первый момент (это аппроксимация, а не точный
расчёт); типичная погрешность KLB — единицы процентов. Формула Кимуры (интерполяция по
D/M/s, M/D/s, M/M/s) отложена — требует точных решений D/M/s.

**Классы расчета:** `GIG1ApproxCalc`, `GIGmApproxCalc` (`most_queue.theory.fifo.gi_g_approx`)

**Пример:**

```python
from most_queue.theory.fifo.gi_g_approx import GIG1ApproxCalc
from most_queue.random.distributions import GammaDistribution

a_params = GammaDistribution.get_params_by_mean_and_cv(1.0, 0.56)
b_params = GammaDistribution.get_params_by_mean_and_cv(0.7, 1.2)

calc = GIG1ApproxCalc()  # или GIG1ApproxCalc(approximation="kingman")
calc.set_sources(GammaDistribution.calc_theory_moments(a_params, 4))
calc.set_servers(GammaDistribution.calc_theory_moments(b_params, 4))
results = calc.run()  # results.w — [w1], только первый момент
```

### H₂/M/c (метод Такахаси-Таками)

**Описание:** Многоканальная система с гиперэкспоненциальным потоком поступления (H₂) и экспоненциальным обслуживанием (M). Использует упрощённый алгоритм §7.6.1 (формулы для z_j, x_j, t_{j,i}, уровень 0).

**Суть:** H₂ («смесь двух экспонент») — универсальная заготовка: подобрав её параметры по
среднему и коэффициенту вариации, можно приблизить почти любое реальное распределение
(при CV < 1 — с комплексными параметрами). Метод Такахаси-Таками — итерационный численный
алгоритм, точно решающий такие многоканальные фазовые модели.

**Класс расчета:** `H2MnCalc`

**Пример:**

```python
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.random.distributions import H2Distribution

calc = H2MnCalc(n=3)

h2_params = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2, is_clx=True)  # mean, cv
#
# Для CV<1 используйте complex-fit: is_clx=True.
# Важно: симулятор `QsSim` не умеет генерировать H2 с комплексными параметрами,
# поэтому сравнение с симуляцией возможно только когда параметры вещественные.
calc.set_sources(h2_params)

calc.set_servers(b=2.0)  # среднее время обслуживания

results = calc.run()
```

### H₂/H₂/c (метод Такахаси-Таками)

**Описание:** Многоканальная система с гиперэкспоненциальным потоком поступления и гиперэкспоненциальным обслуживанием. Использует алгоритм §7.6.2 (CH7).

**Класс расчета:** `HkHkNCalc`

**Пример:**

```python
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc
from most_queue.random.distributions import H2Distribution

calc = HkHkNCalc(n=3, k=2)

h2_arr = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2)
# Для CV<1 используйте complex-fit: is_clx=True (тогда параметры могут быть комплексными).
calc.set_sources(u=[h2_arr.p1, 1 - h2_arr.p1], lam=[h2_arr.mu1, h2_arr.mu2])

h2_srv = H2Distribution.get_params_by_mean_and_cv(2.0, 1.2)
calc.set_servers(y=[h2_srv.p1, 1 - h2_srv.p1], mu=[h2_srv.mu1, h2_srv.mu2])

results = calc.run()
```

**Примечание про CV<1:** при \(CV<1\) H₂ аппроксимация использует *complex-fit* (комплексные параметры).
Симулятор `QsSim` не генерирует H₂ с комплексными параметрами, поэтому для валидации удобно
сравнивать расчёт (H₂ complex-fit) с симуляцией эквивалентной по mean/CV `Gamma`-модели (см. тесты `tests/test_tt_vs_sim_gamma_cvl1.py`).

### M/D/c

**Описание:** Многоканальная система с пуассоновским потоком и детерминированным временем обслуживания.

**Суть:** обслуживание занимает строго одинаковое время (конвейер, такт автомата). Нулевой
разброс обслуживания — лучший случай для очереди: при той же загрузке ожидание вдвое короче,
чем в M/M/c.

**Класс расчета:** `MDn`

**Пример:**

```python
from most_queue.theory.fifo.m_d_n import MDn

calc = MDn(n=3)
calc.set_sources(l=2.0)
calc.set_servers(b=1.0)  # постоянное время обслуживания
results = calc.run()
```

### Eₖ/D/c

**Описание:** Многоканальная система с распределением Эрланга межприходных времен и детерминированным обслуживанием.

**Суть:** поток Эрланга — более «ритмичный», чем пуассоновский (заявки приходят регулярнее),
обслуживание постоянное. Модель почти детерминированных производственных линий.

**Класс расчета:** `EkDn`

**Пример:**

```python
from most_queue.theory.fifo.ek_d_n import EkDn

calc = EkDn(n=3, k=2)  # 3 канала, Эрланга порядка 2
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

h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=1.2, is_clx=True)
calc.set_servers(h2_params)

results = calc.run()
```
