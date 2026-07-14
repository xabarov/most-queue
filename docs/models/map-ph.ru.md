# Матрично-аналитические модели (MAP/PH)

[🇬🇧 English version](map-ph.md) · [← Каталог моделей](../models.ru.md)

![Коррелированный вход MAP](../figures/map_arrivals.ru.png)

**Суть:** реальный трафик пульсирует — за коротким интервалом чаще следует короткий.
Марковский поток (MAP) ловит эту корреляцию парой матриц (D₀, D₁); фазовые (PH) распределения
играют ту же роль для обслуживания. Очередь MAP/PH/1 решается **точно** матрично-геометрическим
(QBD) методом — и ответ может отличаться от renewal-модели с теми же mean/CV в разы
(см. [`tutorials/map_ph_correlation.ipynb`](../../tutorials/map_ph_correlation.ipynb)).

Впервые видите PH и MAP? Они разобраны по шагам — со схемами того, как Exp, Эрланг, H₂ и Кокса
оказываются частными случаями PH, и как устроен MMPP — в
[справочнике распределений](../distributions.ru.md#фазовые-распределения-ph).

### MAP/PH/1

**Описание:** Коррелированный вход, фазовое обслуживание, один прибор. Стационарное распределение — QBD с logarithmic reduction (Latouche–Ramaswami); моменты ожидания — дифференцированием LST прибывающей заявки.

**Класс расчета:** `MapPh1Calc` (`most_queue.theory.matrix.map_ph1`)
**Симуляция:** `QsSim` c `set_sources(map_params, "MAP")` и `set_servers(ph_params, "PH")`

### M/PH/1 и PH/PH/1

**Описание:** Частные случаи на том же QBD-ядре: `MPh1Calc` (пуассоновский вход) в точности воспроизводит Полячека–Хинчина; `PhPh1Calc` (renewal PH-вход) покрывает одноканальные системы типа GI/PH.

**Классы расчета:** `MPh1Calc`, `PhPh1Calc` (`most_queue.theory.matrix.map_ph1`)

### MAP/M/c

**Описание:** Коррелированный вход и **c** экспоненциальных приборов, решается как QBD с уровне-зависимой границей (уровни 0..c-1 — число занятых приборов, однородный блок от уровня c). Реалистичная модель колл-центра или ЦОД с bursty-трафиком.

**Суть:** многоканальный аналог MAP/PH/1 — Erlang C, но с пульсацией входа, которую Erlang C
игнорирует. Однофазный (пуассоновский) MAP в точности воспроизводит Erlang C; bursty-MAP с той
же интенсивностью даёт заметно большее ожидание.

**Класс расчета:** `MapMMcCalc` (`most_queue.theory.matrix.map_mmc`)
**Симуляция:** `QsSim(c)` c `set_sources(map_params, "MAP")` и `set_servers(mu, "M")`

**Пример:**

```python
import numpy as np
from most_queue.random.map_ph import MAP
from most_queue.theory.matrix.map_mmc import MapMMcCalc

mmpp = MAP.mmpp([2.5, 0.5], np.array([[-0.15, 0.15], [0.25, -0.25]]))  # bursty-вход

calc = MapMMcCalc(n=3)  # 3 прибора
calc.set_sources(mmpp)
calc.set_servers(mu=1.0)
results = calc.run()  # вероятности состояний + средние по Литтлу
```

### MAP/PH/c

**Описание:** Самая общая одностанционная модель раздела: коррелированный MAP-вход, фазовое обслуживание и c приборов. Решается как QBD, где фаза — фаза MAP × мультимножество фаз занятых приборов.

**Суть:** совмещает всё — пульсирующий вход *и* вариативное (фазовое) обслуживание *и*
несколько приборов. В точности сводится к MAP/M/c (exp-обслуживание), MAP/PH/1 (один прибор)
и M/H₂/c по Такахаси-Таками (пуассоновский вход). Пространство фаз обслуживания растёт
комбинаторно, поэтому держите порядок PH и c умеренными (например, 2-фазное обслуживание, c ≤ 6).

**Класс расчета:** `MapPhCCalc` (`most_queue.theory.matrix.map_phc`)
**Симуляция:** `QsSim(c)` c `set_sources(map_params, "MAP")` и `set_servers(ph_params, "PH")`

### BMAP/M/1

**Описание:** **Пакетный** марковский вход (заявки приходят коррелированными пачками), один экспоненциальный прибор. Пачка поднимает уровень больше чем на 1, поэтому это цепь M/G/1-типа; решается устойчивым усечением уровней.

**Суть:** трафик, приходящий пачками сразу по несколько заявок (пакетные поезда, оптовые
заказы), причём сам пакетный процесс марковски-модулирован. В точности сводится к M^[X]/M/1
(пуассоновские пачки) и к MAP/M/1 (пачки размера 1).

**Класс расчета:** `BmapM1Calc` (`most_queue.theory.matrix.bmap_m1`)

**Пример:**

```python
from most_queue.random.map_ph import bmap_poisson_batch
from most_queue.theory.matrix.bmap_m1 import BmapM1Calc

# пачки приходят Poisson(rate=0.5); размер 1..5 с этими вероятностями
bmap = bmap_poisson_batch(0.5, [0.2, 0.3, 0.1, 0.2, 0.2])

calc = BmapM1Calc()
calc.set_sources(bmap)
calc.set_servers(mu=2.5)
results = calc.run()
```

### BMAP/PH/1

**Описание:** Пакетный марковский вход с **фазовым** обслуживанием — представитель семейства BMAP с общим обслуживанием. Решается усечением уровней над (уровень, фаза BMAP, фаза обслуживания). Произвольное (не PH) обслуживание задаётся PH-фиттингом по его моментам.

**Суть:** пульсирующий пакетный трафик встречает вариативное (не только экспоненциальное)
обслуживание. В точности сводится к BMAP/M/1 (exp-обслуживание) и к MAP/PH/1 (пачки размера 1).

**Класс расчета:** `BmapPh1Calc` (`most_queue.theory.matrix.bmap_ph1`)
**Симуляция:** `BmapPh1Sim` (`most_queue.sim.bmap`)

**Пример:**

```python
from most_queue.random.map_ph import bmap_poisson_batch, PHDistribution
from most_queue.random.distributions import H2Distribution
from most_queue.theory.matrix.bmap_ph1 import BmapPh1Calc

bmap = bmap_poisson_batch(0.4, [0.2, 0.3, 0.1, 0.2, 0.2])
service = PHDistribution.from_h2(H2Distribution.get_params_by_mean_and_cv(0.5, 1.3))

calc = BmapPh1Calc()
calc.set_sources(bmap)
calc.set_servers(service)
results = calc.run()
```
