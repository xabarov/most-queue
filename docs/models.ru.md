# Каталог поддерживаемых моделей СМО

[🇬🇧 English version](models.md)

Библиотека Most-Queue поддерживает широкий спектр моделей систем массового обслуживания.
Каталог разбит по семействам — у каждого семейства своя страница со схемами,
объяснениями «на пальцах», классами и примерами кода. Схемы генерируются скриптом
[`figures/generate_figures.py`](figures/generate_figures.py) — при добавлении новой модели
добавьте функцию-схему и перегенерируйте PNG.

## Семейства моделей

| Семейство | Что внутри |
|---|---|
| [FIFO системы (дисциплина First In First Out)](models/fifo.ru.md) | M/M/c, Erlang B/C, M/G/1 + size-based дисциплины (SRPT/SJF/PS/FB/LCFS-PR), GI/G-аппроксимации, H₂-решатели Такахаси-Таками |
| [Системы с приоритетами](models/priority.ru.md) | M/G/1 и M/G/c с классами PR/NP, RDR для многоканальных многоприоритетных, точные CTMC-эталоны |
| [Polling-системы (циклический сервер)](models/polling.ru.md) | циклический сервер по Q очередям с переключением, псевдо-закон сохранения |
| [Системы с отпусками (Vacations)](models/vacations.ru.md) | многократные отпуска, N-policy, разогрев/охлаждение, ненадёжный прибор |
| [Системы с отрицательными заявками](models/negative.ru.md) | отрицательные заявки: RCS и disasters, одно- и многоканальные |
| [Fork-Join системы](models/fork-join.ru.md) | параллельное обслуживание fork-join и split-join |
| [Системы с пакетным поступлением](models/batch.ru.md) | пакетное поступление Mˣ/M/1 и групповое обслуживание M/M^[a,b]/1 |
| [Системы с нетерпеливыми заявками](models/impatience.ru.md) | нетерпеливые заявки: M/M/1/D и Erlang-A со staffing |
| [Retrial-очереди (повторные попытки)](models/retrial.ru.md) | retrial-очереди с орбитой (M/M/1, M/G/1) |
| [Матрично-аналитические модели (MAP/PH)](models/map-ph.ru.md) | коррелированные потоки: MAP/PH/1, MAP/M/c, MAP/PH/c, BMAP-варианты, фиттинг MMPP |
| [Multiserver-job системы (MSJ)](models/msj.ru.md) | multiserver-job: заявка занимает k серверов сразу |
| [Балансировка нагрузки / диспетчеризация (mean-field)](models/load-balancing.ru.md) | диспетчеризация power-of-d / JSQ / JIQ, mean-field |
| [Нестационарные очереди Mₜ/M/c (переменная нагрузка)](models/time-varying.ru.md) | нестационарные Mₜ/M/c: PSA и MOL |
| [Age of Information (AoI, свежесть информации)](models/aoi.ru.md) | Age of Information: средний и пиковый возраст |
| [Закрытые системы](models/closed.ru.md) | системы с конечным числом источников (Engset) |
| [Сети массового обслуживания](models/networks.ru.md) | открытые/закрытые сети: декомпозиция, Джексон, QNA, MVA/Бьюзен, BCMP, G-сети, блокировки, fork-join станции |

## Сравнительная таблица моделей

| Модель | Класс расчета | Симуляция | Приоритеты | Особенности |
|--------|--------------|-----------|------------|-------------|
| M/M/c | MMnrCalc | QsSim | - | Базовая модель |
| M/M/n/0 (Erlang B) | ErlangBCalc | QsSim(buffer=0) | - | Потери, нечувствительность M/G/n/0 |
| M/M/n (Erlang C) | ErlangCCalc | QsSim | - | Вероятность ожидания, моменты W |
| M/G/∞ | MGInfCalc | QsSim(n>>a) | - | Бесконечно много приборов |
| M/G/1 | MG1Calc | QsSim | - | Произвольное обслуживание |
| M/G/1 SRPT | MG1SrptCalc | SizeBasedQsSim | - | Size-based, Schrage–Miller |
| M/G/1 SJF | MG1SjfCalc | SizeBasedQsSim | - | Non-preemptive по размеру |
| M/G/1 PSJF | MG1PsjfCalc | SizeBasedQsSim | - | Preemptive по исходному размеру |
| M/G/1 SPJF | MG1SpjfCalc | SizeBasedQsSim | - | По предсказанию Y |
| M/G/1 FB/LAS | MG1FbCalc | FBSim | - | Blind, по attained service |
| M/G/1 PS | MG1PSCalc | ProcessorSharingSim | - | Равное разделение, slowdown 1/(1−ρ) |
| M/G/1 LCFS-PR | MG1LcfsPrCalc | LcfsPRSim | - | Время пребывания = период занятости |
| GI/M/1 | GIM1Calc | QsSim | - | Общий поток |
| GI/G/1, GI/G/m (approx) | GIG1ApproxCalc, GIGmApproxCalc | QsSim | - | Kingman/KLB/Allen–Cunneen, только w1 |
| M/G/c/PR | MGnInvarApproximation | PriorityQueueSimulator | Да | Прерываемый приоритет |
| M/G/c/NP | MGnInvarApproximation | PriorityQueueSimulator | Да | Непрерываемый приоритет |
| M/G/1 multiple vacations | MG1MultipleVacationsCalc | VacationQueueingSystemSimulator | - | Fuhrmann–Cooper |
| M/G/1 N-policy | MG1NPolicyCalc | NPolicyQueueSim | - | Порог включения N |
| M/G/1 unreliable | MG1UnreliableCalc | UnreliableQueueSim | - | Отказы+ремонты, completion time |
| Fork-Join | ForkJoinMarkovianCalc | ForkJoinSim | - | Параллельное обслуживание |
| Mˣ/M/1 | BatchMM1 | QueueingSystemBatchSim | - | Пакетное поступление |
| Erlang-A (M/M/n+M) | MMnImpatienceCalc | ImpatientQueueSim | - | Уходы, staffing-помощник |
| M/M/1 retrial | MM1RetrialCalc | RetrialQueueSim | - | Орбита, точное усечение цепи |
| M/G/1 retrial | MG1RetrialCalc | RetrialQueueSim | - | Формула Falin–Templeton |
| MAP/PH/1 | MapPh1Calc | QsSim("MAP", "PH") | - | Коррелированный вход, QBD |
| M/PH/1, PH/PH/1 | MPh1Calc, PhPh1Calc | QsSim | - | Частные случаи QBD |
| MAP/M/c | MapMMcCalc | QsSim("MAP","M") | - | Многоканальный, коррелированный вход |
| MAP/PH/c | MapPhCCalc | QsSim("MAP","PH") | - | Многоканальный, коррелированный вход + PH-обслуживание |
| BMAP/M/1 | BmapM1Calc | - | - | Пакетный (коррелированный) вход |
| BMAP/PH/1 | BmapPh1Calc | BmapPh1Sim | - | Пакетный вход + PH-обслуживание |
| M/M/k, m классов (RDR-A) | RDRAPriorityCalc | PriorityQueueSimulator | Да | Многоканальные многоприоритетные, RDR |
| M/M/k, m классов (точно) | MMkPriorityExact | PriorityQueueSimulator | Да | Точная CTMC + дисперсия отклика по классам |
| M/PH/k, m классов | RDRAPriorityPH, MPhPhK2Class | PriorityQueueSimulator | Да | Фазовое обслуживание (RDR §2.3) |
| Multiserver-job (MSJ) | MsjExactCalc, MsjSaturatedCalc | MsjSim | - | Заявка занимает k серверов; порог устойчивости |
| Балансировка нагрузки (power-of-d, JSQ, JIQ) | LoadBalancingMeanField | LoadBalancingSim | - | Диспетчеризация по большому пулу (mean-field) |
| Polling (циклический сервер) | PollingCalc | PollingSim | - | Switchover, exhaustive/gated, псевдо-закон сохранения |
| Нестационарная Mₜ/M/c | TimeVaryingMMcCalc | TimeVaryingMMcSim | - | Переменная нагрузка, приближения PSA и MOL |
| Age of Information | AoICalc, LcfsPreemptiveAoICalc | AoISim | - | Средний и пиковый AoI |
| M/M^[a,b]/1 групповое обслуживание | BulkServiceMM1Calc | BulkServiceSim | - | Пакетное обслуживание, батчинг LLM |
| Engset | Engset | QueueingFiniteSourceSim | - | Конечное число источников |
| Открытая сеть (декомпозиция) | OpenNetworkCalc | NetworkSimulator | Да (OpenNetworkCalcPriorities) | Узлы M/G/n, приближённо |
| Сеть Джексона | JacksonNetworkCalc | NetworkSimulator | - | Точный product-form, узлы M/M/n |
| Открытая сеть QNA (Уитт) | OpenNetworkCalcQNA | NetworkSimulator | - | Двухмоментные внутренние потоки, поправка KLB |
| Закрытая сеть | ClosedNetworkCalc | ClosedNetworkSim | - | Точный MVA / свёртка Бьюзена / Швейцер, delay-станции |
| G-сеть (Геленбе) | GNetworkCalc | NegativeNetwork | - | Отрицательные заявки/сигналы, точный product-form |
| BCMP мультиклассовая сеть | BCMPOpenNetworkCalc, BCMPClosedNetworkCalc | - | - | FCFS/PS/LCFS-PR/IS, мультичейн-MVA |

## Рекомендации по выбору модели

1. **Начните с простой модели** — M/M/c для базового понимания
2. **Учитывайте реальные данные** — выберите распределения, соответствующие вашим данным
3. **Используйте симуляцию для проверки** — сравните результаты расчета и симуляции
4. **Учитывайте особенности системы** — приоритеты, отпуска, ограничения
## Примеры использования

Все модели имеют примеры использования в папке `tests/`. Рекомендуется изучить соответствующие тесты для понимания деталей использования.

---

**См. также:**
- [Симуляция СМО](simulation.ru.md) — имитационное моделирование
- [Численные методы](calculation.ru.md) — аналитические расчеты
- [Приоритетные системы](priorities.ru.md) — детали работы с приоритетами
- [Сети очередей](networks.ru.md) — моделирование сетей
