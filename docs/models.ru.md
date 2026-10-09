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
| [FIFO системы (дисциплина First In First Out)](models/fifo.ru.md) | M/M/c, Erlang B/C, M/G/1, GI/M/1, GI/G-аппроксимации |
| [Size-based дисциплины](models/size-based.ru.md) | SRPT/SJF/PSJF/SPJF (с предсказаниями), FB/LAS, PS, LCFS-PR |
| [Многоканальные H₂-системы (Такахаси-Таками)](models/multiserver-h2.ru.md) | итерационные решатели H₂/M/c, H₂/H₂/c, M/H₂/c |
| [Системы с приоритетами, часть 1: статические классы](models/priority.ru.md) | M/G/1 и M/G/c с классами PR/NP, RDR для многоканальных многоприоритетных, точные CTMC-эталоны |
| [Системы с приоритетами, часть 2: динамические и расширенные](models/priority-dynamic.ru.md) | накапливаемый приоритет (APQ), нетерпение, MMAP/PH-вход, retrial с приоритетом, preemptive repeat |
| [Polling-системы (циклический сервер)](models/polling.ru.md) | циклический сервер по Q очередям с переключением, псевдо-закон сохранения |
| [Системы с отпусками (Vacations)](models/vacations.ru.md) | многократные отпуска, N-policy, разогрев/охлаждение, ненадёжный прибор |
| [Системы с отрицательными заявками](models/negative.ru.md) | отрицательные заявки: RCS и disasters, одно- и многоканальные |
| [Fork-Join системы](models/fork-join.ru.md) | параллельное обслуживание fork-join и split-join; точный тяжёлохвостый (Pareto) максимум n подзадач; гетерогенные ветви, series-parallel DAG задач, (n,k)-join |
| [Системы с пакетным поступлением](models/batch.ru.md) | пакетное поступление Mˣ/M/1 и групповое обслуживание M/M^[a,b]/1 (или общее Erlang/H2-подогнанное обслуживание батча) |
| [Occupancy-зависимый continuous batching](models/continuous-batching.ru.md) | continuous batching LLM-инференса: occupancy-зависимая интенсивность обслуживания, жёсткий потолок занятости, точные моменты и хвост времени ожидания |
| [Системы с нетерпеливыми заявками](models/impatience.ru.md) | нетерпеливые заявки: M/M/1/D и Erlang-A со staffing |
| [Retrial-очереди (повторные попытки)](models/retrial.ru.md) | retrial-очереди с орбитой (M/M/1, M/G/1) |
| [Матрично-аналитические модели (MAP/PH)](models/map-ph.ru.md) | коррелированные потоки: MAP/PH/1, MAP/M/c, MAP/PH/c, BMAP-варианты, фиттинг MMPP |
| [Multiserver-job системы (MSJ)](models/msj.ru.md) | PH-FCFS аналитика; general-service backfilling, прогнозы, FirstFit/MSF/Quickswap, бесплатный ServerFilling, цена checkpoint/resume и защита полезного интервала |
| [Балансировка нагрузки / диспетчеризация (mean-field)](models/load-balancing.ru.md) | диспетчеризация power-of-d / JSQ / JIQ, mean-field |
| [Нестационарные очереди Mₜ/M/c (переменная нагрузка)](models/time-varying.ru.md) | нестационарные Mₜ/M/c: PSA и MOL |
| [Age of Information (AoI, свежесть информации)](models/aoi.ru.md) | Age of Information: средний и пиковый возраст |
| [SLA / вероятность нарушения дедлайна](models/sla.ru.md) | Горизонтальная утилита: `P(W > D)` / SLO-квантиль по моментам из fit, для любой модели; пример LLM-serving TTFT SLO |
| [EDF-планирование](models/edf.ru.md) | Earliest-Deadline-First как реальная дисциплина обслуживания (не post-hoc SLA); точна по построению (DES), точной теории нет (открытая задача) |
| [Admission control по дедлайну](models/admission-control.ru.md) | Приём/отклонение при приходе по осуществимости дедлайна (не переупорядочивание); точный сходящийся ряд для Exp(θ) |
| [Queueing-inventory системы](models/inventory.ru.md) | M/M/1, M/M/c, или c гетерогенных серверов (одинаковых, экспоненциальных или с Erlang-/H2-подгонкой на сервер) с расходуемым при обслуживании запасом, общая политика (s,S) (экспоненциальное или Erlang-подогнанное время поставки), backorder или lost sales — точный QBD |
| [Закрытые системы](models/closed.ru.md) | системы с конечным числом источников (Engset) |
| [Надёжность: ненадёжные приборы](models/reliability.ru.md) | отказы и ремонты (M/G/1, M/M/c), machine repair problem (включая 2 гетерогенных ремонтника), working breakdowns, катастрофы с ремонтом, retrial + отказы |
| [Сети массового обслуживания](models/networks.ru.md) | открытые/закрытые сети: декомпозиция, Джексон, QNA, MVA/Бьюзен, BCMP, G-сети, блокировки, fork-join станции |

## Сравнительная таблица моделей

Для MSJ аналитические результаты FCFS и симуляционные дисциплины имеют разные
области применимости. См. [ограничения packing](msj_packing.md),
[явные фазы checkpoint/resume](msj_checkpoint.md) и
[информационный контракт прогнозов](msj_age_runtime.md); дренирование конечной
трассы не доказывает устойчивость или стационарную хвостовую гарантию.

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
| M/M/c отказы и ремонты | MMcBreakdownsCalc | MMcBreakdownsSim | - | Независимые отказы, доступность, R ремонтников |
| Machine repair problem | MachineRepairCalc | MachineRepairSim | - | Конечный парк, тёплый резерв, R ремонтников (Palm) |
| Machine repair, 2 гетерогенных ремонтника | MachineRepairHeterogeneousCalc | MachineRepairHeterogeneousSim | - | Разные скорости ремонта, точная не-birth-death CTMC (Krishnamoorthi 1963) |
| M/M/1 working breakdowns | MM1WorkingBreakdownsCalc | MM1WorkingBreakdownsSim | - | Пониженная скорость во время ремонта (Kalidass-Kasturi) |
| M/M/1 катастрофы + ремонт | MM1DisasterRepairCalc | MM1DisasterRepairSim | - | Сброс очереди, фаза ремонта, P(down)=δ/(δ+η) |
| M/M/1 retrial ненадёжный | MM1RetrialUnreliableCalc | MM1RetrialUnreliableSim | - | Активные отказы, орбита, доступность |
| Fork-Join | ForkJoinMarkovianCalc | ForkJoinSim | - | Параллельное обслуживание |
| Fork-Join, series-parallel DAG | ForkJoinDAGCalc | - | - | Гетерогенные ветви, вложенный series/parallel граф задач |
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
| M/M/2, 2 класса, гетерогенные серверы | MM2PriorityHeterogeneousCalc | MM2PriorityHeterogeneousSim | Да | Точная не-birth-death CTMC (техника Krishnamoorthi 1963) |
| M/PH/k, m классов | RDRAPriorityPH, MPhPhK2Class | PriorityQueueSimulator | Да | Фазовое обслуживание (RDR §2.3) |
| M/G/1 накапливаемый приоритет | MG1AccumulatingPriorityCalc | AccumulatingPrioritySim | Да | Клейнрок/APQ, спектр FIFO <-> строгие приоритеты |
| M/M/n+M приоритет + нетерпение | MMnPriorityImpatienceCalc | MMnPriorityImpatienceSim | Да | Приоритетный Erlang-A, уходы по классам |
| MMAP/PH/1 приоритеты | MapPh1PriorityCalc | PriorityQueueSimulator("MAP") | Да | Коррелированный вход, NP/PR/RS, точная CTMC |
| M/M/1 retrial + приоритет | MM1RetrialPriorityCalc | MM1RetrialPrioritySim | Да | Очередь приоритетных + орбита |
| M/G/1 preemptive repeat (RS) | MG1PreemptiveRepeatCalc | PriorityQueueSimulator("RS") | Да | Точный RS; completion times Гавера для RW |
| Multiserver-job FCFS | MsjExactCalc, MsjSaturatedCalc, MsjPHCalc | MsjSim, MsjGeneralSim | - | Экспоненциальная/PH CTMC малых систем; анализ устойчивости насыщенной системы |
| MSJ general-service дисциплины | - | MsjGeneralSim | - | FCFS/EASY/conservative; без прогнозов FirstFit/MSF/MSFQ/Adaptive Quickswap; MSFQ требует K из {1,k} |
| MSJ ServerFilling | - | MsjGeneralSim | - | k и K — степени двойки; бесплатный preemptive-resume, не SRPT; W включает паузы, есть журнал исполнения |
| [MSJ с ценой checkpoint/resume](msj_checkpoint.md) | - | MsjCheckpointSim | - | k и K — степени двойки; gated-расширение SF, детерминированный overhead удерживает K серверов; выделенная и полезная загрузка разделены, теоремы устойчивости нет |
| [MSJ с защитой полезного интервала](msj_protected_service.md) | - | MsjCheckpointSim | - | Опция min_service_time после каждого полезного старта, сброс после resume; события истечения защиты, независимый offline-подбор; меньше прерываний не означает меньшую задержку |
| [Калибровка MSJ по реальной трассе](real_trace_calibration.md) | - | MsjGeneralSim + SWF-адаптер | - | Историческая подвыборка завершённых работ; обучение empirical/Exp/PH/lognormal только по прошлому, точные K и времена прихода; не реконструкция исходного кластера |
| [История и зависимое обслуживание](real_trace_temporal.md) | - | ConditionalEmpirical + MsjGeneralSim | - | Несколько временных границ; отдельно среднее, форма, точный K и циклические блоки рангов с rank-iid-контролем; условные MC-оценки, не новый планировщик |
| [MSJ: начальное состояние и исходы](real_trace_lifecycle.md) | - | MsjLifecycleSim | - | Opt-in running/waiting, занятый ресурс до отмены и жёсткий runtime limit; отдельный учёт успеха/отмены/timeout, не реконструкция отмен в очереди |
| [Калибровка по современной GPU-трассе](modern_gpu_trace.md) | - | AcmeTrace + MsjLifecycleSim | - | Аудит timestamp Kalos и раздельные метки отказов; empirical replay номинального однородного GPU-пула, не аппаратная загрузка и не исторический планировщик |
| Балансировка нагрузки (power-of-d, JSQ, JIQ) | LoadBalancingMeanField | LoadBalancingSim | - | Диспетчеризация по большому пулу (mean-field) |
| Polling (циклический сервер) | PollingCalc | PollingSim | - | Switchover, exhaustive/gated, псевдо-закон сохранения |
| Нестационарная Mₜ/M/c | TimeVaryingMMcCalc | TimeVaryingMMcSim | - | Переменная нагрузка, приближения PSA и MOL |
| Age of Information | AoICalc, LcfsPreemptiveAoICalc | AoISim | - | Средний и пиковый AoI |
| M/M^[a,b]/1 групповое обслуживание | BulkServiceMM1Calc | BulkServiceSim | - | Пакетное обслуживание, батчинг LLM; точные моменты N/W при a=1 |
| M/Erlang(k)^[a,b]/1 групповое обслуживание | BulkServiceErlangCalc | - | - | Общее (Erlang-подогнанное, CV≤1) время обслуживания батча; сводится к k=1 выше |
| M/H2^[a,b]/1 групповое обслуживание | BulkServiceH2Calc | - | - | Общее (H2-подогнанное, CV≥1) время обслуживания батча; сводится к p1=1 выше |
| Engset | Engset | QueueingFiniteSourceSim | - | Конечное число источников |
| M/M/1 queueing-inventory (s,S) | MM1QueueingInventoryCalc | MM1QueueingInventorySim | - | Backorder или lost sales, точный QBD |
| M/M/c queueing-inventory (s,S) | MMcQueueingInventoryCalc | MMcQueueingInventorySim | - | c одинаковых серверов, точный QBD, сводится к c=1 выше |
| M/M/2 queueing-inventory, гетерогенные | MM2QueueingInventoryHeterogeneousCalc | MM2QueueingInventoryHeterogeneousSim | - | 2 сервера с разной скоростью, точный QBD, сводится к одинаковым c=2 выше |
| M/M/c queueing-inventory, гетерогенные | MMcQueueingInventoryHeterogeneousCalc | MMcQueueingInventoryHeterogeneousSim | - | Общее c серверов с разной скоростью, точный QBD, сводится к c=2 и одинаковым серверам выше |
| M/H2/c queueing-inventory, гетерогенные | MMcQueueingInventoryHeterogeneousH2Calc | MMcQueueingInventoryHeterogeneousH2Sim | - | У каждого сервера своё H2-подогнанное (неэкспоненциальное) обслуживание, точный QBD, сводится к обычным гетерогенным c выше |
| M/Erlang/c queueing-inventory, гетерогенные | MMcQueueingInventoryHeterogeneousErlangCalc | MMcQueueingInventoryHeterogeneousErlangSim | - | У каждого сервера своё Erlang-подогнанное (неэкспоненциальное, CV≤1) обслуживание, точный QBD, сводится к обычным гетерогенным c выше |
| M/M/1 queueing-inventory, Erlang-пополнение | MM1QueueingInventoryErlangReplenishmentCalc | MM1QueueingInventoryErlangReplenishmentSim | - | Erlang-подогнанное (неэкспоненциальное) время пополнения склада, точный QBD, сводится к Exp(theta) выше при r=1 |
| EDF-планирование | - (точной теории нет, см. docs/research/edf-scheduling-2026.md) | EDFQueueSim | - | Дисциплина обслуживания по дедлайну, точна по построению (DES), проверка законом сохранения работы |
| M/M/1 admission control по дедлайну | MM1DeadlineAdmissionControlCalc | MM1DeadlineAdmissionControlSim | - | Дедлайн Exp(θ), точный сходящийся ряд (level-crossing функциональное уравнение) |
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
