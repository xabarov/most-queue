# EPIC-020: Приоритеты, волна 2 — accumulating priority, нетерпение, MAP-вход, retrial, preemptive-repeat

- **Статус:** proposed
- **Создан:** 2026-07-15
- **Roadmap:** обзор — [../research/priority-queues-2026.md](../research/priority-queues-2026.md)

## Цель

Вывести приоритетный стек за пределы статических классов: динамические
(accumulating) приоритеты для триажа и SLA, приоритеты с нетерпением для
call-центров, коррелированный (MAP) вход, приоритеты в retrial-моделях и
аналитика preemptive-repeat дисциплин, которые давно есть в симуляторе. Плюс
мелкие долги стека (экспорты, рассинхрон документации).

## Контекст

Приоритетный стек — один из самых сильных в библиотеке (EPIC-009: RDR-A, точные
CTMC-эталоны, дисперсия отклика), но весь построен на статических классах с
пуассоновским входом и без нетерпения. Ландшафт публикаций и ранжирование — в
[обзоре](../research/priority-queues-2026.md). Первоисточники: Stanford–Taylor–Ziedins
(Queueing Systems 2013), Sharif–Stanford (ORHC 2014), Iravani–Balcıoğlu (QS 2008),
Choi (QS 2001), Takine (JORSJ 1996), Horváth (Performance Evaluation 2012),
Operational Research 2015 (retrial PR), Gaver (JRSS-B 1962).

## Задачи

### П.1 Accumulating priority queue (APQ)

- [ ] `MG1AccumulatingPriorityCalc` (`theory/priority/accumulating.py`): M/G/1 APQ,
      2+ классов с коэффициентами накопления b_k; средние ожидания по классам
      (Stanford–Taylor–Ziedins 2013), опционально — распределения (ЛСТ + численное
      обращение по образцу существующих решателей).
- [ ] `AccumulatingPrioritySim`: симулятор с выбором заявки по максимальному
      накопленному кредиту (b_k · время ожидания); сидируемый.
- [ ] Валидация: b_k → 0/∞ сводится к NP-приоритетам / FIFO (сверка с
      `MG1NonPreemptiveCalc` и `MG1Calc`); calc vs sim; кейс триажа в тесте.

### П.2 Приоритеты + нетерпение (M/M/n + M, 2 класса)

- [ ] `MMnPriorityImpatienceCalc` (`theory/priority/impatience.py`): 2 класса,
      per-class интенсивности ухода θ_k, дисциплина NP (вариант PR — по
      возможности); усечённая CTMC (инфраструктура `theory/reliability/utils.py`);
      метрики: средние w/v, вероятности ухода по классам, occupancy.
- [ ] Симулятор с нетерпением по классам (расширение `ImpatientQueueSim` или
      отдельный сидируемый).
- [ ] Валидация: один класс сводится к Erlang-A (`MMnImpatienceCalc`); θ→0 — к
      приоритетам без нетерпения; calc vs sim.

### П.3 MAP/PH/1 с приоритетами (2 класса)

- [ ] `MapPh1PriorityCalc` (`theory/priority/map_ph_priority.py`): MAP-вход,
      маркированный по классам (MMAP[2]), PH-обслуживание per class, дисциплины NP
      и PR; QBD-решение (утилиты MAP-стека) либо усечённая CTMC; метрики: средние
      w/v по классам, загрузка.
- [ ] Валидация: MAP=Poisson сводится к M/PH/1 приоритетам (`MG1NonPreemptiveCalc`
      / `MG1PreemptiveCalc` на PH-моментах); сверка с `PriorityQueueSimulator`
      (вход MAP уже поддержан симулятором через `create_distribution`).

### П.4 M/M/1 retrial с приоритетами (2 класса)

- [ ] `MM1RetrialPriorityCalc` (`theory/retrial/priority.py` или
      `theory/priority/retrial_priority.py`): приоритетные заявки ждут в очереди,
      обычные при занятом приборе уходят на орбиту (retrial γ); усечённая CTMC
      (очередь приоритетных × орбита × состояние прибора); метрики: средние по
      классам, средняя орбита.
- [ ] Валидация: λ_1→0 сводится к `MM1RetrialCalc`; сверка с симулятором
      (расширение `RetrialQueueSim` двумя классами или отдельный сидируемый сим).

### П.5 M/G/1 preemptive-repeat — аналитика (RS и RW)

- [ ] `MG1PreemptiveRepeatCalc` (`theory/priority/preemptive/mg1_repeat.py`):
      2+ классов, дисциплины preemptive repeat with resampling (RS) и without
      (RW) через completion-time технику (Gaver 1962; та же схема, что в
      `MG1UnreliableCalc`); средние (моменты — где техника позволяет).
- [ ] Валидация: сверка с существующим `PriorityQueueSimulator(prty_type="RS"/"RW")`
      — впервые закрывает разрыв «дисциплина есть в симе, теории нет»;
      экспоненциальное обслуживание: RS == PR (memoryless, машинная точность).

### Общее

- [ ] Экспортировать из `theory/priority/__init__.py` недостающие классы EPIC-009:
      `MMkPriorityExact`, `RDRAPriorityCalc`, `RDRAPriorityPH`, `MPhPhK2Class`.
- [ ] Починить рассинхрон `docs/priorities.md` / `docs/models/priority.md` с
      фактическими сигнатурами (`MG1PreemptiveCalc` без `num_of_classes` и т.п.);
      расширить docs-imports guard на исполнение примеров? — минимум выровнять
      примеры.
- [ ] Каталог: новые модели на страницу `docs/models/priority.md` (+ru), строки в
      сравнительной таблице и README; схема APQ (`generate_figures.py`, EN+RU).
- [ ] Тесты сидированные; black/isort/pylint; эпик done + Результаты.

## Критерии готовности (DoD эпика)

Общий DoD ([../DOD.md](../DOD.md)). Специфично:

- Каждая модель заперта сводимостью к существующему решателю библиотеки с жёстким
  допуском (NP/FIFO для APQ, Erlang-A, M/PH/1-приоритеты, MM1RetrialCalc, PR для
  RS при exp).
- П.5 сверен с существующим симулятором RS/RW — это первый аналитический
  бенчмарк для этих дисциплин в библиотеке.
- Усечения CTMC — с контролем хвоста (переиспользовать
  `theory/reliability/utils.ctmc_stationary`).

## Результаты

Заполняется по завершении.
