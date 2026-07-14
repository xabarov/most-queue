# EPIC-019: Ненадёжные приборы — M/M/c с отказами, machine repair, working breakdowns, катастрофы с ремонтом, retrial + отказы

- **Статус:** done (2026-07-15)
- **Создан:** 2026-07-15
- **Roadmap:** обзор — [../research/unreliable-queues-2026.md](../research/unreliable-queues-2026.md)

## Цель

Превратить «надёжность» из единственной модели (M/G/1 Avi-Itzhak–Naor) в полноценное
семейство: многоканальные системы с отказами серверов, классическая machine repair
problem, working breakdowns (деградация вместо остановки), катастрофы с фазой
ремонта (усиление negative-стека) и retrial с ненадёжным прибором (усиление
retrial-стека). Все модели марковские, решаются точно (конечные/усечённые CTMC),
каждая — с парным сидируемым симулятором.

## Контекст

Текущее состояние и ландшафт публикаций — в
[обзоре](../research/unreliable-queues-2026.md). Первоисточники: Mitrany–Avi-Itzhak
(Management Science 1968), Neuts–Lucantoni (Management Science 1979), Palm (1947) /
машинная интерференция, Kalidass–Kasturi (C&IE 2012), Towsley–Tripathi (ORL 1991),
Wang–Cao–Li (Queueing Systems 2001).

Новое семейство каталога: `docs/models/reliability.md` (+ru); существующий раздел
M/G/1 unreliable переезжает туда из страницы vacations.

## Задачи

### П.1 M/M/c с отказами и ремонтами серверов

- [x] `MMcBreakdownsCalc` (`theory/reliability/mmc_breakdowns.py`): серверы
      отказывают независимо (rate ξ, занятые и свободные), ремонт exp(η) —
      неограниченные ремонтники (вариант R ремонтников — параметром); усечённая
      CTMC (заявки × исправные серверы); метрики: v1/L, загрузка, доступность,
      распределение исправных серверов.
- [x] `MMcBreakdownsSim` — сидируемый симулятор.
- [x] Валидация: ξ→0 сводится к M/M/c (Erlang C, машинная точность);
      calc vs sim; доступность = биномиальная маргиналия η/(ξ+η).

### П.2 Machine repair problem (машинная интерференция)

- [x] `MachineRepairCalc` (`theory/reliability/machine_repair.py`): M машин
      (отказ ξ каждая работающая), R ремонтников (ремонт η), тёплый резерв S
      (spares, отказ ξ_s); конечное birth-death — точно; метрики: доступность
      парка, среднее число неисправных, загрузка ремонтников, throughput отказов.
- [x] `MachineRepairSim` — сидируемый симулятор.
- [x] Валидация: конечное birth-death vs sim; S=0, R=M сводится к независимым
      машинам (биномиальное распределение, машинная точность).

### П.3 M/M/1 working breakdowns

- [x] `MM1WorkingBreakdownsCalc` (`theory/reliability/mm1_working_breakdowns.py`):
      нормальная скорость μ, при поломке (rate ξ) — деградированная μ_d < μ,
      восстановление η; QBD с 2 фазами (усечённая CTMC); метрики: v1/L, загрузка,
      доля времени в деградации.
- [x] `MM1WorkingBreakdownsSim` — сидируемый симулятор.
- [x] Валидация: μ_d = μ сводится к M/M/1 (машинная точность); μ_d = 0 — к
      классическим поломкам; calc vs sim.

### П.4 M/M/1 с катастрофами и фазой ремонта

- [x] `MM1DisasterRepairCalc` (`theory/reliability/mm1_disaster_repair.py`):
      катастрофа (rate δ) сбрасывает всех заявок и выводит прибор в ремонт
      exp(η); прибывающие во время ремонта ждут; усечённая CTMC с фазами
      {исправен, ремонт}; метрики: v1/L, вероятность ремонта, q (доля обслуженных).
- [x] `MM1DisasterRepairSim` — сидируемый симулятор.
- [x] Валидация: η→∞ сводится к M/M/1 с мгновенным DISASTER — сверка с независимой
      замкнутой формой (геометрический закон, корень μz²−(λ+μ+δ)z+λ=0); P(down)=δ/(δ+η)
      точно; calc vs sim.

### П.5 M/M/1 retrial с ненадёжным прибором

- [x] `MM1RetrialUnreliableCalc` (`theory/reliability/retrial_unreliable.py`):
      орбита с повторами σ на заявку, прибор отказывает во время обслуживания
      (rate ξ), прерванная заявка уходит на орбиту, ремонт η; усечённая по орбите
      CTMC (орбита × {свободен, занят, ремонт}); метрики: средняя орбита, v1,
      доступность.
- [x] `MM1RetrialUnreliableSim` — сидируемый симулятор.
- [x] Валидация: ξ→0 сводится к классической M/M/1 retrial — сверка с независимой
      замкнутой формулой Falin–Templeton для средней орбиты; calc vs sim.

### Общее

- [x] Новое семейство каталога `docs/models/reliability.md` (+ru): переезд M/G/1
      unreliable из vacations-страницы, 5 новых моделей, схема семейства
      (существующая unreliable.png + новая схема machine repair в
      `generate_figures.py`), строка в хабе, строки в сравнительной таблице.
- [x] README (EN+RU): строка Reliability обновлена.
- [x] Тесты в `tests/` (сидированные), pylint/black; эпик done + Результаты.

## Критерии готовности (DoD эпика)

Общий DoD ([../DOD.md](../DOD.md)). Специфично:

- Каждая модель имеет проверку сводимости к известному частному случаю с жёстким
  допуском (Erlang C, M/M/1, биномиальная доступность, существующие
  disaster/retrial-калькуляторы).
- Каждая модель сверена с парным сидируемым симулятором.
- Усечения CTMC выбираются автоматически с контролем хвоста (документированный
  критерий).

## Результаты

Новое семейство «Надёжность»: пакет `most_queue/theory/reliability/` (5 калькуляторов,
общий sparse-CTMC решатель `utils.ctmc_stationary` с авторостом усечения по хвосту),
`most_queue/sim/reliability.py` (5 сидируемых симуляторов), 12 тестов
(`tests/test_reliability.py`). Ключевые проверки:

- `MMcBreakdownsCalc`: ξ→0 == Erlang C (1e-6); маргиналия исправных серверов ==
  Binomial(c, η/(ξ+η)) (1e-8); сим в пределах 5%.
- `MachineRepairCalc`: S=0, R=M == Binomial (1e-12); сим 2%.
- `MM1WorkingBreakdownsCalc`: μ_d=μ == M/M/1 (1e-8); доля деградации == ξ/(ξ+η) точно.
- `MM1DisasterRepairCalc`: P(down)=δ/(δ+η) (1e-8); η→∞ — геометрический закон
  M/M/1-катастроф (корень μz²−(λ+μ+δ)z+λ=0, 1e-5); сим 5%.
- `MM1RetrialUnreliableCalc`: ξ=0 == формула Falin–Templeton для орбиты (1e-8); сим 5%.

Каталог: новая страница семейства `docs/models/reliability.md` (+ru) — M/G/1 unreliable
переехал туда из vacations; схема `machine_repair` (EN+RU); строки в хабе, сравнительной
таблице и README (EN+RU). Обзор: `docs/research/unreliable-queues-2026.md`. Резерв
(п. 6 обзора): backup server, working breakdowns + vacations, неоднородные ремонтники,
BMAP/G/1 retrial + breakdowns.
