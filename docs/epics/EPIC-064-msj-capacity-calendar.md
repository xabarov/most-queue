# EPIC-064: MSJ с календарём ёмкости и явными VC-сценариями

- **Статус:** done
- **Предложен:** 2026-10-03
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)
- **Основание:** [EPIC-063](EPIC-063-resource-observability-audit.md)

## Цель

Исследовать нестационарную resource capacity на проверенном Helios, отделяя
модельную гипотезу дневного ограничения от неизвестных production quotas.
Аудит не разрешил интерпретировать daily VC counts как точный hard cap.
Поэтому fixed aggregate pool, daily aggregate pool и изолированные VC должны
быть разными сценариями с явно обозначенными допущениями, не реконструкцией.

## Контракт, который нужно зафиксировать до replay

- Отдельный opt-in simulator/adapter; прежние constant-capacity результаты
  и dispatch semantics не менять. Capacity events должны работать без arrivals.
- Явный календарь с областью определения, нет forward/backfill за пределами
  известных дат. Суточные effective times — модельное соглашение, а не наблюдение.
- Grandfathering уже запущенных jobs при снижении capacity: не прерывать и не
  переписывать S. Пока occupied>capacity, не запускать новые. Проверить reservation
  feasibility будущих стартов; EASY/Conservative не считать совместимыми автоматически.
- Необслуживаемые demands и jobs за горизонтом — явные infeasible/unresolved
  результаты. Не удалять их, не уменьшать K и не ждать бесконечно.
- Frozen cohorts и control S, carry-in/terminal/nonterminal accounting и unknown
  initial state описать до расчёта. CPU-only не объявлять безвредными для реального
  кластера, даже если scalar GPU модель их исключает.
- Ограниченный заранее заданный набор окон и policies, без tuning daily boundary,
  capacity или cohort по test W. Сравнивать успехи и unresolved вместе с T/W.

## Работы и приёмка

- [x] Финализировать протокол и список поддерживаемых policies до replay.
- [x] Строгий capacity calendar, simultaneous events, shrink/growth и явный horizon.
- [x] Аналитические и regression fixtures: constant calendar, no-arrival changes,
  grandfathering, impossible demand, reservation/calendar mismatch.
- [x] Helios adapter с raw hashes, label/identity guards и аудитом исключений.
- [x] Fixed/daily/VC сценарии, повтор, независимая проверка resource accounting.
- [x] Полный/быстрый pytest, документация и отчёт без production/causal claims.

Точный production replay остаётся no-go без intraday quota/borrowing/placement
и initial-state наблюдений. Не считать ненулевое historical W или policy ties
подтверждением точности модели.

## Результаты

`CapacityCalendar` (`most_queue/sim/utils/msj_capacity_calendar.py`) —
кусочно-постоянный, право-непрерывный календарь с явным доменом, без
forward/backfill; `constant_calendar`/`daily_calendar` — единственные
конструкторы. `MsjLifecycleSim.run_capacity_calendar` — отдельный opt-in
метод (constant-capacity путь `run_lifecycle`/`run_trace` не изменён):
grandfathering автоматический через подмену `self.k`/`NonpreemptivePacking.k`
на `calendar.capacity_at(now)` перед каждым dispatch; capacity-breakpoints —
полноценные события цикла (работают без arrivals); задания с need выше
`calendar.max_capacity()` помечаются `infeasible` и исключаются из
диспетчеризации, не блокируя FCFS head-of-line; реплей останавливается на
более раннем из полного дренажа и `calendar.domain_end`, оставляя
`running`/`waiting` как явные нетерминальные результаты; падение ёмкости,
инвалидирующее Conservative-резервацию, перехватывается как явный
`reservation_mismatch_time`, а не переинтерпретируется молча.

23 юнит-теста (`tests/units/test_msj_capacity_calendar.py`) фиксируют примитив
календаря и каждое из этих свойств, включая regression против `run_trace` при
широком постоянном календаре по всем шести `LIFECYCLE_POLICIES`. Helios-адаптер
(`examples/msj_capacity_calendar_experiment.py`) переиспользует верифицированный
архив и парсинг EPIC-063 без нового источника/download-пути; три предписанных
сценария (fixed/daily aggregate pool, isolated VC) на одном предписанном
7-дневном окне кластера Venus — 18 расписаний, побайтовый повтор и независимая
pandas-сверка (`works/msj_capacity_calendar/verify.py`), не импортирующая
основной скрипт.

В выбранном окне общий пул Venus не менялся (первое изменение — на 98-й день
всей 181-дневной трассы), поэтому fixed/daily-сценарии совпали, а ни один
infeasible job или reservation mismatch не возник ни у одной дисциплины —
честное ограничение предписанного окна, а не признак непроверенной механики
(shrink/growth/grandfathering/mismatch зафиксированы юнит-тестами на
синтетике). EASY/Conservative реально backfill'ят на изолированном VC
(до 155/990 заданий), не меняя итоговые completed/running counts относительно
остальных четырёх дисциплин в этом окне. Полный/быстрый
`pytest tests/ -m "not slow" -n auto` проходит; pylint `most_queue` — без
новых замечаний (10.00/10 на новых файлах); black/isort чисты.
[Отчёт](../research/msj-capacity-calendar-results-2026-10.md).
