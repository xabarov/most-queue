# EPIC-064: MSJ с календарём ёмкости и явными VC-сценариями

- **Статус:** proposed
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

- [ ] Финализировать протокол и список поддерживаемых policies до replay.
- [ ] Строгий capacity calendar, simultaneous events, shrink/growth и явный horizon.
- [ ] Аналитические и regression fixtures: constant calendar, no-arrival changes,
  grandfathering, impossible demand, reservation/calendar mismatch.
- [ ] Helios adapter с raw hashes, label/identity guards и аудитом исключений.
- [ ] Fixed/daily/VC сценарии, повтор, независимая проверка resource accounting.
- [ ] Полный/быстрый pytest, документация и отчёт без production/causal claims.

Точный production replay остаётся no-go без intraday quota/borrowing/placement
и initial-state наблюдений. Не считать ненулевое historical W или policy ties
подтверждением точности модели.
