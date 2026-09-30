# EPIC-037: Unified Erlang/H2 auto-dispatch for bulk-service batch-service time

- **Статус:** done
- **Создан:** 2026-09-30
- **Roadmap:** [../roadmaps/bulk_service_auto_dispatch_roadmap.md](../roadmaps/bulk_service_auto_dispatch_roadmap.md)

## Цель

Закрыть последний общий резерв EPIC-035/EPIC-036: единая точка входа, которая сама выбирает
Erlang (CV≤1) или H2 (CV≥1) калькулятор bulk-service по CV переданных моментов, вместо ручного
выбора вызывающим кодом — по аналогии с `theory.utils.sla.fit_from_moments(family="auto")`.

## Контекст

Обзор: [../research/bulk-service-auto-dispatch-2026.md](../research/bulk-service-auto-dispatch-2026.md).
Чисто инженерная задача — новой теории не требуется, обе базовые CTMC (`BulkServiceErlangCalc`,
`BulkServiceH2Calc`) уже провалидированы в EPIC-035/036. Только диспетчеризация по CV, в отличие
от `sla.fit_from_moments` использующая Erlang (не Gamma) для CV≤1, так как именно Erlang — тот
фазовый член семейства Gamma, что совместим с уже реализованной техникой расширения CTMC.

## Задачи

### Ядро

- [x] `most_queue/theory/batch/bulk_service_general.py` (новый файл): `fit_bulk_service_calc(a, b,
      moments, family="auto", queue_truncation=300)` — возвращает настроенный по `set_servers`
      (`set_servers_from_moments`) экземпляр `BulkServiceErlangCalc`/`BulkServiceH2Calc`.

### Тесты

- [x] `family="auto"` выбирает Erlang для CV≤1, совпадает с прямым конструированием.
- [x] `family="auto"` выбирает H2 для CV≥1, совпадает с прямым конструированием.
- [x] Принудительные `family="erlang"`/`"h2"` — валидация CV-совместимости.
- [x] Невалидный `family` — `ValueError`.

### Документация

- [x] `docs/models/batch.md`+`.ru.md` — заметка про диспетчер после подсекций Erlang/H2.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. `family="auto"` даёт тот же результат, что и ручной выбор семейства по CV.
2. Явные `family="erlang"/"h2"` не скрывают несовместимость по CV — кидают `ValueError`.

## Результаты

Новая функция-фабрика `fit_bulk_service_calc(a, b, moments, family="auto", queue_truncation=300)`
(`most_queue/theory/batch/bulk_service_general.py`) — единая точка входа поверх
`BulkServiceErlangCalc` (EPIC-035) и `BulkServiceH2Calc` (EPIC-036): по CV переданных моментов
(`cv ≤ 1` → Erlang, `cv > 1` → H2) сама выбирает и настраивает нужный калькулятор через
`set_servers_from_moments`, по аналогии с `theory.utils.sla.fit_from_moments(family="auto")`.
Явные `family="erlang"`/`"h2"` проверяют CV-совместимость и кидают `ValueError`, а не молча дают
неверную подгонку.

Чисто инженерная задача — новой теории не потребовалось, обе базовые CTMC уже провалидированы в
предыдущих эпиках; риск был минимален, как и планировалось при выборе этого направления.

**Тесты:** 7 новых (`tests/units/test_bulk_service_general.py`) — auto выбирает Erlang/H2 и
совпадает с прямым конструированием на обоих; граничный `cv=1` разрешается в Erlang; форсированные
`family="erlang"`/`"h2"` отклоняют несовместимый CV; форсированный `h2` требует 3 момента;
невалидный `family` — `ValueError`. Полный набор тестов запущен в фоне для подтверждения отсутствия
регрессий.

**Документация:** `docs/models/batch.md`+`.ru.md` — новая подсекция «Auto-dispatch» после подсекций
Erlang/H2, со ссылкой на `fit_bulk_service_calc`.

Резерв EPIC-035/EPIC-036 (точные моменты выше среднего, batch-size-зависимые параметры) остаётся
открытым — этот эпик унифицировал только выбор семейства, не расширил точность/охват ни одного из
калькуляторов.
