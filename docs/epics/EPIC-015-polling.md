# EPIC-015: Polling-системы (циклический сервер)

- **Статус:** done (2026-07-11)
- **Создан:** 2026-07-11
- **Roadmap:** обзор трендов — [../research/queueing-trends-2026.md](../research/queueing-trends-2026.md) (вторая волна, п. B)

## Цель

Реализовать **polling-модели**: один сервер циклически обходит Q очередей с временами переключения
(switchover), обслуживая по дисциплинам exhaustive / gated / k-limited. Даёт средние времена
ожидания по очередям и **псевдо-закон сохранения**.

## Контекст

Классика (Takagi, Boxma–Groenevelt), до сих пор активна: token-ring, USB/Bluetooth polling,
IoT-шлюзы, обслуживание/ТОиР, светофоры. Не work-conserving из-за switchover. Известны:
- **Псевдо-закон сохранения**: взвешенная сумма средних ожиданий = ρ·W_0 + член от switchover (не
  зависит от дисциплины) — прямой аналог conservation law, уже показанного в туториале.
- Средние ожидания по очередям для exhaustive/gated (через workload-декомпозицию / mean-value
  анализ буферов).

Первоисточники: Takagi «Queuing analysis of polling models» (ACM CSUR); Boxma–Groenevelt
«Pseudo-conservation laws» (см. research).

## Задачи

- [x] **`PollingCalc`** (`theory/polling.py`): M/G/1 polling, Q очередей, switchover; exhaustive/
      gated — точный **псевдо-закон сохранения** `Σρ_iW_i` (Boxma–Groenevelt) для асимметрии +
      симметричное `W = Σρ_iW_i/ρ`; средняя длина цикла.
- [x] **`PollingSim`** (`sim/polling.py`): циклический сервер, switchover, exhaustive/gated,
      ожидания по очередям; сидируемый.
- [x] Валидация: PCL и симметричное W сверены с симуляцией (симметрично exhaustive/gated ~0.5%;
      асимметричный PCL 0.1%); gated хуже exhaustive. Тесты `test_polling.py` (6 шт.).
- [x] Каталог (EN+RU).

## Критерии готовности (DoD эпика)

Общий DoD ([../DOD.md](../DOD.md)). Специфично: псевдо-закон сохранения выполняется точно; средние
ожидания совпадают с симуляцией; Q=1 сводится к M/G/1 с отпусками.

## Результаты

- **`PollingCalc`** — псевдо-закон сохранения (точно, любая асимметрия) + симметричное ожидание по
  очереди; **`PollingSim`** — парный симулятор (per-queue, exhaustive/gated). Валидировано против
  симуляции. Каталог EN+RU, тесты (6 шт.).
