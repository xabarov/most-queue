# EPIC-046: Time-limited (T-policy/таймер) дисциплина обслуживания

- **Статус:** proposed
- **Создан:** 2026-09-30

## Цель

Новая дисциплина: сервер обслуживает очередь не до опустошения, а максимум время `T` (таймер),
затем переключается (уходит на паузу/к другой очереди) — по мотивам "Waiting time analysis for
a queueing system with time-limited service and exponential timer" (Naval Research Logistics,
2001, doi:10.1002/nav.1039) и "time-limited service priority queueing system with exponential
timer and server vacations" (Queueing Systems, 2007, doi:10.1007/s11134-007-9055-4).

## Контекст

Кандидат из литературного обзора 2026-09-30. Реалистична для систем с квотами/тайм-слайсами;
частично похожа на polling (switchover), но однa-очередь-с-таймером — отдельная модель.

## Задачи

- [ ] Обзор литературы.
- [ ] Roadmap.
- [ ] Реализация + тесты.
- [ ] Документация, память.

## Результаты

_Не начато._
