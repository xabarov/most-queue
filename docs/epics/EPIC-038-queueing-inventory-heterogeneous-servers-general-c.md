# EPIC-038: Queueing-inventory с общим числом c гетерогенных серверов

- **Статус:** done
- **Создан:** 2026-09-30
- **Roadmap:** [../roadmaps/queueing_inventory_heterogeneous_servers_general_c_roadmap.md](../roadmaps/queueing_inventory_heterogeneous_servers_general_c_roadmap.md)

## Цель

Обобщить EPIC-033 (`MM2QueueingInventoryHeterogeneousCalc`, ровно c=2 гетерогенных сервера) на
произвольное число гетерогенных серверов c — резерв, отложенный три эпика подряд (EPIC-036,
EPIC-037) в пользу более быстрых побед.

## Контекст

Обзор: [../research/queueing-inventory-heterogeneous-servers-general-c-2026.md](../research/queueing-inventory-heterogeneous-servers-general-c-2026.md).
Для гетерогенных (не идентичных) серверов расщепление состояний должно отслеживать, КАКИЕ именно
серверы заняты, а не только сколько — число конфигураций на уровне n растёт как `C(c,n)`, суммарно
`2^c - 1` конфигураций на границе. Решение: механическая конструкция CTMC (canonicalize-паттерн
EPIC-025, избегает ручного вывода) поверх QBD со сложенным граничным суперблоком (EPIC-028/033).
`c=2` должно точно воспроизводить `MM2QueueingInventoryHeterogeneousCalc`, `mu_1=...=mu_c` — точно
воспроизводить `MMcQueueingInventoryCalc`.

## Задачи

### Ядро

- [x] `most_queue/theory/inventory/mmc_heterogeneous_inventory.py` (новый файл):
      `MMcQueueingInventoryHeterogeneousCalc` — CTMC над `(busy-подмножество, stock)` для
      границы, механическая конструкция через `arrival_target`/`departure_targets`.
- [x] `most_queue/sim/inventory.py`: `MMcQueueingInventoryHeterogeneousSim` — DES, явное
      отслеживание `busy` только при `n < c`.

### Тесты

- [x] `c=2` точно воспроизводит `MM2QueueingInventoryHeterogeneousCalc` (структурная регрессия).
- [x] `mu_1=...=mu_c` точно воспроизводит `MMcQueueingInventoryCalc` для c=1,3,4,5.
- [x] QBD residual пренебрежимо мал.
- [x] Ускорение одного сервера не увеличивает `E[W]`/stockout.
- [x] Валидные распределения вероятностей; отклонение невалидных параметров.
- [x] DES cross-validation для c=3.

### Документация

- [x] `docs/models/inventory.md`+`.ru.md` — новая подсекция (обобщение c=2 подсекции EPIC-033).
- [x] `docs/models.md`/`.ru.md` — новая строка.

## Критерии готовности (DoD эпика)

Общий DoD — [../DOD.md](../DOD.md). Специфично:

1. `c=2` — точная структурная регрессия к `MM2QueueingInventoryHeterogeneousCalc`.
2. `mu_1=...=mu_c` — точная регрессия к `MMcQueueingInventoryCalc` при c>2.
3. Теория сверена с независимой DES при c=3, разные ставки.

## Результаты

Новый класс `MMcQueueingInventoryHeterogeneousCalc`
(`most_queue/theory/inventory/mmc_heterogeneous_inventory.py`) — точная M/M/c queueing-inventory
система с c гетерогенными серверами, построенная механически (canonicalize-паттерн EPIC-025) из
трёх чистых функций (`_arrival_target`, `_departure_targets`, `_subset_service_rate`) поверх
QBD со сложенным граничным суперблоком (EPIC-028/033). Граница — `(busy-подмножество, stock)` для
`n=0..c-1` (`2^c - 1` подмножеств), повторяющаяся часть QBD — гомогенная при `n>=c` (все серверы
всегда заняты, подмножество не нужно). Плюс `MMcQueueingInventoryHeterogeneousSim`
(`most_queue/sim/inventory.py`) — DES, отслеживающий `busy` только при `n < c`.

**Найденная и исправленная ошибка:** первая версия `_mean_in_system()` использовала формулу
`boundary_term + c*p_ge_c + mean_extra`, скопированную по прямой аналогии с EPIC-033 (там `c=2`
было захардкожено как `p_ge_2 + mean_extra`, т.е. явно `1×p_ge_2` плюс скрытая `+1` внутри
`mean_extra` — суммарно ровно `2×p_ge_2`, что для `c=2` СЛУЧАЙНО совпадало с корректным
`c*p_ge_c`). При прямом обобщении на произвольное `c` эта случайность исчезла: `mean_extra`
(`pi1 @ (I-R)^-2 @ 1`) по тождеству матрично-геометрического среднего равна `sum_k (k+1)*P_k`, то
есть уже включает одну лишнюю `p_ge_c`. Правильная формула — `boundary_term + (c-1)*p_ge_c +
mean_extra`. Поймано немедленно на первом же прогоне регрессии `c=2` против
`MM2QueueingInventoryHeterogeneousCalc` (`E[W]` разошлось на ~20%, `E[S]` совпало точно — сузило
баг до `_mean_in_system`), до перехода к формальным тестам.

**Валидация:** `c=2` точно (до `1e-9`) воспроизводит `MM2QueueingInventoryHeterogeneousCalc` по
`v`, `w`, `stockout_prob`, `p` (структурная регрессия); `mu_1=...=mu_c` точно (до `~1e-15`)
воспроизводит `MMcQueueingInventoryCalc` для `c=1,3,4,5` (численная регрессия); DES
cross-validation при `c=3`, три разные ставки — совпало в пределах Monte Carlo допуска.

**Тесты:** 14 новых unit-тестов (`tests/units/test_mmc_queueing_inventory_heterogeneous.py`) + 2
новых DES cross-validation теста (`tests/test_inventory.py`). Полный набор тестов:
704 passed, 0 failed (299s, `pytest tests/ -n auto`).

**Документация:** `docs/models/inventory.md`+`.ru.md` — новая подсекция «Heterogeneous servers,
general c» после подсекции c=2 (убран устаревший "Limited to c=2" резерв-комментарий);
`docs/models.md`/`.ru.md` — новая строка.

**Литература:** doi:10.1016/j.matcom.2018.03.001 (2018, c=2 heterogeneous vs homogeneous
precedent), doi:10.1038/s41598-024-81593-7 (2024 Scientific Reports, подтверждает активность темы
multi-server heterogeneous queueing-inventory), doi:10.20944/preprints202310.1176.v1 (2023,
junior/senior servers — мотивация priority-order assignment convention). Полный текст статей
недоступен (paywall) — обоснование по title/abstract/DOI, вывод независимый, как и во всех
эпиках этой сессии.

**Резерв (не в этом эпике):** retrial и finite-source (2024-статьи расширяют модель этим
направлением); негетерогенное-по-времени обслуживание (Erlang/H2-подгонка поверх гетерогенных
серверов — новая комбинация техник EPIC-035/036 с EPIC-038, не пробовалась); приоритетные классы
поверх гетерогенных серверов при c>2 (EPIC-025 покрывает только c=2).
