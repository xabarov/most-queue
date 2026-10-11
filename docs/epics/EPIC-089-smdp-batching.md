# EPIC-089: Выбор запуска пакета как SMDP — Л6

- **Статус:** done
- **Создан:** 2026-10-11
- **Roadmap:** [Л6](../roadmaps/llm_serving_catchup_roadmap.md)

## Контракт

[Xu, Sun, Zhou, Niu, arXiv:2301.12865v3](https://arxiv.org/html/2301.12865v3):
Poisson, nonpreemptive batch service, deterministic size-dependent time/energy;
действие 0 — ждать следующего прихода, действие a>0 — запустить a запросов.
Оптимизируем response/power average objective конечной SMDP с отдельным
overflow-состоянием и cost rate c_o; RVI после uniformization.
Это воспроизведение, не заявка на новую теорию/глобальный optimum бесконечной очереди.

## Задачи

- [x] Основной источник целиком, локальная классика/монография и четыре
      направления поиска; сохранение первоисточника и глубины чтения.
- [x] BaseQueue API, finite overflow kernel, reward/sojourn, compressed RVI,
      gain/Bellman/balance diagnostics, independent policy evaluation.
- [x] Отдельный unbounded event-time DES: admission at decision epochs,
      FCFS batches, horizon without drain, request/queue-area conservation.
- [x] Independent occupation LP и exhaustive small-policy oracle;
      time-change/bias scaling, numerical quadrature, hand replay, M/D/1 corner.
- [x] Reproducible published-profile/synthetic experiments, truncation/penalty
      sensitivity и явная проверка постоянного tail action в реальной очереди.
- [x] EN/RU модель/схема, README/models/simulation, отчёт, эпик/roadmap.
- [x] Target/slow/full/fast, black/isort/pylint/pre-commit, snippets/links.
- [x] Приватные диссертационные карточки/леммы/обзор/гипотезы/план/журнал,
      snapshot archive + SHA-256; сохранить исходную часть журнала.

## Границы

Overflow contribution Δ — diagnostic, не доказанная truncation error bound.
Decision-epoch distribution не равен time distribution реального числа запросов.
Control-limit structure наблюдается, не навязывается solver как теорема.
Нет hard SLO, random service, idle baseline power, token/KV mechanics или
новой GPU calibration. Finite policy переносится на unbounded queue только
с явным tail action; его drift проверяется отдельно от finite Bellman optimum.

## Результат и проверка

[Отчёт](../research/smdp-batching-validation-2026.md),
[JSON](../research/smdp-batching-results-2026.json),
[API EN/RU](../models/smdp-batching.ru.md).
66 targeted; отдельный slow1passed; full3141passed/439.78s;
fast3130passed/334.25s, 16 прежних warnings, без reruns.
Production/experiment pylint10/10; package9.99/10 с прежними сообщениями.
Black/isort/pre-commit, исполняемые snippets, local links и visual inspection
двух схем проверены. Source HTML/text сохранены, §§I–VI/bibliography прочитаны.
Монография Puterman искалась, TOC/full text не доступны; соседние proof conditions
не объявляются независимо аудированными. Это не препятствие атрибутированной
репликации согласованной finite model и не основание новой научной заявки.

Independent source LP: gain difference4.60e−7. Printed TableI gain отличается
на≈.0054%; причина не установлена, author software не проверялся.
10 table rows, 16 sensitivity settings, source LP, 42DES×4seeds +3counter
horizons. Все прогоны сохраняют запросы/площади; paired arrival streams проверены.
Два запуска JSON побайтно одинаковы: SHA256
`6d46881822d72c053e8ee511568180578fcd6079365ae905812b1da84eec45cf`.
Отдельный stress показывает опасность positive real-tail drift несмотря
на малый Δ. Stable tail repair — другая политика, не infinite optimum.
Приватные материалы/журнал и 92 архивных снимка синхронизированы с SHA256;
семь тестов синхронизатора проходят, исходный journal prefix сохранён.
Л5 ранее отправлен как `cc6741f`; новые Л6 изменения пока не закоммичены.
Следующий пункт Л7 требует проверки доступа к закрытому первоисточнику.
