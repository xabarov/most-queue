# EPIC-091: Групповое обслуживание гетерогенными приборами — Л8

- **Статус:** done
- **Создан:** 2026-10-11
- **Roadmap:** [Л8](../roadmaps/llm_serving_catchup_roadmap.md)

Переход к Л8 явно выбран пользователем; Л7 остаётся waiting_source.

## Контракт

[van Ommeren, Baer, Mishra, Roy (2020)](https://doi.org/10.1007/s11134-020-09654-y),
Queueing Systems 95:251–269. Poisson, общая бесконечная FCFS очередь,
Nσ экспоненциальных пакетных приборов каждого типа, собственные Tσ/Bσ/μσ.
Длительность пакета не зависит от размера; приоритет — порог, затем порядок типов.
Точные геометрический BP и конечный FP; PH ожидания и отклика по Теореме 2/Remark 1.
Это воспроизведение существующего результата, не новая теория и не G-service.

## Задачи

- [x] Полный основной источник, локальная классика, специализированная
      диссертация и четыре стороны поиска; сохранять глубину чтения.
- [x] BaseQueue API, sparse FP, exact infinite BP, PH W/V moments/tails/LST,
      потоки по типам, занятость, P(W=0), residual/resource guards.
- [x] Независимый индивидуальный FCFS DES, horizon without drain,
      полностью наблюдаемые cohort tails, accounting/replay.
- [x] Physical capped CTMC + uncompressed tagged chain, Erlang C,
      имеющийся identical bulk solver, permutation/regrouping и ручной replay.
- [x] Все 50 строк Tables 2–3, priority-label discrepancy с арбитрами,
      CTMC refinement и synthetic DES experiments/JSON.
- [x] EN/RU модель/схемы/каталоги/README/simulation, отчёт/roadmap.
- [x] Target/slow/full/fast, formatting/lint/hooks, snippets/links.
- [x] Приватные source cards/lemmas/synthesis/plan/hypotheses/append-only
      journal, source PDF/text и snapshot archive/SHA-256.

## Границы

Нет неэкспоненциального или batch-size-dependent обслуживания, timeout,
приоритета по deadlines, token/KV/migration или GPU calibration.
Queue PMF — только ожидающие; p/pi общего населения не подменяются этой PMF.
Матрицы обслуживания PH не используют ложную независимость W и типа прибора.
CTMC cap — только независимый приближённый эталон, не часть точного калькулятора.
Resource limits и guard около ρ=1 явно численные, не новая граница устойчивости.

## Результат

[Отчёт](../research/heterogeneous-bulk-service-validation-2026.md),
[JSON](../research/heterogeneous-bulk-service-results-2026.json),
[API](../models/heterogeneous-bulk-service.ru.md).
68targeted/4.05s, отдельная slow3/4.53s;
full3213passed/485.83s, fast3199passed/389.71s,
16 прежних warnings, без reruns. Новые production/oracle/experiment pylint10/10,
package9.99 с прежними сообщениями; black/isort/hooks/snippets/links/figures проверены.
50source rows,9 independent physical/tagged cap refinements,10DES×4seeds.
Приоритеты диагонали Table2 соответствуют обратным labels; Table3 согласована.
Eq12/mean/BP→FP/tagged-index typos проверены независимыми арбитрами;
авторский software не аудирован. JSON двух запусков побайтно одинаковый:
SHA256 `4e5139154ee707f012cd1f4ef3a335a1a165dc5c3797fc4f5acac585ac821ae8`.
Source PDF/text, Claeys thesis TOC/model/assumptions и private cards/lemmas/
synthesis/plan/hypotheses/journal сохранены;95 snapshot SHA records/7sync tests.
Журнал: prefix59772bytes SHA256
`404522d9b15e4c9347e3443c3440ea7fd1c7b1cc7f739ce37641ec05fd5f7deb`
сохранён дословно. Приватная папка gitignored, не резервируется Git/push.
Изменения Л8 пока не закоммичены. Л7 остаётся waiting_source;
дальнейший исследовательский/калибровочный трек требует выбора автора.
