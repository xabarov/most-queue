# EPIC-094: точное сокращение стадийного состояния TTFT

- **Статус:** done (exact finite-model research pilot; no publication claim)
- **Создан:** 2026-10-11
- **Предшественник:** [ГЛ3](EPIC-093-stage-aware-ttft.md).
- **Выбор автора:** «выполни следующий этап» после предложения сокращения состояния.

## Протокол до эксперимента

Сохраняются политика, joint law и rewards ГЛ3. При iid метках и admission,
зависящем только от числа ожидающих, типы FIFO waiting jobs можно раскрывать
при допуске. Состояние хранит active attained tokens и waiting count.
Отдельная абляция объединяет перестановки декодеров, сохраняя FIFO prefills.
Это deferred decisions и симметрия, не новый общий метод aggregation.
Интеграция waiting marks не strong lumpability для произвольного начального
слова: требуется пустой старт или совместимое iid распределение очереди.

Primary checks: равенство W, V, TTFT, accepted rate, loss и utilization
полной finite-loss модели с absolute/relative tolerance 1e-10; независимый
token-loss oracle; B1 PK и изолированный предел. Контрпримеры для сортировки
prefills и strong lumpability должны сохраняться в тестах.

Frozen comparison: все48 случаев прежней ГЛ3-сетки, K6; полная модель,
waiting-count only и waiting-count плюс decoder symmetry. Дополнительная
сетка: три типа [(1,1,.3),(2,2,.4),(4,3,.3)], B2/3, M=B/2B,
K2/4/6/8, load=.5, costs=(.1,.02,.003). Отказы полного калькулятора по
его прежним budgets публикуются, не считаются доказательством равенства.
Sensitivity K2/4/6/8/12 для двух прежних законов, B2, M2/4, load=.8.

Timing: прежние четыре случая B2, K6, costs=(.1,.02,.003), load=.5;
три повтора и DES12000jobs с seeds93000..93002. Время включает построение
и решение, excludes input generation для всех методов. Wall и CPU отдельно.
Целевой engineering criterion на mixed laws: >=4x меньше states и >=2x
быстрее full; отношение к DES — secondary diagnostic. Не подгонять после
прогона. Exact structural error0 только для finite model при условиях выше;
численный residual не interval certificate, K sensitivity не infinite bound.

## Задачи

- [x] Математический вывод, границы применимости и первоисточники.
- [x] Калькулятор с явными complexity guards и независимые тесты.
- [x] Frozen comparison, абляция, paired timing и sensitivity.
- [x] EN/RU документация, диссертационный синтез, журнал и архив.
- [x] Fast/full tests, formatting/lint, проверка ссылок.

## Результаты

[Отчёт](../research/stage-aware-ttft-reduction-validation-2026.md):48 прежних
и8 доступных expanded cases совпали по шести rewards до4.36e-14. На8 других
expanded cases full max_states refusal; reduced решает16/16, до2339states.
Отказы не засчитываются как numerical equality validation. Mixed K6/M2/M4:
1907→141 /1270→97states (13.525/13.093x); paired wall speedup27.185/18.224x,
CPU27.181/18.214x. Engineering thresholds пройдены на обоих mixed cases.
Symmetry немного медленнее count-only на этой малой timing-сетке; отрицательный
вывод опубликован. K12 refinement доступен, но infinite error bound не доказан.

32 новых tests;54 stage-aware targeted; fast3270/full3288passed,
16existing warnings; black/isort/gitdiffcheck чистые, new-module pylint10/10.
Диссертационные source cards/гипотезы/план/концепция/append-only journal
обновлены, public research/roadmap snapshots синхронизированы с SHA256.
Основной метод классический, general empirical scalability не установлена.

Tail/SLO, correlated marks, real-support phase compression и hardware
calibration не входят в этот этап.

Следующее возможное самостоятельное исследование — infinite queue-count
blocks и stability/error certificate; альтернативно approximate real-support
phase compression. Оба требуют выбора автора, не начаты этим эпиком.
