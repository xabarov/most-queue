# Availability-aware arrival history

[EPIC-065](epics/EPIC-065-availability-aware-arrival-history.md) устраняет
диагностированный в [EPIC-062](joint_marked_arrivals.md) артефакт: arrival-fit
donor-история там обрывается перед первым ещё не завершённым к cutoff job'ом
(«полностью разрешённый префикс»), хотя его `submit`/`need`/`context`/
`requested_time` уже известны. На SDSC .90 это давало `prefix_lag`=8.2 дня,
на Kalos .70 — 4.6 дня. Это не ограничение данных: обе трассы уже парсят более
богатую, чем completed-only, популяцию (`SwfLifecycleTrace.jobs`, status в
{1,5}; `AcmeTrace.terminal_jobs`, completed+cancelled+failed+timeout+
node_failed) — просто уровень эксперимента искусственно сужал её до
completed-only и до "непрерывного" префикса.

## Контракт

`most_queue.sim.utils.workload_trace.availability_prefix(jobs, cutoff)` —
новая, отдельная от `chronological_split` функция: возвращает каждую запись
с `submit < cutoff`, без условия `completed_at < cutoff` и без остановки на
первом незавершённом. Сама функция не читает `runtime`/`wait`/`status` —
только `submit` и `completed_at` (для диагностики `unresolved_at_cutoff`).
Это гарантирует, что в donor-пул для arrival-fit не просачивается ни одно
поле, требующее завершения job'а; service-fit (S|K) остаётся полностью
отдельным, completed-only, без изменений.

`examples/availability_aware_arrivals_experiment.py` переиспользует
`examples/joint_marked_arrivals_experiment.py` импортом (`prepare`,
`observed_marks`, `fit_arrivals`, `service_fit`, `forecasts_for`, `replay`,
`trace_for`, `uniform_streams`, `diagnostics`, `POLICIES`, `CONFIGS`) и
добавляет только: (1) `prepare()`, вызывающий старый `prepare()` для
contiguous-prefix контроля, затем строящий `avail_prefix`/`avail_recent_prefix`
через `availability_prefix` поверх `source.jobs` (SDSC) / `source.terminal_jobs`
(Kalos); (2) `workloads()` с четырьмя вариантами: `fixed_coarse` (контроль),
`recent_joint_iid_stale` (старый completed-prefix donor, для прямого
сравнения), `recent_availability_iid`/`expanding_availability_iid` (новые).
Те же `donor_u`/`service_u` uniform-потоки (`uniform_streams`, тот же seed,
cutoff, name), что и в EPIC-062, используются для ВСЕХ вариантов внутри
одного seed — различия в результатах изолированы на выбор donor-пула, не на
разные случайные потоки.

Четыре origin (SDSC .85/.90, Kalos .65/.70), split, held-out cohort, policies
и число replications — те же, что в EPIC-062, для прямой сопоставимости
prefix_lag и Q. Блок/shuffle/gap-independent оси EPIC-062 не повторяются
(отдельный, уже закрытый вопрос; этот эпик — только про stale vs
availability-aware donor pool).

## Ограничения, зафиксированные до replay

Failed-статус SWF (status=0) и Acme-записи `state` RUNNING/PENDING без
`start`/`end` остаются вне donor-пула — уже отфильтрованы существующими
парсерами (`parse_swf_lifecycle`/`parse_acme_kalos`); расширение ingestion —
резерв, не часть этого эпика. Service-fit и held-out cohort не меняются;
`runtime` отменённых/неуспешных записей никогда не используется как latent
completed S — ни здесь, ни в донор-marks (`ArrivalMark` несёт только
`gap`/`need`/`context`/`requested_time`).

[Результаты и ограничения](research/availability-aware-arrivals-results-2026-10.md),
[артефакты](../works/availability_aware_arrivals/README.md).
