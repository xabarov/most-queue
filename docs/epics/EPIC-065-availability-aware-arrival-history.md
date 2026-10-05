# EPIC-065: Availability-aware arrival history без completed-prefix staleness

- **Статус:** done
- **Создан:** 2026-10-05
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)
- **Основание:** [EPIC-062](EPIC-062-joint-marked-arrivals.md) (диагноз), отчёт
  [joint-marked-arrivals-results-2026-10.md](../research/joint-marked-arrivals-results-2026-10.md)
  (раздел «Цена полностью разрешённого префикса»)

## Цель и диагноз

EPIC-062 показал, что текущая arrival-mark donor-история (`prepare()` в
`examples/joint_marked_arrivals_experiment.py`) требует **непрерывного
completed-at-cutoff префикса**: donor-последовательность обрывается на первом
ещё не завершённом к cutoff задании, даже если submit этого задания и всех
последующих уже известен. На SDSC .90 это даёт `prefix_lag`=8.2 дня, на Kalos
.70 — 4.6 дня; один незавершённый job останавливает префикс перед 624 уже
известными outcomes на Kalos .70.

Это артефакт эксперимента, не ограничение данных: `MarkedArrivalBootstrap.fit`
использует только `submit`/`need`/`context`/`requested_time` — ни одно из этих
полей не требует знания `runtime`/`completed_at`. Обе трассы уже парсят более
богатую, чем completed-only, популяцию: `SwfLifecycleTrace.jobs` (status в
{1,5}, completed+cancelled, каждый с `requested_time`) и `AcmeTrace.terminal_jobs`
(completed+cancelled+failed+timeout+node_failed). Новая проверка данных или
новый адаптер ingestion **не требуются** — только новая construction donor-
истории на уровне эксперимента, использующая `submit < cutoff` без остановки
на первом незавершённом задании.

## Контракт до replay

- Arrival-fit donor-пул = все записи из существующей богатой популяции
  (`SwfLifecycleTrace.jobs` / `AcmeTrace.terminal_jobs`) с `submit < cutoff`,
  без "contiguous completed prefix" усечения. Service-fit (S|K) остаётся
  completed-only и без изменений — S **нельзя** узнать раньше завершения.
  `runtime` отменённых/неуспешных записей не используется как latent
  completed S нигде в arrival-fit (их marks — только submit/need/context).
- Failed-статус (status=0 SWF, "other_status") и genuine unresolved-at-export
  Acme-записи (`state` RUNNING/PENDING без `start`/`end`) остаются вне scope —
  уже отфильтрованы существующими парсерами, расширение ingestion не
  планируется в этом эпике (резерв).
- Те же четыре origin (SDSC .85/.90, Kalos .65/.70), тот же split/held-out
  cohort, то же число replications/seeds, что и EPIC-062 — используется для
  прямого сравнения prefix_lag и Q/regret, не для подбора нового окна или
  capacity по итогу.
- Fixed-arrival coarse остаётся обязательным контролем. Один новый вариант
  (availability-aware recent iid) сравнивается против старого completed-prefix
  варианта (recent_joint_iid) и fixed_coarse; старый вариант не удаляется и не
  переопределяется задним числом, это новая, независимая ветка сравнения.
- До просмотра результатов зафиксировать: ожидаемое снижение prefix_lag
  (должно быть на порядки меньше completed-prefix lag, больше не определяется
  дотягиванием до первого незавершённого задания), и что это diagnostic/
  generative improvement, не причинное объяснение ошибок очереди.

## Работы и приёмка

- [x] Новая функция построения availability-aware donor-пула (submit<cutoff,
  без completed-prefix усечения) поверх существующих rich-популяций.
- [x] Юнит-тесты: synthetic fixture показывает prefix_lag -> ~0 против
  completed-prefix эквивалента; донор-пул включает cancelled/failed записи
  корректно (marks верны, S не просачивается).
- [x] Новый эксперимент (отдельный модуль, переиспользующий
  `joint_marked_arrivals_experiment`/`feature_service_experiment` импортом):
  recent_availability_iid против recent_joint_iid (old) и fixed_coarse,
  те же origins/seeds/policies.
- [x] Полный/быстрый pytest, pylint/black/isort, побайтовый повтор и
  независимая сверка.
- [x] Отчёт: prefix_lag до/после, влияние на Q/regret, явные ограничения
  (failed-статус и genuine right-censoring вне scope).
- [x] README/models/roadmaps обновлены; эпик переведён в done.

## Критерии готовности (DoD эпика)

Общий [DoD](../DOD.md) + контракт выше.

## Результаты

`most_queue/sim/utils/workload_trace.py` получил `availability_prefix(jobs,
cutoff)` — читает только `submit`/`completed_at`, возвращает все записи с
`submit < cutoff` без остановки на первом незавершённом; 4 юнит-теста
покрывают включение unresolved-записей, снижение lag при длинном stalling
job'е и отказ на коротких/неупорядоченных/некорректных входах.

`examples/availability_aware_arrivals_experiment.py` (новый, переиспользует
`joint_marked_arrivals_experiment` импортом без изменения последнего)
добавляет `recent_availability_iid`/`expanding_availability_iid` к контролям
`fixed_coarse`/`recent_joint_iid_stale`, на тех же четырёх origin
(SDSC .85/.90, Kalos .65/.70), split, cohort, policies, replications, что в
EPIC-062. Донор-пул строится из `source.jobs` (SDSC, status в {1,5}) /
`source.terminal_jobs` (Kalos) — существующих, уже распарсенных богатых
популяций; новый ingestion не потребовался.

792 расписания (4 origin × (1 observed + 4×8 replications) × 6 policies),
побайтовый повтор (8 vs 6 workers) подтверждён для всех шести JSON.
Независимый аудит (`works/availability_aware_arrivals/audit.py`, повторно
использует независимый парсер EPIC-059, не импортирует `availability_prefix`
или `prepare`) подтвердил prefix_jobs/unresolved_at_cutoff/prefix_lag на всех
четырёх origin. 13 новых юнит-тестов эксперимента + 4 для
`availability_prefix` (пересекается с выше) = 16 новых тестов; полный
`pytest tests/ -m "not slow" -n auto` — 1804 passed (было 1788). pylint
`most_queue` без новых замечаний (9.98/10, не изменился), pylint нового кода
— 10/10, black/isort чисты.

**Находка:** prefix_lag падает на порядки (SDSC .90: 8.2 дня → 12.7 минут;
Kalos .70: 4.6 дня → 4 часа) и donor-пул примерно удваивается на каждом
origin — но это **не равномерное улучшение Q**: ошибка очереди заметно
ухудшается на SDSC (контраст положителен и значим), заметно улучшается на
Kalos .70 (контраст отрицателен и значим), и не меняется значимо на Kalos .65.
Выбор дисциплины по mean T не меняется ни на одном origin. Правдоподобный
(не доказанный) механизм — донор-пул сдвигает средний gap относительно
observed в РАЗНЫЕ стороны на разных источниках (ближе на Kalos .70, дальше на
SDSC), совпадая по направлению с эффектом на Q; need-group TV при этом
улучшается (ближе к observed) на обоих источниках, то есть расхождение
объясняется темпом приходов, не миксом K.

Failed-статус SWF (status=0) и genuinely unresolved-at-export Acme-записи
(`state` RUNNING/PENDING без start/end) остаются вне donor-пула — резерв,
не реализовано здесь. [Отчёт](../research/availability-aware-arrivals-results-2026-10.md).
