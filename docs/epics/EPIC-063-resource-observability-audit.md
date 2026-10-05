# EPIC-063: Аудит наблюдаемости ресурсных ограничений

- **Статус:** done
- **Создан:** 2026-10-02
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)

## Цель и границы

Выбрать следующий источник для ресурсно-ограниченного replay по доступным
наблюдениям, а не по возможности получить красивый ranking policies.
Сравнить Helios, Philly, Alibaba PAI 2020, Alibaba GPU 2023/2026 и Google 2019.
Закрепить версии, проверить условия использования, timestamps, submission-time
marks, фактическое размещение, квоты и доступность. Отдельно отмечать schema-only
свидетельства и проверку сырых строк. Отсутствующее поле не восстанавливать из W.

Основной количественный аудит — полный опубликованный Helios: четыре job logs
и четыре дневных VC-конфигурации. Выбор обусловлен совместным наличием VC,
job timestamps и GPU counts по VC/day, не результатами replay. Это аудит данных,
не реализация нового scheduler или production reconstruction. Нет платных
BigQuery-запросов, отправки форм, контактов с владельцами или новых учётных записей.
Данные остаются в ignored cache; в git только агрегаты, hashes и код проверки.

## Контракт до количественного аудита

1. Закреплённый архив, SHA-256 и hashes восьми извлечённых payloads. Схемы должны
   совпадать; никакого extractall или запуска кода из чужого архива.
2. Считать все строки и все исходные статусы, CPU-only отдельно. Проверить
   unique job IDs внутри cluster, неизвестные VC, пропуски/порядок timestamp,
   отрицательные/нулевые интервалы и равенства duration=end−start,
   queue=start−submit. Не превращать invalid/censored records в S=0.
3. Проверить даты VC-конфигураций, пропуски дней, изменение total и отдельных VC,
   сумму VC против total. Дневная конфигурация не доказывает внутридневной hard cap.
   Не назначать неизвестную timezone UTC и не переносить future daily row назад.
4. Сопоставить запрос GPU с VC-count на ту же календарную дату submit/start,
   отдельно считая отсутствующие даты/VC и нулевые counts. Это проверка совместимости
   при условной трактовке дневной строки, не доказательство нарушения квот.
5. Для валидных закрытых GPU execution intervals всех terminal statuses построить
   requested occupancy и work. Сопоставление с daily VC-count допустимо только как
   явно обозначенная диагностическая гипотеза постоянного значения в течение дня.
   Нет claims физической utilization; неполный execution log не равен полному ресурсу.
6. До рекомендации сформулировать go/no-go отдельно для bounded scenario и точного
   воспроизведения. Документированные поля не гарантируют online availability,
   user/VC label не равен численной квоте, utilization не равно capacity,
   node_num не равно размещению, retries не равно полезному S.

## Работы и приёмка

- [x] Матрица шести кандидатов с pinned primary sources и условиями использования.
- [x] Воспроизводимый raw-аудитор Helios, строгие offline fixtures и проверки ошибок.
- [x] Полный аудит четырёх clusters, output hashes, побайтовый повтор и независимая сверка.
- [x] Полный/быстрый pytest и стандартные проверки качества для нового кода.
- [x] Отчёт с gap map, решением по источнику и контрактом следующего эпика.
- [x] README/models/roadmaps и независимый reader review.

## Результаты

Полный raw-аудит всех четырёх опубликованных кластеров Helios (3,362,981 строк,
SHA-256 архива `3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac`)
с независимой проверкой (`works/resource_observability/verify.py`, pandas,
без импорта основного аудитора). Матрица шести источников (Helios, Philly,
Alibaba PAI 2020, Alibaba GPU 2023/2026, Google 2019) с pinned-версиями и
условиями использования — `docs/research/resource-observability-results-2026-10.md`.

Решение: Helios daily VC GPU-counts годятся только для явно обозначенного
**bounded daily-capacity сценария**, не для production replay. 17 459 GPU-строк
стартуют при daily VC-count=0 и 17 525 запрашивают больше указанного — дневная
конфигурация не является доказанным intraday hard cap. При этом aggregate-pool
excess=0 на всех четырёх кластерах по опубликованным датам.

34 offline-теста (`tests/units/test_resource_observability_audit.py`), чистые
black/isort, pylint без новых замечаний, полный `pytest tests/ -m "not slow" -n auto`
(1765 passed). README/models.md/epics-реестр обновлены.

Контракт следующего эпика зафиксирован в [EPIC-064](EPIC-064-msj-capacity-calendar.md):
opt-in capacity calendar поверх MSJ на этом же Helios-аудите, с grandfathering
и явными infeasible-результатами вместо их отбрасывания.
