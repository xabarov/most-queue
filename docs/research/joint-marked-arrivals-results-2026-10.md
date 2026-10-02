# Совместные приходы и ресурсные признаки: перенос на поздние периоды

Дата: 2026-10-02. [Методика](../joint_marked_arrivals.md),
[эпик](../epics/EPIC-062-joint-marked-arrivals.md),
[артефакты](../../works/joint_marked_arrivals/README.md).

**Итог:** joint/block генерация не дала устойчивого улучшения очередей против
fixed-arrival coarse. Сохранение связи интервала прихода с K и блочного порядка
само по себе не исправляет смену интенсивности и состава нагрузки. На позднем
Kalos генератор использует устаревший completed-prefix и сильно промахивается
по длительности потока и K-mix. Это диагностическое наблюдение, не доказанное
причинное объяснение. Ни один вариант не назначается новым baseline после test.

## Что сравнивается

Выполнено **1176 расписаний**: четыре origins, шесть generated variants,
восемь seeds, шесть прежних policies и 24 observed controls. Во всех сценариях
completed-only empty start; SDSC warmup=200/targets=1000, Kalos 100/300.
Нет carry-in, failed/cancelled окружения или caps. Observed — replay записанных
длительностей в этой модели, не исторические W/T. Поэтому числа ниже не
сопоставимы с lifecycle reference EPIC-061 как с одинаковой популяцией.

Общий service mechanism — recent coarse S|K; forecasts frozen expanding p90.
Генерируются Δ/K/context/request, но context/request не влияют на S или policy.
Job IDs синтетические, горизонты не подгоняются под observed. Источники и эти
даты изучались ранее: заданная temporal evaluation не является blind holdout.

## Ошибка задержек и парные контроли

Q — среднее абсолютных log-errors MC means по шести policies и mean/p99 T,
отдельно по origin; меньше лучше. Сначала mean восьми репликаций, потом log.

| Source / origin | Fixed coarse | Gap independent | Recent joint iid | Block20 | Shuffle20 | Expanding joint iid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDSC .85 | 0.230030 | 0.426189 | 0.396040 | 0.426653 | 0.399129 | 0.132855 |
| SDSC .90 | 0.092814 | 0.931875 | 0.765741 | 0.998084 | 0.919471 | 0.804339 |
| Kalos .65 | 1.231780 | 0.825125 | 0.526145 | 1.060112 | 1.060112 | 0.704014 |
| Kalos .70 | 1.615072 | 1.919705 | 1.728569 | 1.120797 | 1.183566 | 1.942671 |

Лучший точечный вариант меняется: expanding iid, fixed coarse, recent iid,
block20 соответственно. Это не процедура выбора и не свидетельство переносимости.
Парные seed-level log-loss contrasts оценивают **другой функционал** — mean L(a)−L(b).
Отрицательное лучше для левого. 95% t-CI условны на fit/cohorts, без поправки
на множественные сравнения; четыре origins не используются как независимая выборка.

| Origin | Joint iid − gap-independent | Block20 − shuffle20 | Recent iid − expanding iid |
| --- | --- | --- | --- |
| SDSC .85 | −0.0130 [−0.2596, 0.2336] | −0.0129 [−0.0812, 0.0553] | 0.0780 [−0.1526, 0.3086] |
| SDSC .90 | −0.2124 [−0.5064, 0.0815] | 0.0676 [−0.2252, 0.3604] | 0.0135 [−0.6602, 0.6872] |
| Kalos .65 | −0.1704 [−0.6390, 0.2982] | 0 [0, 0] | −0.0121 [−0.3644, 0.3402] |
| Kalos .70 | −0.1201 [−0.5485, 0.3082] | 0.0066 [−0.0090, 0.0223] | −0.1273 [−0.5621, 0.3076] |

Все невырожденные интервалы в этой таблице включают ноль. Точный Kalos .65
block/shuffle tie связан с нулевым W в обоих вариантах и одинаковым inventory S;
он не доказывает безразличие к порядку в другой нагрузке. Block/shuffle сохраняют
work и horizon, но меняют также warmup-order: это не эффект только target-order
из одинакового состояния. Gap-independent сохраняет K/context/request bundles,
но service перестраивается по перемещённому K; его work может отличаться.

На SDSC .90 все пять новых generators имеют положительные seed-level contrasts
против fixed coarse с нижними границами CI выше нуля. Например, recent joint iid:
**0.5809 [0.2653, 0.8965]**. На Kalos .65 его contrast, напротив,
**−0.5044 [−0.8409, −0.1679]**. Универсального преимущества нет.

Особенно важно не смешивать Q и средний seed-loss: на Kalos .70 block20 имеет
Q=1.120797 против fixed 1.615072, но seed-level block−fixed=+0.3376
[−0.2917, 0.9668]. На SDSC .85 expanding Q ниже fixed, однако seed-level
expanding−fixed=+0.0881 [−0.1170, 0.2932]. Усреднение метрик до нелинейного
loss может скрыть разнонаправленные ошибки репликаций.

## Что происходит с потоком

Ниже horizon targets в днях и K-group TV к observed. Модельные числа — среднее
восьми seeds, не один сгенерированный поток. Fixed_coarse сохраняет observed
horizon и нулевой TV, поскольку arrival/K зафиксированы.

| Origin | Observed horizon | Recent iid horizon | Block20 horizon | Expanding horizon | Recent iid K-TV | Block20 K-TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDSC .85 | 24.1227 | 21.6090 | 21.3486 | 16.0352 | 0.1221 | 0.1365 |
| SDSC .90 | 33.3133 | 24.0447 | 22.7669 | 16.3308 | 0.0636 | 0.0593 |
| Kalos .65 | 1.9328 | 0.8273 | 0.5485 | 1.4133 | 0.0163 | 0.0067 |
| Kalos .70 | 19.2367 | 0.6998 | 1.0495 | 1.9102 | 0.6292 | 0.6200 |

На SDSC .90 block20 восстанавливает lag-1 K до 0.2389 при observed 0.2584
(shuffle −0.0100), но Q остаётся хуже fixed. Сохранённая корреляция не равна
правильной интенсивности, классовой смеси или задержкам. Joint iid Δ↔K=0.2373,
gap-independent −0.0122, observed 0.1706: контроль действительно ослабляет
связь, но не обязательно улучшает её приближение к позднему периоду.

На Kalos .70 observed target counts по K=1/2–8/9–32/>=33 равны [107,75,31,87].
Joint iid K-TV=0.6292, joint K-group×context TV=0.6304; его zero-gap fraction
54.35% против observed 1.34%. Target resource work — 28.77 млн requested-GPU-s
против observed 267.48 млн; fixed coarse даёт 571.11 млн. Даже общий S|K fit
не означает одинаковую общую работу при разных K-mix.

На Kalos .65 observed targets все K=1, W=0; recent joint iid добавляет редкие
широкие jobs: K-TV всего 0.0163, но mean target work 19.82 млн GPU-s против
0.187 млн observed. Малый marginal TV сам по себе не контролирует ресурсно
взвешенные последствия. Понижение Q здесь не является достаточным основанием
для признания генератора реалистичным.

### Цена полностью разрешённого префикса

Arrival fit останавливается перед первым eligible completed с end>=cutoff,
не соединяя arrivals через пропуски. Известные более поздние S доступны
service fit, но не arrival prefix. Lag указан в днях; suffix counts — jobs.

| Origin | Prefix donors | Omitted suffix | Уже завершены, но исключены | Prefix lag, days |
| --- | ---: | ---: | ---: | ---: |
| SDSC .85 | 39628 | 54 | 42 | 3.0303 |
| SDSC .90 | 40901 | 209 | 191 | 8.2035 |
| Kalos .65 | 9666 | 0 | 0 | 0.0699 |
| Kalos .70 | 9666 | 625 | 624 | 4.6288 |

Один ещё не завершённый job в Kalos .70 останавливает prefix перед 624 уже
известными outcomes. Arrival donors те же, что в .65, хотя service history
и cohort обновились. Следовательно название recent не означает свежесть к cutoff.
Это ограничение выбранного контракта, зафиксированное до replay. После просмотра
результатов prefix/window/block не изменялись. Возможный будущий вариант —
arrival fit по действительно доступным submission-time marks всей нужной
популяции, с отдельным учётом цензурированного S; его нельзя имитировать
скрытым использованием будущего completed-label.

## Выбор дисциплины

На SDSC .85 все варианты выбирают FirstFit по mean T, как reference. По p99
все выбирают FirstFit вместо MSF, regret **0.5203%**. На .90 reference MSF
лучший по обеим целям; все модели выбирают FirstFit по mean T, regret **0.6234%**.
По p99 fixed coarse и expanding iid правильно выбирают MSF; остальные четыре
варианта выбирают FirstFit, regret **6.6813%**. Улучшение отдельного Q не
гарантирует выбора хвостовой policy.

Kalos reference имеет W=0 и шесть ties по mean/p99 на обоих origins.
Модельные policies могут различаться, но zero regret здесь не валидирует
дисциплину. Контекст, квоты, топология и меняющаяся доступность ресурсов не
восстановлены; номинальная capacity не подбиралась к историческому ожиданию.

## Воспроизводимость и продолжение

Основной запуск на восьми worker и повтор на шести дали **шесть побайтово
одинаковых JSON**, включая protocol и manifest. Проверены 20 implementation
hashes и пять output hashes. Независимый аудитор восстановил 196 workload
лент, проверил 32 block/shuffle inventory pairs, пересчитал все 24 observed
schedules и проверил 1176 строк, 1008 model intervals, 24 Q, 32 paired contrasts
и 48 policy decisions. Он использует те же dispatchers, не второй scheduler.

Добавлено 59 offline-тестов: exact adjacent gaps и batches, wrap/truncation,
строгая валидация, matched inventory/work/horizon, временной prefix и future-S
guards, совпадение старого fixed replay, формула Линдли для capacity=1,
нулевой measured span, независимые PRNG streams, одинаковые serial/parallel
JSON и сохранение protocol до replay. Полный pytest — **1740 passed**, быстрый —
**1731 passed**, оба с 16 warnings. Использована CI-политика одного повтора
AssertionError; фактических повторов не потребовалось. Новые модуль/runner:
pylint **10/10**, библиотека **9.98/10**, без новых замечаний.
Black/isort/pre-commit проходят. Независимый reader review уточнил до replay
контракт prefix и перестановок; финальная сверка таблиц/ограничений с JSON
пройдена без существенных замечаний.

Следующий этап основного roadmap — **аудит нового источника с наблюдаемыми
quotas/placement/доступностью ресурсов и submission-time marks**. Сначала
проверить timestamps, лицензии и полноту; затем отдельным протоколом оценить
различающие policies reference и цензурирование. Если этих полей нет, честный
результат — карта пробелов, а не восстановление квот из test W. Параллельное
исследовательское ответвление — availability-aware arrival history без
completed-prefix staleness, с отдельным ранним выбором окна и новыми периодами.
Fixed-arrival coarse остаётся обязательным контролем; подбор блоков, capacity
или нового победителя по уже рассмотренным четырём origins не выполняется.
