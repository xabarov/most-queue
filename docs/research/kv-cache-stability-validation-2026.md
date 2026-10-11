# Устойчивость KV-ограниченного обслуживания: метод и независимые проверки

Л4 / [EPIC-087](../epics/EPIC-087-kv-cache-stability.md), 2026-10-11.
Реализованы условные границы устойчивости и независимая слотовая имитация.
Верхняя ресурсная граница не объявляется точной пропускной способностью;
максимальная упаковка с бесплатным swap не подменяется hard-resident FCFS.

## Источники и прицельная проверка

[Chengyi Nie, Nian Si, Zijie Zhou, arXiv:2605.04595v1](https://arxiv.org/html/2605.04595v1),
6 May 2026; отметка Accepted in ICML 2026 подтверждена на
[странице arXiv](https://arxiv.org/abs/2605.04595). Прочитан полный текст:
§1–6, Impact Statement, библиография, приложения A–C. Реализация — §3–4
и packing-доказательство приложения A; замеры §5 и эмпирическая замена
времени слота в приложении C не превращаются в новые теоретические гарантии.

До кода проверены локальные реестры источников/лемм, MSJ packing и resource
envelope. Рыжиков 2018, `docs/books/RBOOQLan18/CH4w.TEX`, §4.1 прочитан:
устойчивость открытой цепи требует нормируемости бесконечной меры, не
автоматически следует из конечной нормировки. Одномерная occupancy-цепь
Л3 не хранит индивидуальные стадии и растущую память, поэтому не используется
как точный арбитр Л4.

Специализированный источник: Dai и Harrison, *Processing Networks: Fluid
Models and Stability* (2020). Прочитано полное опубликованное
[оглавление Cambridge](https://www.cambridge.org/core/books/abs/processing-networks/contents/E04EDCB2595AAF0863B08B693C367307):
главы 1–12 и приложения A–F, в частности stability achievable, fluid limits,
positive recurrence, policy-specific fluid equations и Lyapunov functions.
Полный текст книги **не** прочитан: PDF spnbook.org оказался недоступен
(SSL hostname mismatch в локальном запросе, timeout веб-доступа).
Ссылка на thesis proposal Yige Hong из поиска перенаправляет на авторскую
главную страницу; оглавление proposal не считается прочитанным.

Прицельный поиск по механизмам: `stochastic bandwidth packing stability`,
`time-varying resource requirements bin packing`, `cumulative resource
demand memory admission`, `AoI resource-constrained updating`, `LLM growing
KV stability flow control`. Четыре направления:

| Направление | Проверенный первичный источник | Статус чтения и вывод |
|---|---|---|
| Классическая ТМО | [Gamarnik 2004, Stochastic bandwidth packing process](https://research.ibm.com/publications/stochastic-bandwidth-packing-process-stability-conditions-via-lyapunov-function-technique) | Авторская abstract-страница: largest-first, разные размеры, unit-time stages, Lyapunov stability; упаковка и drift не новые методы |
| Изменяемые требования | [Hong, Xie, Wang, arXiv:2209.04123](https://arxiv.org/abs/2209.04123) | Abstract: Markovian time-varying sizes и число занятых серверов; сам механизм изменения footprint уже исследован |
| Real-time | [Eisenklam et al., ECRTS 2026, MPORA](https://doi.org/10.4230/LIPIcs.ECRTS.2026.17) | Abstract, introduction и §3.1: динамическое перераспределение cache/bandwidth по фазам и deadline constraints; не та же очередь и не доказательство KV-границы |
| AoI | [Yates et al., arXiv:2007.08564](https://arxiv.org/abs/2007.08564) | Abstract и тематический поиск resource-constrained updating; временная ценность/ограничения ресурса — соседний вопрос, не отсутствие packing-результатов во всей AoI-литературе |
| ML systems | [Jaillet et al., arXiv:2502.07115v5](https://arxiv.org/abs/2502.07115v5) | Abstract: hindsight IP и online competitive scheduling при KV; не выдается за изученное доказательство recurrence |
| ML stability | [Dong, Cao, arXiv:2604.11001v1](https://arxiv.org/html/2604.11001v1) | Abstract и начало introduction: flow-controlled admission, known/unknown output lengths, необходимые/достаточные условия; найден прямой сосед, полный текст не прочитан |
| ML throughput | [Li, Dai, Peng, arXiv:2504.07347](https://arxiv.org/abs/2504.07347) | Abstract поискового результата: work-conserving throughput; полное доказательство не проверено |

Это ограниченная проверка литературы, не систематический exhaustive review.
Новизна не заявляется. Авторское «first framework» не повторяется как
установленный факт: рядом есть классическая packing-теория и прямые LLM
статьи о stability, пусть с отличающимися контрактами.

## Математический контракт

Один worker, постоянный слот b, бесконечная очередь, iid число приходов
со средним lambda*b. Joint finite PMF P(S=s,O=o), допускающий корреляцию;
s,o — положительные целые, s кратно h. В каждом слоте выбранная заявка
продвигается на одну стадию. Следы prefill: h,2h,...,s; decode: s+1,...,s+o.
Последняя стадия оплачивается полностью, затем память освобождается.

Формула площади памяти §4:

```text
g(s,o) = h*sum(j, j=1..s/h) + sum(s+j, j=1..o)
       = s*(s/h+1)/2 + s*o + o*(o+1)/2
E[g] = E[S²]/(2h) + E[S]/2 + E[SO] + E[O²]/2 + E[O]/2
mu_upper = M/(b*E[g])
delta = max(s+o)/M over positive-probability support
mu_lower = (M-max(s+o))/(b*E[g])
```

Таким образом, замена совместного закона средними теряет вторые моменты
и covariance. Независимый пример с одинаковыми маргиналами {2,6}/2:
при h=2 positive correlation {(2,2),(6,6)} даёт E[g]=39,
negative {(2,6),(6,2)} — 31. Разница ровно 8 — изменение E[SO].
Пик одновременно меняется с 12 до 8, влияя на достаточный запас.

Литературный текст говорит о первом токене после prefill, но формула g
содержит все o charged decode stages. Библиотека явно следует этой формуле;
в prefill дополнительный output не добавляется. Это не автоматически
конвенция Л2, где первая генерация входит в prefill.

### Что именно доказано

Theorem 4.1: lambda > mu_upper означает перегрузку при любой дисциплине
этой модели. За T слотов поступает работа со средним T*lambda*b*E[g],
а выполнить можно не более T*M. При конечном support рост остаточной
работы также требует роста числа незавершённых заявок.

Theorem 4.2: lambda < mu_lower достаточно при максимальной упаковке
следующих стадий и бесплатной выгрузке. Не выбранная частичная заявка
не держит GPU-память. При backlog > M найдётся достаточно кандидатов,
а каждый следующий footprint <= peak. Максимальная упаковка не может
оставить больше peak свободной памяти; выполненная старая работа U >= M−peak.

Правильно определённый V — сумма **оставшихся** footprint всех заявок.
Тогда в точности `V_next = V + incoming_area − U`, и вне конечного
множества при bounded support:

```text
E[V_next−V | state] <= lambda*b*E[g] − (M−peak) < 0.
```

При стандартных предпосылках Markov/recurrence это Foster–Lyapunov аргумент.
Приходы в симуляторе — в начале слота, в доказательстве — между границами;
переиндексация не меняет средний вход работы и drift верхней границы.
Packing-аргумент в явной free-swap модели полезен при любом peak < M;
«delta << 1» мотивирует близость границ и GPU-аппроксимацию, а не выдуманный
числовой cutoff. При delta=1 нижний интервал пуст. Все равенства и зазор
между границами остаются inconclusive.

### Почему резервирование пика — другая модель

Контрпример: s=4,o=2,h=2,M=40,b=1. Полная area=17;
mu_upper=40/17, mu_lower=34/17=2. Резервирующая пик дисциплина
может держать лишь floor(40/6)=6 заявок; каждая занимает место 4 слота.
Её насыщенная пропускная способность 6/4=1.5. При lambda=1.75 она
перегружена, несмотря на lambda < 2. Это независимый счёт слотов/резервов,
не симуляция той же Lyapunov-формулы. Нижняя теорема к ней неприменима.
В API `zero_cost_swap=False` отключает sufficient certificate.

## Ошибки индексации в приложении A

Печатные упрощённые выражения остаточного v_i включают уже обслуженную
стадию. При s=4,o=2,h=2, завершённом prefill chunk c=1 остаются
следы 4,5,6 — **15** token-slots. Буквальная simplified prefill formula
приложения даёт **17**. При завершённом decode d=1 остаётся след 6 —
**6**, а simplified decode expression даёт **11**.

Корректные оставшиеся суммы:

```text
prefill: h*sum(j, j=c+1..s/h) + sum(s+j, j=1..o)
decode:  sum(s+j, j=d+1..o)
```

Независимый арбитр — прямое перечисление стадий, закреплённое тестом.
Калькулятор использует корректный initial g; симулятор — перечисленные
следы и вычитание фактически выполненной работы. Ошибочные residual
expressions не копируются. Это уточнение печатной алгебры, **не** утверждение
о проверке или ошибке авторского программного кода. Packing/drift аргумент
восстанавливается из определения V без этих упрощённых выражений.

## Независимые арбитры и эксперимент

- Прямые суммы footprints на четырёх масштабах, включая h=512,s=1024,o=256.
- Ручная двухзаявочная трасса M=4,s=2,o=2,h=2: U=[4,3,4,3,4,0],
  две CPU request-slots, общая работа 18; проверяется release и growth.
- Все 3^6=729 arrival-конфигураций трёх типов на трёх слотах: точное
  сохранение работы, предел памяти и large-backlog packing lower bound.
- Независимый configuration LP для s=o=h=b=1. Все пары (a,b) с
  a+2b<=M перечислены, stage flow равен throughput, смеси конфигураций
  суммируются в 1. При M=2..8 LP и явные насыщенные циклы ongoing-first
  дают M/3, внутри границ. Общая LP не объявляется точной для всех политик.
- Poisson: шесть репликаций двух stable-настроек; throughput сверяется
  с arrivals с SE и absolute floor из `tests/default_params.yaml`.
  Overload-прогон проверяет рост остаточной работы; slow — 100000 слотов
  correlated-length flow. Ни один конечный прогон не доказывает recurrence.

[Скрипт](../../examples/kv_cache_stability_experiment.py) и
[JSON](kv-cache-stability-results-2026.json): 36 настроек = три joint PMF ×
M={40,200} × h={1,2} × три интенсивности. На настройку четыре seeded
репликации по 2000 слотов, warmup=400. Интенсивности: .65*lower, midpoint
(lower+upper)/2 и 1.2*upper. Время слота b=1 искусственное.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python examples/kv_cache_stability_experiment.py
```

Например, M=40,h=2, below_lower:

| Joint law | E[g] | lower | upper | lambda | throughput mean ± SE |
|---|---:|---:|---:|---:|---:|
| fixed (4,2) | 17 | 2.0000 | 2.3529 | 1.3000 | 1.2970 ± .0079 |
| positive correlation | 39 | .7179 | 1.0256 | .4667 | .4672 ± .0032 |
| negative correlation | 31 | 1.0323 | 1.2903 | .6710 | .6588 ± .0192 |

Across 12 below-lower settings, throughput/arrival ratio is .9327–1.0179,
finite-horizon outstanding work 0–890 across individual replications.
Gap settings show approximately matching flow on this horizon but are
still **inconclusive**. Above upper, throughput/arrival .8133–.8448 and
outstanding work 11453–88289; all 144 runs obey exact work/count conservation.
These are finite-horizon observations, not error percentages for real GPUs.

## Почему не воспроизведены A100-таблицы

Статья §5 использует M=131000,h=512 и workload-specific медианы batch times.
Uniform integer input lengths 10..1600 и другие диапазоны не кратны h,
хотя формальный контракт §3 это предполагает. Правило последнего неполного
чанка меняет g; в библиотеке оно не придумывается скрыто. Есть и неоднозначность
между independent P/D lengths и фразой о принудительном matching вывода входу.
Без авторских traces/кода нельзя независимо восстановить их preprocessing.

В приложении C median заменяется trimmed mean при heavy tails. Это
эмпирический predictor, не перенос theorem на произвольную случайную
длительность: отбрасывание медленных слотов не уменьшает реально потраченное
время. В библиотеке b — заданная константа, а не «сертифицированная» медиана.
Tensor parallelism, раздельные P/D workers и pipeline также не объявляются
линейной экстраполяцией восьми single-GPU replicas.

## Реализация, границы и дальнейшая работа

`KvCacheStabilityCalc` / `KvCacheStabilitySim`, структуры результатов,
[EN модель](../models/kv-cache-stability.md) / [RU модель](../models/kv-cache-stability.ru.md).
Калькулятор возвращает None для w/v/p/pi, а simulator — flow/work на горизонте.
Отступление от общего jobs-to-completion интерфейса намеренное: принудительный
drain скрыл бы overload и дал смещённую censored latency статистику.

Replica planning использует **разные** ceil(lambda/upper) и строгий
floor(lambda/lower)+1, одинаковые мощности и iid per-worker потоки. Ни lower
bound на число GPU, ни наличие стационарности не является TTFT/SLO-гарантией.

Далее Л5: отдельно изучить service-induced congestion при платном
вытеснении/росте памяти. Flow-controlled scheduling Dong/Cao — кандидат
на будущий полный разбор, не расширение уже выполненного Л4. Полный
совместный workload из реальных traces требует отдельного train/test
протокола, а случайные slot times — нового доказательства/модели.

## Финальная проверка

106 целевых тестов, включая slow, прошли (3.42 s). Полная регрессия:
**2992 passed** (442.79 s); отдельная fast: **2983 passed** (340.89 s).
Оба прогона — 16 прежних warnings, без reruns. Команды:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest tests/ -q -n auto \
  --reruns 1 --only-rerun AssertionError
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest tests/ -q -m 'not slow' -n auto \
  --reruns 1 --only-rerun AssertionError
```

Black/isort чисты; новые модули и experiment pylint10/10, пакет9.98/10
с прежними замечаниями. Финально 468 локальных ссылок проверены без ошибок; EN/RU
схемы осмотрены и четыре примера выполнены. Все 36 аналитических строк JSON
совпали с повторным расчётом; 144 count/work баланса точны. Дополнительно
20 DES-репликаций полностью воспроизведены по seeds. Эпик/roadmap,
локальная карточка и память синхронизированы; Л4 закрыт, Л5 не начат.
