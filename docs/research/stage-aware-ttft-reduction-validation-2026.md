# Точное сокращение стадийного состояния среднего TTFT

2026-10-11. Продолжение [ГЛ3](stage-aware-ttft-validation-2026.md),
[протокол EPIC-094](../epics/EPIC-094-stage-aware-ttft-reduction.md).
При iid joint marks FIFO waiting word можно интегрировать точно, а
перестановки декодеров объединить без изменения политики. Сокращается
конечная loss-модель; её отличие от бесконечной очереди остаётся.

## Результаты фиксированного сравнения

Все48 прежних случаев совпали. На расширенной сетке с тремя типами,
B2/3, M=B/2B, K2/4/6/8 полная модель решила8/16 случаев; во всех восьми
равенство также подтверждено. Остальные8 full refusals вызваны прежним
max_states20000. Reduced решил16/16, maximum2339states; на отказавших
full cases численное равенство не считается проверенным. Maximum absolute
difference по W/V/TTFT/accepted rate/loss/utilization —4.3521e-14,
maximum reduced stationary residual на64 cases —1.3878e-16.
В B1 сравнивается та же классическая PK shortcut, не самостоятельная редукция.

Прежние cases B2,K6,load.5,costs(.1,.02,.003):

| Law | M | Full states | Waiting count only | Count + symmetry |
|---|---:|---:|---:|---:|
| (2,2) | 2 | 50 | 50 | 50 |
| (2,2) | 4 | 42 | 42 | 42 |
| (1,1,.6),(3,2,.4) | 2 | 1907 | 141 | 141 |
| (1,1,.6),(3,2,.4) | 4 | 1270 | 104 | 97 |

На expanded B3/M6/K8 waiting-only1943states → combined1660;
symmetry имеет дополнительный выигрыш, но waiting integration доминирует.
Ни одна метка/support/age bin не подгоняется по результатам.

Refinement четырёх прежних laws/M при load.8: K12 blocking<=3.3301e-6.
Изменение TTFT K6→K12 относительно K12: homogeneous M2/M4=0.7714%/0.1303%,
mixed M2/M4=0.4781%/0.2090%. Это finite sensitivity, не infinite error bound.
У исходного K6 относительно infinite DES остаётся прежняя ошибка ГЛ3:
exact reduction не улучшает accuracy за счёт самой замены состояния.

32 новых tests включают independent product lifting по каждой строке,
finite-loss replay с отдельным seed, полную модель, B1/zero-load, guards,
reconfiguration, decoder permutations и два контрпримера. Совместно с
исходными stage-aware tests —54. Monte Carlo tolerances в
`tests/default_params.yaml`: mean relative3.5%, waiting absolute.007,
loss absolute.012 для30000arrivals/warmup3000, не SLO-tail tolerance.

## Время и вычислительная пригодность

После завершения suites выполнены3 paired repeats, seeds93000..93002,
DES12000jobs,20%warmup; generation/setup исключены для всех методов, kernel
build и sparse solve включены. Analytical execution order ротируется.
Python3.12.3, NumPy2.2.5, SciPy1.13.0. Clock ratios — медианы отношений
внутри пар, **не** отношения отдельных медиан времени. Это diagnostic на
данной машине, не universal runtime bound и не hardware serving benchmark.

| Law | M | Full/reduced states | Paired wall full/reduced | Paired wall reduced/DES |
|---|---:|---:|---:|---:|
| (2,2) | 2 | 50/50 | 2.045 | 0.03723 |
| (2,2) | 4 | 42/42 | 1.868 | 0.03597 |
| Mixed | 2 | 1907/141 | 27.185 | 0.06911 |
| Mixed | 4 | 1270/97 | 18.224 | 0.09488 |

Для mixed laws state factors13.525/13.093, CPU speed factors27.181/18.214.
Prespecified engineering targets >=4x fewer states и >=2x faster full
пройдены на обоих mixed cases. Homogeneous cases не сокращают state count;
их speedup — преимущественно construction caches/vectorization.
Reduced быстрее12000-job infinite DES на этих коротких laws, но это
secondary diagnostic: finite predictor и infinite DES различаются моделью,
а prediction error/tail uncertainty не уравниваются одним числом jobs.

Waiting-only/combined paired wall ratios для mixed M2/M4=.92034/.92063:
canonical sorting делает combined примерно на8.6% медленнее waiting-only
в этих малых случаях, несмотря на сокращение104→97states при M4.
Symmetry не гарантирует speedup; флаг абляции позволяет выбрать вариант
по отдельному вычислительному budget. Результаты не используются для fitting.

## Итоговая проверка

Fast `pytest tests/ -m "not slow" -n auto -q`:3270passed,16warnings,373.86s.
Full `pytest tests/ -n auto -q`:3288passed,16warnings,373.08s.
Оба окончательных прогона после resource-guard patch, последовательно,
с process-local OPENBLAS/OMP/MKL_NUM_THREADS=1; repository settings не менялись.
Предварительные overlapping runs остановлены, не засчитаны как полные проверки.
54 targeted tests прошли после последней правки; новых slow tests нет.
Black/isort/gitdiffcheck чистые, pylint нового library/example кода10/10.

## Условия и состояние

Poisson arrivals с постоянной положительной интенсивностью lambda независимы
от iid finite-support joint marks `(S,O)` с вероятностями p. Внутри одной
метки S и O могут быть зависимы. На границах итераций admission FIFO до B;
decode-first выдаёт каждому декодеру один токен, остаток M расходуется
на prefills в их FIFO порядке. M>=B или None. Политика не использует
будущую длину O. Потеря определяется только числом ожидающих K, не меткой.
Стоимость итерации определяется active state, не waiting marks:
`d=alpha+sum(beta*amount+gamma*(attained+amount))` по получившим токены.

Состояние `(a,h)` хранит active `(type,attained_tokens)` и число ожидающих.
Пусть s — число survivors после итерации. За holding d принято
`c=min(N,K-h)`, N~Poisson(lambda*d); затем admission
`f=min(B-s,h+c)`, `h'=h+c-f`. Для f допущенных iid типов генерируется
слово u с вероятностью `prod(p[k] for k in u)`.
Новый active — survivors плюс `(k,0)` для допущенных.
При idle holding имеет среднее `1/lambda`, первая iid метка сразу начинает
service. Эта строка принимает ровно один запрос.

## Почему интеграция waiting marks точная

Рассмотрим фильтрацию, знающую моменты прибытия, решения admission/loss,
active marks и стадии, но не метки ещё ожидающих. Индукция по итерациям
сохраняет условный iid product law этих меток: holding/loss не смотрят на
них; новые принятые метки iid; FIFO admission раскрывает префикс, оставляя
независимый суффикс. Поэтому раскрытие метки только при admission даёт тот
же закон наблюдаемого процесса из пустого старта. Совместимый начальный
waiting law — product p, независимый от active/arrival history.

Для ordered-active count chain зададим lifting L: `(a,h)` превращается
в `(a,w)` с iid словом длины h. В row convention выполняется
`L P_full = P_count L`. Условный product law суффикса обеспечивает эту
тождественность, а holding/rewards не зависят от w при заданной длине.
Тест проверяет её для каждой достижимой строки малого случая, включая
распределение остаточного waiting word, не только стационарные средние.

Это **не strong lumpability** проекции известного waiting word в его длину.
При одинаковом finishing decoder и очередях `[type0]`, `[type1]` следующий
допущенный тип различается. Контрпример закреплён отдельным тестом.
Нельзя использовать интеграцию для условного прогноза по уже известному
содержимому реальной очереди, serially dependent marks, length-aware
admission, deadlines/cancellation или KV-based loss.

## Симметрия декодеров и запрет сортировки prefills

Декодеры получают по одному токену независимо от позиции. Их число
определяет остаток prefill budget; их вклад в holding — симметричная сумма.
Поэтому можно хранить отсортированный decoder multiset, за ним исходную
FIFO последовательность prefills. Перестановка/перемещение декодеров через
prefills не меняет действие, reward и canonical successor. Это отдельный
symmetry quotient count chain; multiplicities одинаковых декодеров сохраняются.
Если prefill станет decoder на этой итерации, его можно сортировать только
в следующем состоянии. Новые prefills добавляются после старых.

Сортировка **prefills** неверна: при ограниченном M один prompt может
исчерпать бюджет раньше другого. Тест сохраняет этот контрпример.
Флаг `canonical_decoders=False` оставляет ordered active для абляции.
На некоторых reachable laws symmetry не даёт дополнительного сокращения:
эквивалентные перестановки там просто не достижимы.

## Rewards и численное решение

При r=K-h, x=lambda*d, Q_j=P(N>=j)=gammainc(j,x) ожидаемое число
принятых за holding — A=sum(Q_j,j=1..r). Их population area
`J=sum(d*Q_j-j/lambda*Q_(j+1),j=1..r)` учитывает остаток итерации.
Это интеграл E[min(N(t),r)]dt; при r0 обе величины0.
Для `U=h+sum(q<=S_k for (k,q) in active)` six rewards:
holding d, accepted A, unfirst area Ud+J, waiting area hd+J,
total population area `(h+len(active))*d+J`, busy d.
`q==S` ещё не получило первого output token и входит в U.

Стационарная embedded pi находится регенеративным sparse solve с одной
idle visit на цикл. Mean TTFT/W/V — соответствующая pi-area / pi-accepted;
accepted rate — pi-accepted / pi-holding. Time utilization — pi-busy /
pi-holding; blocking=1-accepted_rate/lambda. Embedded pi не выдаётся как
time/Palm queue probabilities. Residual и row sums проверяются; NaN,
negative regenerative visits и complexity budgets вызывают явный отказ.
Нулевой structural error не равен rigorous floating-point certificate.

B1 сохраняет infinite M/G/1 PK shortcut полной модели, включая требование
lambda*E[full]<1; K в нём игнорируется. `states=transition_words=0`.

## Сложность и границы выигрыша

Full waiting representation имеет до R^h слов длины h; новая — одно число.
Full переходы перечисляют R^c новых arrival words, reduced — R^f допущенных,
f<=B. При фиксированном active space зависимость числа count states от K
не хуже линейной; число arrival-count branches на строку O(K). Общий
transition work может быть квадратичен в K; фазовый active space всё ещё
комбинаторно зависит от B, длин и R. `max_states` и
`max_transition_words` не заменяют support приблизительно.
Последнее поле считает planned admission words по count branches, включая
idle iid types; оно не имеет прежней full arrival-word semantics.

## Литературное пересечение

[He–Alfa2015](https://doi.org/10.1016/j.peva.2015.08.001) построили
фазово-счётную representation для discrete MAP/PH/K queues.
[He–Alfa2018](https://doi.org/10.1287/ijoc.2017.0759), §1–2 и начало §3.1,
сопоставляют tracking отдельных фаз с подсчётом фаз и относят этот подход
к известному аппарату. Здесь decoders имеют shared state-dependent holding,
а FIFO prefills несимметричны; алгоритм независимых PH servers не переносится
целиком. Используется проверенная вручную симметрия данной политики.
General Markov-function/intertwining framework известен также по
[Rogers–Pitman1981](https://doi.org/10.1214/aop/1176994363);
это bibliographic pointer, не заявление о прочитанном полном доказательстве.
Waiting product-law proof выше самодостаточен. Новая общая теорема
aggregation, новый configuration optimizer или научная новизна не заявляются.

## Воспроизведение и интерпретация

`python -m examples.stage_aware_ttft_reduction_experiment` сохраняет
[comparison/refinement](stage-aware-ttft-reduction-results-2026.json).
Отдельный `--timing` на свободной машине сохраняет
[same-clock timing](stage-aware-ttft-reduction-timing-2026.json).
Предыдущие ГЛ3 JSON не перезаписываются. Абляция отделяет waiting integration
от symmetry, но runtime отличается также из-за cache/vectorized construction;
speedup нельзя приписать только меньшему числу states.
Сравнения mean accuracy с infinite DES остаются прежними: сохранение finite
ответа не уменьшает его finite/infinite bias. Capacity refinement — диагностика,
не bound; accepted-flow mean может вести себя немонотонно из-за loss selection.

Tail/SLO и empirical-support compression остаются отдельными этапами.
Снятие waiting-word explosion позволяет рассмотреть homogeneous queue-count
blocks для infinite model, но устойчивость, численное решение и tail certificate
потребуют самостоятельного вывода. До него low loss не заменяет сертификат.
