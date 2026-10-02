# EPIC-052 — MSJ: prediction-free packing, Quickswap и ServerFilling

- **Статус:** done
- **Создан:** 2026-10-02
- **Завершён:** 2026-10-02
- **Предшественник:** [EPIC-051](EPIC-051-msj-age-residual-runtime.md)
- **Roadmap:** [MSJ](../roadmaps/msj-ph-backfilling.md)

## Цель и работы

Отделить пользу прогноза времени от выбора порядка упаковки и возможности
прерывать работы. Сравнить на одной фактической работе, не скрывая задержки
широких классов за общим средним.

- [x] FirstFit, непрерывающий MSF, one-or-all MSFQ и Adaptive
      Quickswap: проверенные по первоисточникам правила, явные ограничения API.
- [x] FCFS-based ServerFilling, preemptive-resume без накладных расходов,
      только степени двойки; сохранение работы, журнал интервалов, счётчики.
- [x] Точные трассы, информационная независимость от будущего S/прогнозов,
      capacity/work invariants, аналитические пределы M/G/1 и M/M/k.
- [x] Парный воспроизводимый эксперимент с general service, историческим KM,
      классами, хвостами, обещаниями и прерываниями; строгие JSON и отчёт.
- [x] EN/RU документация, пример, fast/full pytest, black/isort, pylint;
      обновлённый roadmap и закрытый эпик.

## Контракт и источники

MSF — nonpreemptive, убывание K, FIFO ties, пропуск не помещающихся заявок.
MSFQ — четыре фазы §4.2, порог ell от 0 до k−1 включительно; начальная
маленькая пачка допускается перед проверкой порога, чтобы пустые фазы не
создавали цикл в нулевом времени. При n1<=ell закрывается допуск до завершения
активных малых работ, затем приоритет у больших. Это буквальная фазовая версия
статьи, включая drain без ожидающих больших работ; не дополнительная эвристика
из текущего кода авторского симулятора. ell=0 совпадает с MSF.
Adaptive Quickswap проверяет trigger после заполнения в MSF-порядке; в drain
допускается только текущая наибольшая ожидающая заявка, затем working.
Ресурсный класс определяется K, а не произвольной меткой входного класса.

ServerFilling выбирает минимальный FCFS-префикс незавершённых работ с суммой
K>=k (или все при недостатке), сортирует по убыванию K и заполняет приборы.
Пересчёт на arrivals/completions; вытесненная работа продолжается с остатка.
Это НЕ ServerFilling-SRPT: остаток знает только движок событий. И k, и K —
степени двойки. W включает все паузы, не только ожидание первого старта.

- Chen et al. *Improving nonpreemptive multiserver job scheduling with
  quickswap*, Performance Evaluation 171 (2026), 102525,
  [DOI](https://doi.org/10.1016/j.peva.2025.102525),
  [авторский PDF](https://jcpwfloi.com/assets/publications/performance2025-final49.pdf),
  [симулятор авторов](https://github.com/UniVe-NeDS-Lab/mjqm-simulator).
- Grosof, Harchol-Balter. *ServerFilling: A better approach to packing
  multiserver jobs*, ApPLIED 2023,
  [DOI](https://doi.org/10.1145/3584684.3597264),
  [авторский PDF, §6](https://www.cs.cmu.edu/~harchol/Papers/Applied23.pdf).

Теоремы MSFQ для Poisson/exponential one-or-all не переносим на general service
и Adaptive Quickswap. ServerFilling — отдельный идеальный preemption-reference,
не реалистичное обещание бесплатного checkpoint/resume. Static Quickswap,
DivisorFilling, SRPT, цена checkpoint, реальные трассы — вне этого эпика.

## Протокол до запуска

k=4. Режимы: one_or_all K=(1,4), p=(.8,.2), mu=(.4,2.2);
powers_of_two K=(1,2,4) и general K=(1,3,4), p=(.5,.3,.2), mu=(.4,2.2,2.2).
S=mu[class]*exp(.7*Z−.7²/2)*U, Z~N(0,1); независимый U~Gamma(2,.5)
или mean-one lognormal с CV=(.8,2) / (.8,1.5,2). Это не маргинальный Erlang.
lambda=rho*k/E[K*S], rho=.35/.55, без нормировки по случайному test.

Новые seed 52000–52007; 400 warm-up + 4000 измеряемых приходов; независимая
история 4000 с C~Exp(mean=5*mu[class]). KM по min(S,C), completed=(S<=C),
квантиль .90, фиксированный до replay. Раздельные RNG history/censor/test/arrivals.
Без oracle и подбора ell на test. Одинаковые трассы и оценки внутри сценария.

FCFS, FirstFit, MSF, Adaptive Quickswap; EASY/conservative × KM-fixed/KM-age;
ServerFilling только в первых двух режимах; MSFQ ell=1 и ell=3 только one-or-all.
(11+9+8)*2 закона*2 нагрузки*8 seed = 896 scheduler-прогонов.
Неприменимые политики явно отсутствуют в протоколе, заявки не удаляются.

Метрики: средние W/T, p99 T, средние/хвосты по классам, weighted mean T с
весами p*K*mu/E[K*S], utilization/idle-with-queue; обещания и нарушения только
для backfilling, прерывания и число затронутых работ для ServerFilling.
W=всё время вне сервиса, weighted T не является доказательством fairness.
Счётчики — вся трасса с warm-up/drain; задержки — измеряемая arrival-cohort;
time averages — от первого измеряемого до последнего прихода.
95% Student-интервалы по восьми seed, парные разности с FCFS и MSF, без
поправки на множественность. Повтор двух seed каждого режима для аудита.

## Результаты

- Пять prediction-free дисциплин в `MsjGeneralSim`, явная область каждой;
  чистый selector получает только `(id, K)`. Существующие FCFS/EASY/conservative
  defaults и API сохранены. ServerFilling ведёт интервалы исполнения и счётчики,
  возобновляет остаток без потери работы; W включает все паузы.
- 72 новых тестовых случая, в том числе десять сверок M/E2/1 и M/M/4 по общим
  допускам. Дополнительный независимый remaining-work reference подтвердил
  160 трасс ServerFilling для k=1,2,4,8 по стартам, завершениям, W и прерываниям.
- 896 scheduler-прогонов, шесть strict JSON файлов; все 4400 работ каждого
  прогона дренированы. 224 расписания повторены точно; аудит всех артефактов
  воспроизвёл 7264 интервальных оценки. Старые артефакты не изменены.
- FirstFit улучшил общее mean T/p99 во всех 12 сценариях, но mean T K=4
  ухудшился в двух. MSF улучшил mean T K=4 во всех 12, но проиграл общему
  mean T в четырёх. MSFQ ell=1/3 не улучшил MSF по общему/weighted mean T.
  ServerFilling улучшил mean T в 6/8 точек при сотнях zero-cost прерываний.
  Это наблюдения объявленной синтетической сетки, не универсальные гарантии.
- Быстрый pytest: **1111 passed** (320.30 s); полный: **1120 passed**
  (546.22 s), оба с 16 прежними предупреждениями, без rerun/изменения допусков.
  Black/isort, pre-commit JSON/размер/black и `git diff --check` пройдены.
  Pylint MSJ-модулей, примера и новых тестов — 10/10; весь `most_queue` —
  те же 43 замечания, без добавлений. В `structs.py` сохранены прежние
  несвязанные Unicode-комментарии и замечание о локальном import.
- [Методика и исполняемый пример EN](../msj_packing.md), EN/RU страницы модели
  и симуляции, [полный отчёт](../research/msj-packing-results-2026-10.md),
  [артефакты/команды](../../works/msj_packing/README.md), roadmap обновлены.

Следующий кандидат — явная цена checkpoint/resume, с сохранением текущего
ServerFilling как zero-cost reference. Реальные трассы, drift, Static Quickswap,
DivisorFilling и SRPT не реализованы и не входят в закрытый объём этого эпика.
