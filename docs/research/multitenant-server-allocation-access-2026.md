# Л7 — доступ к первоисточнику распределения приборов

Проверка 2026-10-11. Реализация [EPIC-090](../epics/EPIC-090-multitenant-server-allocation.md)
ожидает полный текст. Метаданные подтверждены, но условия модели, формулы
и гарантии оптимизации не прочитаны. Подменять их экстраполяцией Л1/Л6 нельзя.

## Подтверждённая библиография

Ryota Okuyama, Yoshiaki Inoue, *Optimizing Server Allocation for Multi-Tenant
GPU Inference With Dynamic Batching: A Queueing Theoretic Approach*.
IEEE Transactions on Cloud Computing, volume14, issue3, pages2017–2032,
published July2026. DOI [10.1109/TCC.2026.3696613](https://doi.org/10.1109/TCC.2026.3696613).
[Crossref record](https://api.crossref.org/works/10.1109/tcc.2026.3696613)
вернул HTTP200: название, авторы, журнал, том/номер/страницы и дата сверены.
Дня публикации в поле published нет; начало license July1 не заменяет его.
Abstract в полученной записи отсутствует. License metadata не удостоверяет
доступность полного текста в текущей среде.

## Проверенные пути доступа

| Путь | Результат | Что можно заключить |
|---|---|---|
| [IEEE document11534992](https://ieeexplore.ieee.org/document/11534992), abstract и DOI redirect | web-reader вернул пустую/недоступную страницу | Основной текст и авторский abstract не прочитаны |
| [IEEE stampPDF](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11534992) | Прямой запрос HTTP418, HTML вместо PDF | PDF не получен; это ограничение запроса, не доказательство универсального paywall |
| [IEEE publisher PDF](https://ieeexplore.ieee.org/ielx8/6245519/11678135/11534992.pdf) | HTTP502, HTML вместо PDF | Не сохранён как статья; содержимое не использовано для модели |
| [arXiv API](https://export.arxiv.org/api/query) | По DOI suffix3696613 — 0 результатов; по title phrase — 1 другая работа; Okuyama AND Inoue — 2 нерелевантные работы | Нужный препринт в этих запросах не найден; это не доказательство отсутствия любой авторской версии |
| [Авторская университетская страница](https://www2.comm.eng.osaka-u.ac.jp/~yoshiaki/), [researcher profile](https://rd.iai.osaka-u.ac.jp/ja/3bbcc638cdb22e2b.html) | ConnectionError / TLS error, web-reader недоступен | Авторский список/файлы не проверены целиком |
| [Университетский репозиторий OUKA](https://ir.library.osaka-u.ac.jp/) | HTTP200; запрос с title вернул общую страницу, не подтверждённую выдачу поиска | Нельзя выдавать её за исчерпывающий поиск репозитория; целевой domain search не нашёл копию |
| Локальная литература/библиография/книги и web search по точному названию/DOI | Полного текста не найдено | Вторичные каталоги не использованы как источник формул или гарантий |

PDF-сигнатура проверена у ответов IEEE: они не начинаются с `%PDF`.
Файл PDF/HTML статьи не сохранён: страницы отказа не маскируются под первоисточник.
Чужие credentials не искались и не применялись; платный доступ, сообщения
авторам и обход ограничений не выполнялись.

## Возобновление и альтернативы

Для продолжения нужен полный текст Л7: пользовательская PDF-копия или ссылка
на доступную авторскую версию. Затем фиксируются реальные arrival/service,
batch/routing/sharing assumptions, целевая функция и ограничения; только после
этого выбираются API, independent oracle и эксперименты. Сейчас не заявляются
convexity, exact optimum, hardSLO или неэкспоненциальный контракт этой статьи.

Возможные смены порядка, **не выполненные автоматически**:

- Л8: *Batch service systems with heterogeneous servers*, van Ommeren,
  Baer, Mishra, Roy, [DOI](https://doi.org/10.1007/s11134-020-09654-y).
  Crossref подтверждает название/авторов и CC-BY4.0 license с HTML/PDF links;
  полный текст в этой проверке не прочитан и фактический download не проверен.
- Резерв слояС: Zhang, Yang, Pan, Yuan, *Joint Optimization of Resource
  Allocation and Request Batching for Multi-Tenant Inference Serving on GPU*,
  [DOI](https://doi.org/10.1109/TPDS.2025.3627574). Метаданные сверены Crossref;
  доступ к полному тексту и его условия ещё не проверены. Это другая работа,
  не эквивалентная замена Л7 без отдельного согласования.

Л6 закоммичен и отправлен в origin/main как `11959d6`; перед commit повторно
прошли66 целевых тестов и pre-commit hooks. Для проверки доступа Л7 production
код не менялся, новая модель/эксперимент не добавлялись. Карточка, журнал,
статусы и локальный архив диссертации обновлены; папка остаётся gitignored.
