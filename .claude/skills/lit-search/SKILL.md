---
name: lit-search
description: Поиск научной литературы по теории очередей (arXiv + OpenAlex + Crossref) для работ над most_queue. Use when searching for queueing theory papers, verifying references, or doing gap-analysis of models (EPIC-001 and beyond).
---

# Поиск литературы для most_queue

## Быстрый поиск по трём базам

```bash
python .claude/skills/lit-search/lit_search.py "<запрос>" --max 10
```

Опрашивает arXiv API, OpenAlex и Crossref (все бесплатные, без ключей), дедуплицирует по
названию и печатает markdown-таблицу, отсортированную по цитируемости. Опции:
`--source arxiv|openalex|crossref|all`, `--abstracts` (аннотации, только arXiv).

## Что чем искать

- **OpenAlex** — основной источник: покрывает журналы (Operations Research, Queueing Systems,
  Performance Evaluation), даёт цитируемость и DOI. Классика теории очередей — здесь.
- **Crossref** — проверка/уточнение DOI и метаданных, широкий охват.
- **arXiv** — свежие препринты (категории math.PR, cs.PF); единственный источник с аннотациями
  и бесплатными PDF: `https://arxiv.org/pdf/<id>`.
- **WebSearch/WebFetch** — обзорный поиск, страницы авторов, лекционные заметки
  (например, Harchol-Balter публикует главы книг у себя на сайте).
- **Semantic Scholar API** — без ключа почти всегда 429, не использовать.

## Куда складывать найденное

- PDF первоисточников → `docs/статьи/` (используемые в документации) или `works/articles/`.
- Конспекты → `works/queueing_systems_review/<тема>/`.
- Ссылки в roadmap/эпике оформлять как «Автор, Название, Журнал, Год, DOI».

## Чтение PDF

arXiv PDF скачивать в scratchpad (`curl -L -o`), читать инструментом Read (постранично) или
скиллом pdf. Платные журнальные PDF недоступны — искать препринт на arXiv/author page через
WebSearch или опираться на конспекты в `works/`.
