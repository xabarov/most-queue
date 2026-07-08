#!/usr/bin/env python3
"""Literature search helper: queries arXiv, OpenAlex and Crossref, prints a markdown table.

Usage:
    python lit_search.py "retrial queue M/G/1" [--max 10] [--source arxiv|openalex|crossref|all]
    python lit_search.py "MAP/PH/1" --max 5 --abstracts

Only stdlib is used. All APIs are free and keyless.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "most-queue-lit-search/1.0 (mailto:xabarov1985@gmail.com)"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def search_arxiv(query: str, limit: int) -> list[dict]:
    """Search arXiv Atom API, return normalized result dicts."""
    q = urllib.parse.quote(f'all:"{query}"' if " " in query else f"all:{query}")
    url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results={limit}&sortBy=relevance"
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(_get(url))
    out = []
    for e in root.findall("a:entry", ns):
        out.append(
            {
                "source": "arxiv",
                "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                "year": (e.findtext("a:published", "", ns) or "")[:4],
                "venue": "arXiv",
                "citations": None,
                "id": (e.findtext("a:id", "", ns) or "").replace("http://arxiv.org/abs/", "arXiv:"),
                "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split()),
            }
        )
    return out


def search_openalex(query: str, limit: int) -> list[dict]:
    """Search OpenAlex works API, return normalized result dicts."""
    q = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works?search={q}&per-page={limit}"
        "&select=title,publication_year,doi,cited_by_count,primary_location&sort=relevance_score:desc"
    )
    data = json.loads(_get(url))
    out = []
    for w in data.get("results", []):
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}).get("display_name") or ""
        out.append(
            {
                "source": "openalex",
                "title": w.get("title") or "",
                "year": str(w.get("publication_year") or ""),
                "venue": src,
                "citations": w.get("cited_by_count"),
                "id": (w.get("doi") or "").replace("https://doi.org/", "doi:"),
                "abstract": "",
            }
        )
    return out


def search_crossref(query: str, limit: int) -> list[dict]:
    """Search Crossref works API, return normalized result dicts."""
    q = urllib.parse.quote(query)
    url = (
        f"https://api.crossref.org/works?query={q}&rows={limit}"
        "&select=title,DOI,is-referenced-by-count,container-title,published"
    )
    data = json.loads(_get(url))
    out = []
    for w in data.get("message", {}).get("items", []):
        pub = w.get("published", {}).get("date-parts", [[None]])
        out.append(
            {
                "source": "crossref",
                "title": (w.get("title") or [""])[0],
                "year": str(pub[0][0] or ""),
                "venue": (w.get("container-title") or [""])[0],
                "citations": w.get("is-referenced-by-count"),
                "id": "doi:" + w.get("DOI", ""),
                "abstract": "",
            }
        )
    return out


SOURCES = {"arxiv": search_arxiv, "openalex": search_openalex, "crossref": search_crossref}


def main() -> None:
    """CLI entry point: query sources, dedup, print markdown table."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("query")
    ap.add_argument("--max", type=int, default=10, dest="limit", help="results per source")
    ap.add_argument("--source", default="all", choices=[*SOURCES, "all"])
    ap.add_argument("--abstracts", action="store_true", help="print abstracts (arXiv only)")
    args = ap.parse_args()

    names = list(SOURCES) if args.source == "all" else [args.source]
    rows = []
    for name in names:
        try:
            rows += SOURCES[name](args.query, args.limit)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[warn] {name} failed: {exc}", file=sys.stderr)
        time.sleep(0.5)

    # dedup by normalized title, prefer entries with citation counts
    seen: dict[str, dict] = {}
    for r in rows:
        key = "".join(ch for ch in r["title"].lower() if ch.isalnum())[:80]
        if key not in seen or (r["citations"] or 0) > (seen[key]["citations"] or 0):
            seen[key] = r
    rows = sorted(seen.values(), key=lambda r: -(r["citations"] or 0))

    print(f"## Результаты поиска: `{args.query}` ({len(rows)} уникальных)\n")
    print("| Год | Название | Где | Цит. | ID | Источник |")
    print("|-----|----------|-----|------|----|----------|")
    for r in rows:
        cit = "" if r["citations"] is None else str(r["citations"])
        print(f"| {r['year']} | {r['title'][:100]} | {r['venue'][:40]} | {cit} | {r['id']} | {r['source']} |")

    if args.abstracts:
        for r in rows:
            if r["abstract"]:
                print(f"\n### {r['title']}\n{r['abstract']}")


if __name__ == "__main__":
    main()
