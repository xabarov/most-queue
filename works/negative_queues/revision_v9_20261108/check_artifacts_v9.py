"""Validate Word data fidelity, native math, references, and frozen baselines."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
WORD = ROOT.parent / "word"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(node):
    return "".join(node.xpath(".//w:t/text()|.//m:t/text()", namespaces=NS))


def math_count(obj):
    if isinstance(obj, dict):
        return int(obj.get("t") == "Math") + sum(math_count(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(math_count(v) for v in obj)
    return 0


def check(supplement=False):
    source = ROOT / "supplement" if supplement else ROOT
    stem = "negative_queues_supplement_v9" if supplement else "negative_queues_editable_v9"
    out = WORD / ("verification_supplement_v9" if supplement else "verification_v9")
    with ZipFile(WORD / (stem + ".docx")) as z:
        root = E.fromstring(z.read("word/document.xml"))
        images = [n for n in z.namelist() if n.startswith("word/media/")]
    tables = [t for t in root.xpath("//w:body/w:tbl", namespaces=NS) if not t.xpath(".//w:drawing", namespaces=NS)]
    # Formula alignment tables have no manifest and only mathematical content.
    tables = [t for t in tables if t.xpath(".//w:t", namespaces=NS)]
    specs = json.loads((source / "table-manifest.json").read_text())
    assert len(tables) == len(specs), (len(tables), len(specs))
    count = 0
    for tbl, spec in zip(tables, specs):
        expected = spec["expected_rows"]
        if not expected:
            continue
        actual = []
        labels = {row[0] for row in expected}
        for tr in tbl.xpath("w:tr", namespaces=NS):
            cells = tr.xpath("w:tc", namespaces=NS)
            if len(cells) == len(spec["widths_cm"]) and text(cells[0]).strip() in labels:
                actual.append([text(c).strip() for c in cells])
        assert actual == expected, (spec["id"], actual, expected)
        count += sum(len(row) - 1 for row in expected)
    ast = json.loads((out / "document-ast.json").read_text())
    native = len(root.xpath("//m:oMath", namespaces=NS))
    expected_math = math_count(ast)
    assert native == expected_math, (native, expected_math)
    tex = (source / "negative_queues_takahasi_takami.tex").read_text()
    matrices = len(root.xpath("//m:m", namespaces=NS))
    assert matrices == tex.count(r"\begin{pmatrix}") + tex.count(r"\begin{bmatrix}"), matrices
    assert len(images) == (0 if supplement else 4)
    bookmarks = set(root.xpath("//w:bookmarkStart/@w:name", namespaces=NS))
    for instr in root.xpath("//w:fldSimple/@w:instr", namespaces=NS):
        if instr.strip().startswith("REF "):
            assert instr.strip().split()[1] in bookmarks, instr
    pdftext = subprocess.check_output(
        ["pdftotext", "-layout", str(WORD / (stem + ".pdf")), "-"], text=True, encoding="utf-8"
    )
    assert not any(c in pdftext for c in ["¿", "\ufffd"])
    assert not re.search(r"XREFX|SEQX|TABLECAPX|FIGBLOCKX|PANELCAPX|MAINCAPX", pdftext)
    # Editorial requirements checked on the actual Word XML and rendered text.
    assert not re.search(
        r"помеченн|меченн|Основные примеры и протокол|Независимые проверки и причины расхождений", tex, re.IGNORECASE
    )
    russian = re.sub(r"Abstract\..*?Keywords:.*?(?=1\. Введение)", "", text(root), flags=re.S)
    assert not re.search(r"\b(?:Gamma|Erlang|Beta)\b", russian, re.IGNORECASE)
    groups = []
    abbreviations = []
    for tbl in tables:
        for p in tbl.xpath(".//w:p", namespaces=NS):
            value = text(p)
            if value.startswith("Коэффициент вариации:"):
                groups.append(value)
                assert not p.xpath(".//m:oMath", namespaces=NS), value
                assert p.xpath("w:pPr/w:jc/@w:val", namespaces=NS) == ["left"], value
                for run in p.xpath("w:r[w:t]", namespaces=NS):
                    assert run.xpath("w:rPr/w:sz/@w:val", namespaces=NS) == ["20"], value
                    assert not run.xpath("w:rPr/w:b[not(@w:val='0') and not(@w:val='false')]", namespaces=NS), value
            if re.search(r"\b(?:ИМ|ДИ)\b", value, re.IGNORECASE):
                assert not re.search(r"\b(?:им|ди)\b", value), value
                abbreviations.append(value)
                for run in p.xpath("w:r[w:t]", namespaces=NS):
                    assert not run.xpath(
                        "w:rPr/w:smallCaps[not(@w:val='0') and not(@w:val='false')]", namespaces=NS
                    ), value
        for run in tbl.xpath(".//m:r", namespaces=NS):
            assert run.xpath("w:rPr/w:sz/@w:val", namespaces=NS) == ["20"], text(run)
    assert len(groups) == (0 if supplement else 6), groups
    assert abbreviations
    assert not re.search(r"(?<![\w,])-\d", " ".join(text(t) for t in tables)), "hyphen instead of minus in a table"
    if not supplement:
        assert r"\subsection{Метод Такахаси--Таками для удаления и очищения}" in tex
        assert r"\appendix" not in tex
    pages = len([s for s in pdftext.split("\f") if s.strip()])
    payload = dict(
        native_equations=native,
        source_math_nodes=expected_math,
        matrices=matrices,
        images=len(images),
        numeric_cells_checked=count,
        pdf_pages=pages,
        broken_reference_fields=0,
        cv_group_headers_10pt=len(groups),
        uppercase_abbreviation_paragraphs=len(abbreviations),
        table_math_10pt=True,
        docx_sha256=sha(WORD / (stem + ".docx")),
        pdf_sha256=sha(WORD / (stem + ".pdf")),
    )
    (out / "content-checks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(stem, payload)


if __name__ == "__main__":
    for supplement in [False, True]:
        check(supplement)
    baseline = json.loads((ROOT / "baseline.json").read_text())
    repo = ROOT.parents[2]
    for path, digest in baseline.items():
        assert sha(repo / path) == digest, path
    print("Frozen v7 data and v8 documents remain unchanged.")
