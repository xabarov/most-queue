"""Verify numbers, native equations, fields and preservation of previous versions."""

import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path(__file__).resolve().parent
WORD = ROOT.parent / "word"
OUT = WORD / "verification_v4"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def celltext(node):
    return "".join(node.xpath(".//w:t/text()|.//m:t/text()", namespaces=NS))


def num(x):
    return f"{x:.4f}".replace(".", ",")


def main():
    file = WORD / "negative_queues_editable_v4.docx"
    with ZipFile(file) as z:
        root = E.fromstring(z.read("word/document.xml"))
        images = [n for n in z.namelist() if n.startswith("word/media/")]
    tables = [
        t
        for t in root.xpath("//w:body/w:tbl", namespaces=NS)
        if len(t.xpath("w:tblGrid/w:gridCol", namespaces=NS)) == 7
    ]
    assert len(tables) == 3
    data = json.loads((ROOT / "experiment-results.json").read_text())["results"]
    table_data = [[r for r in data if r["scenario"] != "fixed"], [r for r in data if r["scenario"] == "fixed"], data]
    checked = 0
    for idx, (table, rows) in enumerate(zip(tables, table_data)):
        actual = []
        for tr in table.xpath("w:tr", namespaces=NS):
            cells = tr.xpath("w:tc", namespaces=NS)
            if len(cells) == 7 and re.fullmatch(r"-?\d+,\d+", celltext(cells[1])):
                actual.append([celltext(c) for c in cells[1:]])
        expected = []
        for cv in [0.8, 1.2]:
            for row in rows:
                if row["cv"] != cv:
                    continue
                values = []
                for key in ["w", "v"] if idx < 2 else ["sw", "sv"]:
                    c = row["calculation"][key]
                    if isinstance(c, list):
                        c = c[0]
                    s = row["simulation"][key]
                    values += [
                        num(c),
                        num(s["mean"]) + " ± " + num(s["half_ci95"]),
                        f"{100*(c/s['mean']-1):.2f}".replace(".", ","),
                    ]
                expected.append(values)
        assert actual == expected, (idx, actual, expected)
        checked += sum(len(r) for r in actual)

    def math_count(x):
        if isinstance(x, dict):
            return int(x.get("t") == "Math") + sum(math_count(v) for v in x.values())
        if isinstance(x, list):
            return sum(math_count(v) for v in x)
        return 0

    ast = json.loads((OUT / "document-ast.json").read_text())
    source_math = math_count(ast)
    native_math = len(root.xpath("//m:oMath", namespaces=NS))
    assert native_math == source_math, (native_math, source_math)
    assert len(root.xpath("//m:m", namespaces=NS)) == 6
    assert len(images) == 2
    bookmarks = set(root.xpath("//w:bookmarkStart/@w:name", namespaces=NS))
    refs = root.xpath("//w:fldSimple/@w:instr", namespaces=NS)
    for instr in refs:
        if instr.strip().startswith("REF "):
            assert instr.strip().split()[1] in bookmarks, instr
    old_hashes = {
        "negative_queues_editable.docx": "026c8691755ad191eda921ff66fd7f136d01842bc78570125707bc8713008e99",
        "negative_queues_editable.pdf": "03a2c7d30e2a727a899dffc40f503ec3036651f4860eb6eb203e7564444a87a8",
        "negative_queues_editable_v2.docx": "9fb0777b29e3fa0a088c3a77f8e51e3d5a8f03539d595da44616b1501c25d249",
        "negative_queues_editable_v2.pdf": "822a1a9ae3b8d39513a4e65c7704b9d22a71dbd3a8275f2f8c01b71faad41ded",
    }
    for name, value in old_hashes.items():
        assert sha(WORD / name) == value, name
    manifest = json.loads((ROOT / "baseline-manifest.json").read_text())
    repo = ROOT.parents[2]
    for path, value in manifest["files"].items():
        if path.startswith("works/"):
            assert sha(repo / path) == value, path
    preview = Path("/tmp/negative-word-v4-20261107/preview.txt").read_text()
    assert not any(c in preview for c in ["¿", "\ufffd"])
    pages = len([p for p in preview.split("\f") if p.strip()])
    shutil.copy2(
        "/tmp/negative-word-v4-20261107/render/negative_queues_editable_v4.pdf",
        WORD / "negative_queues_editable_v4.pdf",
    )
    result = dict(
        native_equations=native_math,
        source_math_nodes=source_math,
        matrices=6,
        images=2,
        numeric_cells_checked=checked,
        rows_checked=24,
        pdf_pages=pages,
        previous_versions_unchanged=True,
        original_article_and_tables_unchanged=True,
        broken_reference_fields=0,
        docx_sha256=sha(file),
        pdf_sha256=sha(WORD / "negative_queues_editable_v4.pdf"),
    )
    (OUT / "content-checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
