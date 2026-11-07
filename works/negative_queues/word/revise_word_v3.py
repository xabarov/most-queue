"""Remove redundant English glosses and compact related equations, retaining v2.

uv run --no-project --with python-docx --with lxml python revise_word_v3.py
"""

from pathlib import Path
from zipfile import ZipFile
import copy
import hashlib
import json
import re
from collections import Counter

from docx import Document
from docx.shared import Cm, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "negative_queues_editable_v2.docx"
OUT = ROOT / "negative_queues_editable_v3.docx"
CHECKS = ROOT / "verification_v3"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def prop(parent, tag, **attrs):
    node = parent.find(qn(tag))
    if node is None:
        node = OxmlElement(tag)
        parent.append(node)
    for key, value in attrs.items():
        node.set(qn("w:" + key), str(value))
    return node


def remove_text(p, old):
    """Delete a literal text span across runs without changing math or run styles."""
    nodes = list(p._p.iter(qn("w:t")))
    text = "".join(n.text or "" for n in nodes)
    assert text.count(old) == 1, (old, text)
    lo = text.index(old)
    hi = lo + len(old)
    at = 0
    for node in nodes:
        val = node.text or ""
        end = at + len(val)
        if at < hi and end > lo:
            node.text = val[: max(0, lo - at)] + val[min(len(val), hi - at) :]
            node.set(qn("xml:space"), "preserve")
        at = end


def equation_row(doc, paras, widths, label, log):
    assert all(p._p.find(qn("m:oMathPara")) is not None for p in paras)
    assert all(a._p.getnext() is b._p for a, b in zip(paras, paras[1:]))
    tbl = doc.add_table(rows=1, cols=len(paras))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    pr = tbl._tbl.tblPr
    width = prop(pr, "w:tblW", w=sum(Cm(w).twips for w in widths), type="dxa")
    borders = prop(pr, "w:tblBorders")
    pr.remove(borders)
    pr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        prop(borders, "w:" + side, val="nil")
    margins = prop(pr, "w:tblCellMar")
    pr.remove(margins)
    pr.insert_element_before(margins, "w:tblLook")
    for side in ("top", "left", "bottom", "right"):
        prop(margins, "w:" + side, w=0, type="dxa")
    for col, width in zip(tbl.columns, widths):
        col.width = Cm(width)
    prop(tbl.rows[0]._tr.get_or_add_trPr(), "w:cantSplit")
    paras[0]._p.addprevious(tbl._tbl)
    for cell, p, width in zip(tbl.rows[0].cells, paras, widths):
        cell.width = Cm(width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        f = p.paragraph_format
        f.left_indent = Cm(0)
        f.right_indent = Cm(0)
        f.first_line_indent = Cm(0)
        f.space_before = Pt(2)
        f.space_after = Pt(4)
        f.keep_together = True
        f.keep_with_next = False
        cell._tc.remove(cell.paragraphs[0]._p)
        cell._tc.append(p._p)
    log.append({"group": label, "formula_count": len(paras), "widths_cm": widths})


def main():
    CHECKS.mkdir(exist_ok=True)
    baseline_hash = hashlib.sha256(BASE.read_bytes()).hexdigest()
    doc = Document(BASE)
    changes = []
    # Russian prose only: keep the English abstract and bibliographic titles.
    redundant = [" (negative customers)", " (product-form)", ", level", ", microstate", " (fixed-service restarts)"]
    for old in redundant:
        found = [
            p
            for p in doc.paragraphs
            if old in p.text and re.search("[А-Яа-яЁё]", p.text) and p.style.name != "Bibliography"
        ]
        assert len(found) == 1, (old, len(found))
        remove_text(found[0], old)
        changes.append(
            {"removed": old, "reason": "The Russian term is sufficient; the English gloss is not used later."}
        )
    expansions = [
        p for p in doc.paragraphs if " (Remove Customer in Service)" in p.text and re.search("[А-Яа-яЁё]", p.text)
    ]
    assert len(expansions) == 3
    for p in expansions[1:]:
        remove_text(p, " (Remove Customer in Service)")
        changes.append(
            {
                "removed": " (Remove Customer in Service)",
                "reason": "RCS is already expanded at first mention in the Russian introduction.",
            }
        )
    # Allow a break between the two large panels, never between a panel and
    # its caption. Each caption identifies figure 1 with a live REF field.
    figure1 = next(t for t in doc.tables if list(t._tbl.iter(qn("w:drawing"))))
    for ri, letter, old_label in [(1, "а", "(a)"), (3, "б", "(b)")]:
        p = figure1.rows[ri].cells[0].paragraphs[0]
        remove_text(p, old_label)
        prefix = []
        for text in ["Рисунок ", letter + "."]:
            run = OxmlElement("w:r")
            rp = OxmlElement("w:rPr")
            prop(rp, "w:rFonts", ascii="Times New Roman", hAnsi="Times New Roman")
            prop(rp, "w:sz", val=21)
            run.append(rp)
            t = OxmlElement("w:t")
            t.text = text
            t.set(qn("xml:space"), "preserve")
            run.append(t)
            prefix.append(run)
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), "REF fig_1 \\h")
        fr = copy.deepcopy(prefix[0])
        fr.find(qn("w:t")).text = "1"
        fld.append(fr)
        insertion = 1 if p._p.pPr is not None else 0
        for k, node in enumerate([prefix[0], fld, prefix[1]]):
            p._p.insert(insertion + k, node)
        p.paragraph_format.keep_with_next = ri == 3
    expected_paragraphs = Counter(
        "".join(t.text or "" for t in p.iter(qn("w:t"))) for p in doc._element.iter(qn("w:p"))
    )

    # A small spacing adjustment keeps the 12 pt type, page margins and all
    # image sizes while avoiding whole-page losses at large block boundaries.
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        doc.styles[name].paragraph_format.line_spacing = 1.08
        doc.styles[name].paragraph_format.space_after = Pt(3)
    in_body = False
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            in_body = True
        if not in_body:
            continue
        if p.style.name.startswith("Heading"):
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
        elif (
            p.style.name != "Bibliography"
            and not p.text.startswith(("Таблица ", "Рисунок "))
            and p._p.find(qn("m:oMathPara")) is None
        ):
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.space_after = Pt(3)
    # Tighten spacing around displays and between adjacent display lines.
    for p in doc.paragraphs:
        if p._p.find(qn("m:oMathPara")) is None:
            continue
        f = p.paragraph_format
        prev = p._p.getprevious()
        nxt = p._p.getnext()
        follows_formula = prev is not None and prev.find(qn("m:oMathPara")) is not None
        followed_by_formula = nxt is not None and nxt.find(qn("m:oMathPara")) is not None
        f.space_before = Pt(0 if follows_formula else 2)
        f.space_after = Pt(1 if followed_by_formula else 4)
        f.keep_with_next = followed_by_formula
    ps = list(doc.paragraphs)
    # The source version is fixed, so indices identify exact existing displays.
    # Additional formula-prefix assertions guard against using a different file.
    groups = [
        ([49, 50, 51], [5, 4, 7.5], "A1, B2, D2 — базовые матрицы", ["A", "B", "D"]),
        ([94, 95], [8.25, 8.25], "B1, C1 — матрицы катастроф", ["B", "C"]),
        ([105, 106], [8.25, 8.25], "S и распределение Y", ["S", "Y"]),
        ([114, 115], [6.5, 10], "Первый и второй условные моменты T", ["E", "E"]),
        ([120, 121], [8.25, 8.25], "Определения W и V", ["W", "V"]),
        ([125, 126], [8.25, 8.25], "ПЛС W и V", ["W", "V"]),
        ([130, 131], [7.2, 9.3], "Условные ПЛС для двух исходов", ["E", "E"]),
    ]
    layouts = []
    for indices, widths, title, prefixes in groups:
        selected = [ps[i] for i in indices]
        for p, prefix in zip(selected, prefixes):
            texts = [t.text or "" for t in p._p.iter(qn("m:t"))]
            assert texts and "".join(texts).startswith(prefix), (title, texts)
        equation_row(doc, selected, widths, title, layouts)
    ps[100].paragraph_format.space_before = Pt(3)
    # Short enumerations should not leave their second item on the next page.
    list_groups = {}
    for p in doc.paragraphs:
        if p._p.pPr is None:
            continue
        np = p._p.pPr.find(qn("w:numPr"))
        if np is not None:
            ident = np.find(qn("w:numId")).get(qn("w:val"))
            list_groups.setdefault(ident, []).append(p)
    for group in list_groups.values():
        if sum(len(p.text) for p in group) < 600:
            for p in group[:-1]:
                p.paragraph_format.keep_with_next = True
    # An introductory colon belongs with the first equation, including the
    # newly combined equation rows.
    for p in doc.paragraphs:
        nxt = p._p.getnext()
        if not p.text.rstrip().endswith(":") or nxt is None:
            continue
        if nxt.find(qn("m:oMathPara")) is not None or (nxt.tag == qn("w:tbl") and list(nxt.iter(qn("m:oMathPara")))):
            p.paragraph_format.keep_with_next = True
    doc.save(OUT)

    with ZipFile(BASE) as old, ZipFile(OUT) as new:
        a = E.fromstring(old.read("word/document.xml"))
        b = E.fromstring(new.read("word/document.xml"))

        def maths(root):
            return [E.tostring(x, method="c14n", exclusive=True) for x in root.xpath("//m:oMath", namespaces=NS)]

        assert Counter(maths(a)) == Counter(maths(b))
        assert expected_paragraphs == Counter(
            "".join(p.xpath(".//w:t/text()", namespaces=NS)) for p in b.xpath("//w:p", namespaces=NS)
        )
        for path in old.namelist():
            if path.startswith("word/media/"):
                assert old.read(path) == new.read(path)
        assert len(b.xpath("//m:oMath", namespaces=NS)) == 477
        assert len(b.xpath("//m:m", namespaces=NS)) == 6
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == baseline_hash
    report = {
        "base_document": BASE.name,
        "base_sha256": baseline_hash,
        "base_unchanged": True,
        "english_gloss_edits": changes,
        "formula_layout_groups": layouts,
        "all_477_native_math_objects_unchanged": True,
        "all_6_matrices_unchanged": True,
        "all_images_unchanged": True,
        "body_font_size_pt": 12,
        "numeric_table_font_size_pt": 10.5,
        "figure1_position": "Original position; page break permitted between panels, each labelled Figure 1a/1b with a live REF field.",
        "body_line_spacing": {"before": 1.12, "after": 1.08},
        "body_paragraph_after_pt": {"before": 4, "after": 3},
        "heading_spacing_before_after_pt": [8, 4],
        "page_margins_and_figure_sizes_unchanged": True,
        "sent_to_telegram": False,
    }
    (CHECKS / "revision-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
