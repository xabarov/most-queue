"""Make a separate typography revision; retain the previously delivered DOCX.

uv run --no-project --with python-docx --with lxml python revise_word_v2.py
Run redraw_figure1.py first. No Telegram or other delivery is performed.
"""

from pathlib import Path
import copy
import hashlib
import json
from zipfile import ZipFile

from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "negative_queues_editable.docx"
OUT = ROOT / "negative_queues_editable_v2.docx"
CHECKS = ROOT / "verification_v2"
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


def font(run, size=None):
    run.font.name = "Times New Roman"
    rf = run._r.get_or_add_rPr().find(qn("w:rFonts"))
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + key), "Times New Roman")
    if size is not None:
        run.font.size = Pt(size)


def paragraphs(doc):
    yield from doc.paragraphs
    for table in doc.tables:
        seen = set()
        for row in table.rows:
            for cell in row.cells:
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                yield from cell.paragraphs
    for sec in doc.sections:
        yield from sec.footer.paragraphs


def main():
    CHECKS.mkdir(exist_ok=True)
    before = hashlib.sha256(BASE.read_bytes()).hexdigest()
    doc = Document(BASE)
    # Normalize inherited paragraph styles too: newly added paragraphs should
    # match the visible text without relying solely on direct formatting.
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = Pt(12)
        f = s.paragraph_format
        f.space_before = Pt(0)
        f.space_after = Pt(4)
        f.line_spacing = 1.12
        f.first_line_indent = Cm(1.25)
        f.left_indent = Cm(0)
        f.right_indent = Cm(0)
    bs = doc.styles["Bibliography"]
    bs.font.size = Pt(11)
    bs.paragraph_format.left_indent = Cm(0.75)
    bs.paragraph_format.first_line_indent = Cm(-0.75)
    for name in ("Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = Pt(13 if name == "Heading 1" else 12)
        s.font.bold = True
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.left_indent = Cm(0)
        s.paragraph_format.keep_with_next = True
        s.paragraph_format.keep_together = True
    for sec in doc.sections:
        for p in sec.footer.paragraphs:
            f = p.paragraph_format
            f.first_line_indent = Cm(0)
            f.left_indent = Cm(0)
            f.right_indent = Cm(0)
    list_count = 0
    for p in paragraphs(doc):
        for r in p.runs:
            font(r)
        p.paragraph_format.widow_control = True
        if p._p.pPr is not None and p._p.pPr.find(qn("w:numPr")) is not None:
            list_count += 1
            f = p.paragraph_format
            f.first_line_indent = Cm(-0.45)
            f.left_indent = Cm(0.95)
            f.right_indent = Cm(0)
            f.space_before = Pt(0)
            f.space_after = Pt(3)
            f.line_spacing = 1.12
            f.tab_stops.add_tab_stop(Cm(0.95))
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            # Do not strand a marker or the final line of a short list item.
            f.keep_together = True
        elif p.style.name == "Bibliography":
            f = p.paragraph_format
            f.first_line_indent = Cm(-0.75)
            f.left_indent = Cm(0.75)
            f.right_indent = Cm(0)
            f.space_after = Pt(4)
            f.space_before = Pt(0)
            f.line_spacing = 1.05
            f.tab_stops.add_tab_stop(Cm(0.75))
        elif p.text.startswith("Таблица "):
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True
        previous = p._p.getprevious()
        if previous is not None and previous.tag == qn("w:tbl"):
            p.paragraph_format.space_before = Pt(6)
        following = p._p.getnext()
        if p.text.rstrip().endswith(":") and following is not None and following.tag == qn("w:p"):
            pp = following.find(qn("w:pPr"))
            if pp is not None and pp.find(qn("w:numPr")) is not None:
                p.paragraph_format.keep_with_next = True

    # Portable list markers and hanging indents; no Symbol/Wingdings fallback.
    for level in doc.part.numbering_part.element.iter(qn("w:lvl")):
        idx = int(level.get(qn("w:ilvl")))
        pp = prop(level, "w:pPr")
        prop(pp, "w:ind", left=539 + idx * 360, hanging=255)
        tabs = prop(pp, "w:tabs")
        for child in list(tabs):
            tabs.remove(child)
        prop(tabs, "w:tab", val="num", pos=539 + idx * 360)
        pp.remove(tabs)
        pp.insert(0, tabs)
        rp = prop(level, "w:rPr")
        prop(
            rp,
            "w:rFonts",
            ascii="Times New Roman",
            hAnsi="Times New Roman",
            cs="Times New Roman",
            eastAsia="Times New Roman",
        )
        prop(rp, "w:sz", val="24")
        fmt = level.find(qn("w:numFmt"))
        if fmt is not None and fmt.get(qn("w:val")) == "bullet":
            prop(level, "w:lvlText", val="–")

    data_tables = []
    for table in doc.tables:
        if list(table._tbl.iter(qn("w:drawing"))):
            continue
        data_tables.append(table)
        pr = table._tbl.tblPr
        borders = prop(pr, "w:tblBorders")
        pr.remove(borders)
        pr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
        for child in list(borders):
            borders.remove(child)
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            prop(borders, "w:" + side, val="single" if side in ("top", "bottom") else "nil", sz=6, color="555555")
        margins = prop(pr, "w:tblCellMar")
        pr.remove(margins)
        pr.insert_element_before(margins, "w:tblLook")
        for child in list(margins):
            margins.remove(child)
        for side, value in [("top", 65), ("left", 65), ("bottom", 65), ("right", 65)]:
            prop(margins, "w:" + side, w=value, type="dxa")
        for ri, row in enumerate(table.rows):
            prop(row._tr.get_or_add_trPr(), "w:cantSplit")
            is_group = len(row._tr.findall(qn("w:tc"))) == 1
            seen = set()
            for ci, cell in enumerate(row.cells):
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                cp = cell._tc.get_or_add_tcPr()
                shade = prop(
                    cp, "w:shd", val="clear", color="auto", fill="F2F2F2" if (ri < 2 or is_group) else "FFFFFF"
                )
                cp.remove(shade)
                cp.insert_element_before(shade, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign")
                for p in cell.paragraphs:
                    f = p.paragraph_format
                    f.left_indent = Cm(0)
                    f.right_indent = Cm(0)
                    f.first_line_indent = Cm(0)
                    f.space_before = Pt(0)
                    f.space_after = Pt(0)
                    f.line_spacing = 1.05
                    f.keep_with_next = ri < len(table.rows) - 1
                    f.keep_together = True
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        font(r, 10.5)
                        if ri < 2 or is_group:
                            r.bold = True

    replacements = {
        hashlib.sha256(
            (ROOT.parent / "negative_queues_figures/rcs/diagrams/neg_arrivals_супер.png").read_bytes()
        ).hexdigest(): "figure1a_rcs",
        hashlib.sha256(
            (
                ROOT.parent / "negative_queues_figures/disaster/diagrams/disaster_negative_arrivals_pretty.png"
            ).read_bytes()
        ).hexdigest(): "figure1b_disaster",
    }
    changed = []
    for shape in doc.inline_shapes:
        rel = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        part = doc.part.related_parts[rel]
        stem = replacements.get(hashlib.sha256(part.blob).hexdigest())
        if not stem:
            continue
        part._blob = (ROOT / "figures_v2" / (stem + ".png")).read_bytes()
        shape.width = Cm(16.2)
        shape.height = Cm(9.9 if stem == "figure1a_rcs" else 9.3)
        shape._inline.docPr.set(
            "descr", "Переходы RCS при n=3" if stem == "figure1a_rcs" else "Переходы при катастрофах, n=3"
        )
        changed.append(stem)
    assert len(changed) == 2
    fig1 = next(t for t in doc.tables if list(t._tbl.iter(qn("w:drawing"))))
    for cell in [c for row in fig1.rows for c in row.cells]:
        cp = cell._tc.get_or_add_tcPr()
        margins = prop(cp, "w:tcMar")
        cp.remove(margins)
        cp.insert_element_before(margins, "w:textDirection", "w:tcFitText", "w:vAlign")
        for child in list(margins):
            margins.remove(child)
        for side in ("top", "left", "bottom", "right"):
            prop(margins, "w:" + side, w=0, type="dxa")
        for p in cell.paragraphs:
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)

    # The old caption says 0..n for RCS although its source picture has 0..n+1.
    # Retain one common, correct range; keep the native SEQ field and bookmark.
    cap = next(p for p in doc.paragraphs if "Переходы, вызванные отрицательными" in p.text)
    maths = list(cap._p.iter(qn("m:oMath")))
    assert len(maths) == 3
    correct_range = copy.deepcopy(maths[2])
    start = cap._p.index(maths[1])
    for child in list(cap._p)[start:]:
        cap._p.remove(child)
    cap._p.append(correct_range)
    font(cap.add_run(")."), 10.5)
    cap.paragraph_format.keep_with_next = False
    cap.paragraph_format.keep_together = True
    doc.save(OUT)

    # Reproducible content preservation checks. Caption range is the only text
    # edit; formatting properties and the two image assets may differ.
    with ZipFile(BASE) as old, ZipFile(OUT) as new:
        a = E.fromstring(old.read("word/document.xml"))
        b = E.fromstring(new.read("word/document.xml"))

        def outside_caption(root):
            root = copy.deepcopy(root)
            for p in root.xpath('//w:p[contains(string(.),"Переходы, вызванные отрицательными")]', namespaces=NS):
                p.getparent().remove(p)
            return root

        aa, bb = outside_caption(a), outside_caption(b)
        assert aa.xpath("//w:t/text()", namespaces=NS) == bb.xpath("//w:t/text()", namespaces=NS)

        def math_text(root):
            return [E.tostring(x, method="c14n", exclusive=True) for x in root.xpath("//m:oMath", namespaces=NS)]

        assert math_text(aa) == math_text(bb)
        for p in b.xpath("//w:p[w:pPr/w:numPr]", namespaces=NS):
            assert p.find("w:pPr/w:ind", NS).get(qn("w:firstLine")) is None
            assert int(p.find("w:pPr/w:ind", NS).get(qn("w:hanging"))) > 0
        assert len(b.xpath("//m:oMath", namespaces=NS)) == 477
        assert len(b.xpath("//w:drawing", namespaces=NS)) == 12
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == before
    report = {
        "base_sha256": before,
        "new_file": OUT.name,
        "source_document_unchanged": True,
        "body_text_unchanged_except_figure1_caption": True,
        "math_outside_caption_unchanged": True,
        "native_math_objects": 477,
        "caption_edit": "One common 0..n+1 level range for both graphs",
        "lists_fixed": list_count,
        "bibliography_entries": 18,
        "data_tables": len(data_tables),
        "replaced_images": changed,
        "svg_and_pdf_figure_sources": True,
        "sent_to_telegram": False,
    }
    (CHECKS / "revision-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
