"""Build scientific revision v4 with native editable Word equations."""

import importlib.util
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent
WORD = ROOT.parent / "word"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = load("build", WORD / "build_word.py")
style = load("style", WORD / "revise_word_v2.py")
compact = load("compact", WORD / "revise_word_v3.py")
build.SOURCE = ROOT
build.ROOT = WORD
build.STEM = "negative_queues_editable_v4"
build.CHECKS = WORD / "verification_v4"
build.REFS["tab:results_fixed"] = ("tab_2", "2", "")
build.REFS["tab:sko"] = ("tab_3", "3", "")


def data_tables(doc):
    widths = [3.6, 1.7, 3.0, 1.35, 1.7, 3.0, 1.35]
    for tbl in doc.tables:
        tbl.autofit = False
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for col, width in zip(tbl.columns, widths):
            col.width = Cm(width)
        for ri, row in enumerate(tbl.rows):
            build.prop(row._tr.get_or_add_trPr(), "cantSplit")
            if ri < 2:
                build.prop(row._tr.get_or_add_trPr(), "tblHeader")
            position = 0
            for cell in row._tr.findall(qn("w:tc")):
                pr = cell.find(qn("w:tcPr"))
                span = pr.find(qn("w:gridSpan"))
                count = int(span.get(qn("w:val"))) if span is not None else 1
                tcw = build.prop(pr, "tcW", w=sum(Cm(x).twips for x in widths[position : position + count]), type="dxa")
                pr.remove(tcw)
                pr.insert_element_before(
                    tcw,
                    "w:gridSpan",
                    "w:hMerge",
                    "w:vMerge",
                    "w:tcBorders",
                    "w:shd",
                    "w:noWrap",
                    "w:tcMar",
                    "w:textDirection",
                    "w:vAlign",
                )
                position += count
            for ci, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    build.pformat(p, 10, WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER, False)
                    f = p.paragraph_format
                    f.space_before = Pt(2)
                    f.space_after = Pt(2)
                    f.line_spacing = 1.0
                    f.keep_together = True
                    f.keep_with_next = ri < len(tbl.rows) - 1


build.make_data_tables = data_tables


def main():
    build.main()
    path = WORD / (build.STEM + ".docx")
    doc = Document(path)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = Pt(12)
        f = s.paragraph_format
        f.space_after = Pt(3)
        f.line_spacing = 1.08
        f.first_line_indent = Cm(1.25)
        f.left_indent = Cm(0)
        f.right_indent = Cm(0)
    for p in style.paragraphs(doc):
        f = p.paragraph_format
        f.widow_control = True
        for r in p.runs:
            style.font(r)
        if p.style.name.startswith("Heading"):
            f.space_before = Pt(8)
            f.space_after = Pt(4)
            f.keep_with_next = True
        elif p.style.name == "Bibliography":
            f.left_indent = Cm(0.75)
            f.first_line_indent = Cm(-0.75)
            f.space_after = Pt(4)
            f.line_spacing = 1.05
        elif p._p.pPr is not None and p._p.pPr.find(qn("w:numPr")) is not None:
            f.left_indent = Cm(0.95)
            f.first_line_indent = Cm(-0.45)
            f.space_after = Pt(3)
            f.line_spacing = 1.08
            f.tab_stops.add_tab_stop(Cm(0.95))
            f.keep_together = True
        elif p in doc.paragraphs:  # all top-level body paragraphs are handled below
            f.line_spacing = 1.08
    in_body = False
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            in_body = True
        if in_body and not p.style.name.startswith("Heading") and p.style.name != "Bibliography":
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.08
        if p._p.find(qn("m:oMathPara")) is not None:
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            nxt = p._p.getnext()
            if nxt is not None and nxt.find(qn("m:oMathPara")) is not None:
                p.paragraph_format.keep_with_next = True
        nxt = p._p.getnext()
        if p.text.rstrip().endswith(":") and nxt is not None:
            p.paragraph_format.keep_with_next = True
    for lvl in doc.part.numbering_part.element.iter(qn("w:lvl")):
        idx = int(lvl.get(qn("w:ilvl")))
        pp = style.prop(lvl, "w:pPr")
        style.prop(pp, "w:ind", left=539 + 360 * idx, hanging=255)
        tabs = style.prop(pp, "w:tabs")
        for child in list(tabs):
            tabs.remove(child)
        style.prop(tabs, "w:tab", val="num", pos=539 + 360 * idx)
        pp.remove(tabs)
        pp.insert(0, tabs)
        rp = style.prop(lvl, "w:rPr")
        style.prop(rp, "w:rFonts", ascii="Times New Roman", hAnsi="Times New Roman")
        fmt = lvl.find(qn("w:numFmt"))
        if fmt is not None and fmt.get(qn("w:val")) == "bullet":
            style.prop(lvl, "w:lvlText", val="–")
    for table in doc.tables:
        if list(table._tbl.iter(qn("w:drawing"))):
            continue
        pr = table._tbl.tblPr
        borders = style.prop(pr, "w:tblBorders")
        pr.remove(borders)
        pr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
        for child in list(borders):
            borders.remove(child)
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            style.prop(borders, "w:" + side, val="single" if side in ("top", "bottom") else "nil", sz=6, color="555555")
        for ri, row in enumerate(table.rows):
            group = len(row._tr.findall(qn("w:tc"))) == 1
            seen = set()
            for cell in row.cells:
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                cp = cell._tc.get_or_add_tcPr()
                shade = style.prop(
                    cp, "w:shd", val="clear", color="auto", fill="F2F2F2" if ri < 2 or group else "FFFFFF"
                )
                cp.remove(shade)
                cp.insert_element_before(shade, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign")
                for p in cell.paragraphs:
                    for r in p.runs:
                        style.font(r, 10)
                        if ri < 2 or group:
                            r.bold = True
    for i, shape in enumerate(doc.inline_shapes):
        shape.width = Cm(16.2)
        shape.height = Cm(9.9 if i == 0 else 8.3)
    # Literal transform and positive-part markers must not be parsed as
    # incomplete operators by LibreOffice's OMML importer.
    for run in doc._element.iter(qn("m:r")):
        t = run.find(qn("m:t"))
        if (
            t is not None
            and t.text == "+"
            and run.getparent().tag in (qn("m:sub"), qn("m:sup"))
            and len(list(run.getparent().iter(qn("m:t")))) == 1
        ):
            rp = run.find(qn("m:rPr"))
            if rp is None:
                from docx.oxml import OxmlElement

                rp = OxmlElement("m:rPr")
                run.insert(0, rp)
            for x in list(rp.findall(qn("m:sty"))):
                rp.remove(x)
            from docx.oxml import OxmlElement

            rp.insert(0, OxmlElement("m:nor"))
    # Pair short related displays; retain native OMML and 12-point body text.
    layout = []
    groups = [
        ("A1=", "B2=", [8.25, 8.25]),
        ("B1=", "C1=", [8.25, 8.25]),
        ("λw1=", "λv1=", [8.25, 8.25]),
        ("W=min", "V=min", [8.25, 8.25]),
        ("Rj=", "aj(", [10, 6.5]),
        ("(i,j)→(i+1,j):", "(i,j)→(i+1,j+1):", [8.25, 8.25]),
        ("(i,j)→(i−1,j):", "(i,j)→(i−1,j−1):", [8.25, 8.25]),
        ("W*(s)=δ", "V*(s)=δ", [8.25, 8.25]),
        ("Vзав*", "Vуд*", [8.25, 8.25]),
        ("E[T∣B=b]", "E[T2∣B=b]", [6.5, 10]),
        ("E[T]=", "E[T2]=", [5.5, 11]),
    ]

    def mtext(p):
        return "".join(t.text or "" for t in p._p.iter(qn("m:t"))).replace(" ", "")

    for first, second, widths in groups:
        ps = list(doc.paragraphs)
        for a, b in zip(ps, ps[1:]):
            if mtext(a).startswith(first) and mtext(b).startswith(second):
                compact.equation_row(doc, [a, b], widths, first + " / " + second, layout)
                break
    # Keep both large illustrations on a dedicated, legible page when necessary.
    fig = next(t for t in doc.tables if list(t._tbl.iter(qn("w:drawing"))))
    for row in fig.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
                p.paragraph_format.first_line_indent = Cm(0)
    # Allow a page break between panels, keeping each image with its caption.
    fig.rows[1].cells[0].paragraphs[0].paragraph_format.keep_with_next = False
    for ri, letter in [(1, "а"), (3, "б")]:
        p = fig.rows[ri].cells[0].paragraphs[0]
        compact.remove_text(p, "(a)" if ri == 1 else "(b)")
        first = p._p.find(qn("w:r"))
        from docx.oxml import OxmlElement

        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = "Рисунок 1" + letter + ". "
        t.set(qn("xml:space"), "preserve")
        r.append(t)
        first.addprevious(r)
        for run in p.runs:
            style.font(run, 10.5)
    for sec in doc.sections:
        for p in sec.footer.paragraphs:
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.left_indent = Cm(0)
    doc.save(path)
    (build.CHECKS / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    main()
