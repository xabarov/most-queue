"""Native Word equations and consistent layout for the article and supplement."""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent
WORD = ROOT.parent / "word"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(supplement=False):
    build = load("word_base", WORD / "build_word.py")
    style = load("word_style", WORD / "revise_word_v2.py")
    compact = load("word_compact", WORD / "revise_word_v3.py")
    build.SOURCE = ROOT / "supplement" if supplement else ROOT
    build.ROOT = WORD
    build.STEM = "negative_queues_supplement_v7" if supplement else "negative_queues_editable_v7"
    build.CHECKS = WORD / ("verification_supplement_v7" if supplement else "verification_v7")
    refs = json.loads((build.SOURCE / "references.json").read_text())
    build.REFS = refs
    table_specs = json.loads((build.SOURCE / "table-manifest.json").read_text())

    def data_tables(doc):
        assert len(doc.tables) == len(table_specs), (len(doc.tables), len(table_specs))
        for tbl, spec in zip(doc.tables, table_specs):
            widths = spec["widths_cm"]
            heads = spec["header_rows"]
            tbl.autofit = False
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            for col, w in zip(tbl.columns, widths):
                col.width = Cm(w)
            for ri, row in enumerate(tbl.rows):
                pr = row._tr.get_or_add_trPr()
                build.prop(pr, "cantSplit")
                if ri < heads:
                    build.prop(pr, "tblHeader")
                position = 0
                for cell in row._tr.findall(qn("w:tc")):
                    cp = cell.find(qn("w:tcPr"))
                    span = cp.find(qn("w:gridSpan"))
                    count = int(span.get(qn("w:val"))) if span is not None else 1
                    tcw = build.prop(
                        cp, "tcW", w=sum(Cm(x).twips for x in widths[position : position + count]), type="dxa"
                    )
                    cp.remove(tcw)
                    cp.insert_element_before(
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
                        build.pformat(
                            p,
                            spec.get("font_pt", 10),
                            WD_ALIGN_PARAGRAPH.LEFT if ci == 0 or len(widths) == 3 else WD_ALIGN_PARAGRAPH.CENTER,
                            False,
                        )
                        p.paragraph_format.line_spacing = 1
                        p.paragraph_format.space_before = Pt(2)
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.keep_with_next = ri < heads
                        p.paragraph_format.keep_together = True

    build.make_data_tables = data_tables
    build.main()
    path = WORD / (build.STEM + ".docx")
    doc = Document(path)
    body_spacing = 1.08 if supplement else 1.05
    for name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = Pt(12)
        f = s.paragraph_format
        f.line_spacing = body_spacing
        f.space_after = Pt(3)
        f.first_line_indent = Cm(1.25)
        f.left_indent = Cm(0)
        f.right_indent = Cm(0)
    for p in style.paragraphs(doc):
        p.paragraph_format.widow_control = True
        for run in p.runs:
            style.font(run)
        if p.style.name.startswith("Heading"):
            f = p.paragraph_format
            f.space_before = Pt(8)
            f.space_after = Pt(4)
            f.keep_with_next = True
        if p.style.name == "Bibliography":
            f = p.paragraph_format
            f.left_indent = Cm(0.75)
            f.first_line_indent = Cm(-0.75)
            f.space_after = Pt(4)
            f.line_spacing = 1.05
    for p in doc.paragraphs:
        if p.style.name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
            p.paragraph_format.line_spacing = body_spacing
            p.paragraph_format.space_after = Pt(3)
        if list(p._p.iter(qn("m:oMathPara"))):
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
        if p.text.rstrip().endswith(":"):
            p.paragraph_format.keep_with_next = True
    # Fixed table geometry and restrained rules, with repeated column headings.
    for tbl in doc.tables:
        if list(tbl._tbl.iter(qn("w:drawing"))):
            continue
        spec = table_specs.pop(0)
        prop = tbl._tbl.tblPr
        borders = style.prop(prop, "w:tblBorders")
        prop.remove(borders)
        prop.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
        for child in list(borders):
            borders.remove(child)
        for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            style.prop(borders, "w:" + side, val="single" if side in ["top", "bottom"] else "nil", sz=6, color="555555")
        for ri, row in enumerate(tbl.rows):
            group = len(row._tr.findall(qn("w:tc"))) == 1
            seen = set()
            for cell in row.cells:
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                cp = cell._tc.get_or_add_tcPr()
                sh = style.prop(
                    cp,
                    "w:shd",
                    val="clear",
                    color="auto",
                    fill="F2F2F2" if ri < spec["header_rows"] or group else "FFFFFF",
                )
                cp.remove(sh)
                cp.insert_element_before(sh, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign")
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Cm(0)
                    for run in p.runs:
                        style.font(run, spec.get("font_pt", 10))
                        run.bold = ri < spec["header_rows"] or group
        # Keep each CV block together so a continued table identifies its regime.
        starts = [i for i, row in enumerate(tbl.rows) if len(row._tr.findall(qn("w:tc"))) == 1]
        for start, end in zip(starts, starts[1:] + [len(tbl.rows)]):
            for row in tbl.rows[start : end - 1]:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.keep_with_next = True
    for shape in doc.inline_shapes:
        ratio = shape.height / shape.width
        shape.width = Cm(15.3 if supplement else 16.2)
        shape.height = int(shape.width * ratio)
    for tbl in doc.tables:
        if not list(tbl._tbl.iter(qn("w:drawing"))):
            continue
        for ri, row in enumerate(tbl.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.left_indent = Cm(0)
                    p.paragraph_format.right_indent = Cm(0)
                    p.paragraph_format.keep_with_next = ri % 2 == 0
                    if ri % 2:
                        for t in p._p.iter(qn("w:t")):
                            if t.text and t.text.startswith("(a)"):
                                t.text = t.text.replace("(a)", "" if supplement else "(а)", 1).lstrip()
                            elif t.text and t.text.startswith("(b)"):
                                t.text = t.text.replace("(b)", "(б)", 1)
                        for run in p.runs:
                            style.font(run, 10.5)
    for t in doc._element.iter(qn("w:t")):
        if t.text:
            t.text = re.sub(r"(?<=\d)[ \u2000-\u200a](?=\d{3}\b)", "\u00a0", t.text)
    # Keep literal superscript +/- markers safe for LibreOffice's OMML parser.
    for run in doc._element.iter(qn("m:r")):
        t = run.find(qn("m:t"))
        if t is not None and t.text in ["+", "−"] and run.getparent().tag in [qn("m:sup"), qn("m:sub")]:
            rp = run.find(qn("m:rPr"))
            if rp is None:
                rp = OxmlElement("m:rPr")
                run.insert(0, rp)
            for x in list(rp.findall(qn("m:sty"))):
                rp.remove(x)
            if rp.find(qn("m:nor")) is None:
                rp.insert(0, OxmlElement("m:nor"))
    # Short paired formulas stay editable and fit the text width.
    layouts = []
    pairs = [
        ("λw1=", "λv1=", [8.25, 8.25]),
        ("B1=", "C1=", [8.25, 8.25]),
        ("v1=w1+", "v2=w2+", [6, 10.5]),
        ("GmRCS=", "Gmобщ=", [8.25, 8.25]),
    ]

    def mathtext(p):
        return "".join(x.text or "" for x in p._p.iter(qn("m:t"))).replace(" ", "")

    for first, second, widths in pairs:
        ps = list(doc.paragraphs)
        for a, b in zip(ps, ps[1:]):
            if mathtext(a).startswith(first) and mathtext(b).startswith(second):
                compact.equation_row(doc, [a, b], widths, first, layouts)
                break
    blocks = [b for b in doc._element.body if b.tag in [qn("w:p"), qn("w:tbl")]]
    continuations = []
    for before, current in zip(blocks, blocks[1:]):
        before_math = bool(list(before.iter(qn("m:oMathPara"))))
        current_math = bool(list(current.iter(qn("m:oMathPara"))))
        if before.tag == qn("w:p") and not before_math and current_math:
            Paragraph(before, doc._body).paragraph_format.keep_with_next = True
        if current.tag == qn("w:p") and before_math and not current_math:
            p = Paragraph(current, doc._body)
            if p.style.name == "First Paragraph":
                p.paragraph_format.first_line_indent = Cm(0)
                continuations.append(p.text)
    for sec in doc.sections:
        sec.top_margin = Cm(2)
        sec.bottom_margin = Cm(2)
        sec.left_margin = Cm(3)
        sec.right_margin = Cm(1.5)
    if supplement:
        doc.core_properties.title = "Дополнительные материалы к статье о СМО с отрицательными заявками"
    doc.save(path)
    (build.CHECKS / "layout.json").write_text(
        json.dumps(
            dict(continuations_without_indent=continuations, paired_equations=layouts), ensure_ascii=False, indent=2
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", action="store_true")
    main(parser.parse_args().supplement)
