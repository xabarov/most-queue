"""Build an editable Word working copy from the frozen LaTeX manuscript.

Run: uv run --no-project --with pypandoc_binary --with python-docx --with lxml python build_word.py
The manuscript, figures, numerical tables and original PDF are read only.
"""

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
CHECKS = ROOT / "verification"
STEM = "negative_queues_editable"
PANDOC = pypandoc.get_pandoc_path()
FIGURES = []
REFERENCES = []
CORRECTIONS = []
CAPTIONS = []
REFS = {
    "fig:rcs_transitions": ("fig_1", "1", "a"),
    "fig:disaster_transitions": ("fig_1", "1", "b"),
    "fig:negative_transitions": ("fig_1", "1", ""),
    "fig:v1_n": ("fig_2", "2", ""),
    "fig:v1_n_fixed": ("fig_3", "3", ""),
    "fig:v1_delta_cv_compare": ("fig_4", "4", ""),
    "fig:v1_delta_cv_compare_fixed": ("fig_5", "5", ""),
    "fig:q_delta_cv_compare": ("fig_6", "6", ""),
    "tab:results": ("tab_1", "1", ""),
    "tab:sko": ("tab_2", "2", ""),
    "tab:results_fixed": ("tab_3", "3", ""),
    "sec:times": ("sec_5", "5", ""),
    "sec:results": ("sec_6", "6", ""),
}


def argument(text, offset):
    assert text[offset] == "{"
    depth = 1
    for i in range(offset + 1, len(text)):
        if text[i] in "{}" and text[i - 1] != "\\":
            depth += 1 if text[i] == "{" else -1
            if depth == 0:
                return text[offset + 1 : i], i + 1
    raise ValueError("Unclosed TeX argument")


def caption(text, command=r"\caption"):
    start = text.index(command + "{") + len(command)
    value, end = argument(text, start)
    return value, start - len(command), end


def figure(match):
    block = match[0]
    panels = re.findall(r"\\begin\{(?:subfigure|minipage)\}.*?\\end\{(?:subfigure|minipage)\}", block, re.S)
    data = {"panels": [], "captions": []}
    for i, panel in enumerate(panels):
        path = re.search(r"\\maybeincludegraphics\[[^]]*\]\{([^}]+)\}", panel)[1]
        assert (SOURCE / path).is_file(), path
        cap = caption(panel, r"\captionof{figure}" if r"\captionof" in panel else r"\caption")[0]
        independent = r"\captionof" in panel
        if independent:
            key = re.search(r"\\label\{([^}]+)\}", panel)[1]
            anchor, number, _ = REFS[key]
            cap = rf"Рисунок SEQX{anchor}X. " + cap
            CAPTIONS.append((anchor, number, "Figure"))
        else:
            cap = f"({chr(97+i)}) " + cap
        data["panels"].append({"path": path, "caption": cap})
    rest = block
    for panel in panels:
        rest = rest.replace(panel, "")
    if r"\caption{" in rest:
        cap = caption(rest)[0]
        key = re.search(r"\\label\{([^}]+)\}", rest)[1]
        anchor, number, _ = REFS[key]
        data["main_caption"] = rf"Рисунок SEQX{anchor}X. " + cap
        CAPTIONS.append((anchor, number, "Figure"))
    data["columns"] = 1 if len(FIGURES) == 0 else 2
    group = len(FIGURES)
    FIGURES.append(data)
    out = [f"\\noindent FIGBLOCKX{group}X\\par"]
    for i, panel in enumerate(data["panels"]):
        out.append(f"\\noindent PANELCAPX{group}X{i}X " + panel["caption"] + r"\par")
    if "main_caption" in data:
        out.append(f"\\noindent MAINCAPX{group}X " + data["main_caption"] + r"\par")
    return "\n\n".join(out)


def table(match):
    block = match[0]
    cap, start, end = caption(block)
    key = re.search(r"\\label\{([^}]+)\}", block)[1]
    anchor, number, _ = REFS[key]
    CAPTIONS.append((anchor, number, "Table"))
    block = block[:start] + block[end:]
    block = re.sub(r"\\begin\{table\}(?:\[[^]]*\])?|\\end\{table\}|\\centering|\\small", "", block)
    block = block.replace("@{}", "")
    block = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", block)
    return rf"\noindent TABLECAPX Рисунок\par".replace("Рисунок", rf"Таблица SEQX{anchor}X. " + cap) + "\n" + block


def xref(match):
    value = REFS[match[1]]
    REFERENCES.append(value)
    return f"XREFX{len(REFERENCES)-1}X"


def preprocess():
    text = (SOURCE / "negative_queues_takahasi_takami.tex").read_text()
    text = text.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = text.replace(r"\input{generated_tables_fixed.tex}", (SOURCE / "generated_tables_fixed.tex").read_text())
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    # Correct stale literal section pointers after comparing their named targets.
    for old, new in [(r"раздел~4", r"раздел~\ref{sec:times}"), (r"раздел~5", r"раздел~\ref{sec:results}")]:
        CORRECTIONS.append({"original": old, "replacement": new, "occurrences": text.count(old)})
        text = text.replace(old, new)
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", figure, text, flags=re.S)
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", table, text, flags=re.S)
    text = re.sub(r"\\ref\{([^}]+)\}", xref, text)
    # Caption text is inserted above before references are expanded.
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = text.replace(r"\begin{samepage}", "").replace(r"\end{samepage}", "")
    text = text.replace(r"\begin{abstract}", r"\noindent\textbf{Аннотация.} ").replace(r"\end{abstract}", "")
    text = text.replace(r"\printbibliography", "")
    # Explicit grouping avoids two LibreOffice OMML import ambiguities.
    for old, new in [
        (
            r"\max_i|x[i]^{\mathrm{new}}-x[i]^{\mathrm{old}}|<\varepsilon",
            r"\max_i\left|{x[i]}^{\mathrm{new}}-{x[i]}^{\mathrm{old}}\right|<\varepsilon",
        ),
        (r"T^{* (k)}(0)", r"\left(T^*\right)^{(k)}(0)"),
    ]:
        CORRECTIONS.append(
            {
                "original": old,
                "replacement": new,
                "occurrences": text.count(old),
                "reason": "Equivalent mathematical grouping for Word/LibreOffice",
            }
        )
        text = text.replace(old, new)
    text = re.sub(r"\\setlength\{\\arraycolsep\}\{[^}]+\}", "", text)
    text = re.sub(r"\{\\scriptsize\s*(.*?)\n\}", lambda m: "\n\n" + m[1] + "\n\n", text, flags=re.S)

    # Outer arrays only align independent matrices. Keep each inner matrix native.
    def matrices(m):
        parts = re.findall(r"[ABCD]_\d+=\\begin\{pmatrix\}.*?\\end\{pmatrix\}[.,]?", m[1], re.S)
        assert len(parts) in (2, 3)
        return "\n\n".join(r"\[" + part + r"\]" for part in parts)

    text = re.sub(r"\\\[\s*\\begin\{array\}(.*?)\\end\{array\}\s*\\\]", matrices, text, flags=re.S)
    text = text.replace("можно компактно записать на одном уровне:", "можно записать в следующем виде:")

    # Use separate native display paragraphs for long independent equalities.
    def display(m):
        return "\n\n" + "\n\n".join(r"\[" + part.strip() + r"\]" for part in m[1].split(r"\qquad")) + "\n\n"

    text = re.sub(r"(?<!\\)\\\[(.*?)\\\]", display, text, flags=re.S)

    def align(m):
        return (
            "\n\n"
            + "\n\n".join(r"\[" + part.replace("&", "").strip() + r"\]" for part in m[1].split(r"\\") if part.strip())
            + "\n\n"
        )

    text = re.sub(r"\\begin\{align\*\}(.*?)\\end\{align\*\}", align, text, flags=re.S)
    # Explicit headings match the source PDF and remain normal Word headings.
    counters = [0, 0, 0]

    def heading(m):
        level = {"section": 0, "subsection": 1, "subsubsection": 2}[m[1]]
        counters[level] += 1
        for i in range(level + 1, 3):
            counters[i] = 0
        return "\\" + m[1] + "*{" + ".".join(map(str, counters[: level + 1])) + ". "

    text = re.sub(r"\\(section|subsection|subsubsection)\{", heading, text)
    (ROOT / f"{STEM}_source.tex").write_text(text)
    return text


def element(tag, **attrs):
    node = OxmlElement(tag)
    for key, value in attrs.items():
        node.set(qn("w:" + key), str(value))
    return node


def prop(parent, name, **attrs):
    node = parent.find(qn("w:" + name))
    if node is None:
        node = element("w:" + name)
        parent.append(node)
    for key, value in attrs.items():
        node.set(qn("w:" + key), str(value))
    return node


def pformat(p, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True):
    f = p.paragraph_format
    f.alignment = align
    f.first_line_indent = Cm(1.25 if indent else 0)
    f.space_before = Pt(0)
    f.space_after = Pt(4)
    f.line_spacing = 1.12
    f.widow_control = True
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)


def remove_token(p, token):
    for t in p._p.iter(qn("w:t")):
        if token in (t.text or ""):
            t.text = t.text.replace(token, "", 1).lstrip()
            return
    raise ValueError(token)


def make_figures(doc):
    paragraphs = list(doc.paragraphs)
    for g, data in enumerate(FIGURES):
        anchor = next(p for p in paragraphs if p.text == f"FIGBLOCKX{g}X")
        cols = data["columns"]
        rows = (len(data["panels"]) + cols - 1) // cols
        tbl = doc.add_table(rows=rows * 2, cols=cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        width = Cm(16.5 / cols)
        for col in tbl.columns:
            col.width = width
        for tr in tbl.rows:
            for cell in tr.cells:
                cell.width = width
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        borders = prop(tbl._tbl.tblPr, "tblBorders")
        tbl._tbl.tblPr.remove(borders)
        tbl._tbl.tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            prop(borders, side, val="nil")
        for i, panel in enumerate(data["panels"]):
            r, c = divmod(i, cols)
            pic = tbl.cell(2 * r, c).paragraphs[0]
            pformat(pic, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            pic.paragraph_format.keep_with_next = True
            pic.add_run().add_picture(str(SOURCE / panel["path"]), width=Cm(13.5 if cols == 1 else 7.9))
            cap = next(p for p in paragraphs if p.text.startswith(f"PANELCAPX{g}X{i}X"))
            remove_token(cap, f"PANELCAPX{g}X{i}X")
            pformat(cap, 10.5, WD_ALIGN_PARAGRAPH.CENTER, False)
            cap.paragraph_format.keep_together = True
            cap.paragraph_format.keep_with_next = i < len(data["panels"]) - cols or "main_caption" in data
            cell = tbl.cell(2 * r + 1, c)
            cell._tc.remove(cell.paragraphs[0]._p)
            cell._tc.append(cap._p)
        for row in tbl.rows:
            prop(row._tr.get_or_add_trPr(), "cantSplit")
        anchor._p.addprevious(tbl._tbl)
        anchor._p.getparent().remove(anchor._p)
        if "main_caption" in data:
            cap = next(p for p in paragraphs if p.text.startswith(f"MAINCAPX{g}X"))
            remove_token(cap, f"MAINCAPX{g}X")
            pformat(cap, 10.5, WD_ALIGN_PARAGRAPH.CENTER, False)
            cap.paragraph_format.keep_together = True


def make_data_tables(doc):
    widths = [Cm(5.1)] + [Cm(1.9)] * 6
    assert len(doc.tables) == 3
    for tbl in doc.tables:
        tbl.autofit = False
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for col, width in zip(tbl.columns, widths):
            col.width = width
        for ri, row in enumerate(tbl.rows):
            prop(row._tr.get_or_add_trPr(), "cantSplit")
            if ri < 2:
                prop(row._tr.get_or_add_trPr(), "tblHeader")
            column = 0
            for cell in row._tr.findall(qn("w:tc")):
                pr = cell.find(qn("w:tcPr"))
                span = pr.find(qn("w:gridSpan"))
                count = int(span.get(qn("w:val"))) if span is not None else 1
                tcw = prop(pr, "tcW", w=sum(width.twips for width in widths[column : column + count]), type="dxa")
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
                column += count
            for ci, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for p in cell.paragraphs:
                    pformat(p, 10, WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER, False)
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(3)
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.keep_with_next = ri < len(tbl.rows) - 1


def field(instruction, value):
    node = element("w:fldSimple", instr=instruction)
    r = element("w:r")
    t = element("w:t")
    t.text = value
    r.append(t)
    node.append(r)
    return node


def replace_tokens(root, pattern, builder):
    for t in list(root.iter(qn("w:t"))):
        text = t.text or ""
        if not re.search(pattern, text):
            continue
        run = t.getparent()
        assert run.tag == qn("w:r") and len([x for x in run if x.tag != qn("w:rPr")]) == 1
        parent = run.getparent()
        at = parent.index(run)
        parts = []
        offset = 0
        for m in re.finditer(pattern, text):
            if m.start() > offset:
                r = copy.deepcopy(run)
                rt = r.find(qn("w:t"))
                rt.text = text[offset : m.start()]
                rt.set(qn("xml:space"), "preserve")
                parts.append(r)
            replacement = builder(m)
            rpr = run.find(qn("w:rPr"))
            if rpr is not None:
                for node in replacement:
                    for child in node.iter(qn("w:r")):
                        if child.find(qn("w:rPr")) is None:
                            child.insert(0, copy.deepcopy(rpr))
            parts.extend(replacement)
            offset = m.end()
        if offset < len(text):
            r = copy.deepcopy(run)
            rt = r.find(qn("w:t"))
            rt.text = text[offset:]
            rt.set(qn("xml:space"), "preserve")
            parts.append(r)
        parent.remove(run)
        for i, node in enumerate(parts):
            parent.insert(at + i, node)


def finish(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2)
    sec.bottom_margin = Cm(2)
    sec.left_margin = Cm(3)
    sec.right_margin = Cm(1.5)
    sec.header_distance = Cm(1)
    sec.gutter = Cm(0)
    sec.footer_distance = Cm(1)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    for style in doc.styles:
        if style.type == 1:
            style.font.name = "Times New Roman"
            style.font.color.rgb = __import__("docx").shared.RGBColor(0, 0, 0)
    front = True
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            front = False
        pformat(p, size=11 if front else 12)
        if front:
            p.paragraph_format.first_line_indent = Cm(0)
            if p.text.startswith(("Расчёт многоканальных", "Calculation of")):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(15 if p.text.startswith("Расчёт") else 12)
            if "E-mail:" in p.text:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if p.style.name.startswith("Heading"):
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.keep_together = True
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(5)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if p.style.name == "Heading 1" else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(13 if p.style.name == "Heading 1" else 12)
        if p._p.find(qn("m:oMathPara")) is not None:
            pformat(p, 12, WD_ALIGN_PARAGRAPH.CENTER, False)
            p.paragraph_format.keep_together = True
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(5)
        if p.text.startswith("TABLECAPX"):
            remove_token(p, "TABLECAPX")
            pformat(p, 10.5, WD_ALIGN_PARAGRAPH.LEFT, False)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.keep_together = True
        if p.style.name == "Bibliography":
            pformat(p, 11, WD_ALIGN_PARAGRAPH.LEFT, False)
            p.paragraph_format.keep_together = True
    make_data_tables(doc)
    make_figures(doc)
    max_id = max((int(x.get(qn("w:id"))) for x in doc._element.iter(qn("w:bookmarkStart"))), default=0)
    counter = max_id + 1
    info = {a: (number, seq) for a, number, seq in CAPTIONS}

    def seq(m):
        nonlocal counter
        name = m[1]
        value, typ = info[name]
        ident = counter
        counter += 1
        return [
            element("w:bookmarkStart", id=ident, name=name),
            field(f"SEQ {typ} \\* ARABIC", value),
            element("w:bookmarkEnd", id=ident),
        ]

    replace_tokens(doc._element, r"SEQX([a-z]+_\d+)X", seq)
    for p in doc.paragraphs:
        if p.style.name == "Heading 1" and re.match(r"[56]\. ", p.text):
            number = p.text[0]
            ident = counter
            counter += 1

            def section(m):
                r = element("w:r")
                t = element("w:t")
                t.text = number
                r.append(t)
                return [
                    element("w:bookmarkStart", id=ident, name=f"sec_{number}"),
                    r,
                    element("w:bookmarkEnd", id=ident),
                ]

            replace_tokens(p._p, r"^[56](?=\. )", section)

    def ref(m):
        anchor, value, suffix = REFERENCES[int(m[1])]
        nodes = [field(f"REF {anchor} \\h", value)]
        if suffix:
            r = element("w:r")
            t = element("w:t")
            t.text = suffix
            r.append(t)
            nodes.append(r)
        return nodes

    replace_tokens(doc._element, r"XREFX(\d+)X", ref)
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(field("PAGE", "1"))
    update = prop(doc.settings.element, "updateFields", val="true")
    doc.settings.element.remove(update)
    doc.settings.element.insert_element_before(
        update, "w:compat", "w:docVars", "w:rsids", "m:mathPr", "w:themeFontLang"
    )
    # Pandoc emits matrix-column properties in reverse XSD order.
    for mcpr in doc._element.iter(qn("m:mcPr")):
        count = mcpr.find(qn("m:count"))
        if count is not None:
            mcpr.remove(count)
            mcpr.insert(0, count)
    # A superscript asterisk is a transform marker, not a binary operation.
    # Mark it literal so LibreOffice does not parse it as incomplete multiplication.
    literal_runs = []
    for mr in doc._element.iter(qn("m:r")):
        t = mr.find(qn("m:t"))
        if t is not None and t.text == "*":
            literal_runs.append(mr)
    for math in doc._element.iter(qn("m:oMath")):
        if len(math) and math[0].tag == qn("m:r"):
            t = math[0].find(qn("m:t"))
            if t is not None and t.text in ("<", ">"):
                literal_runs.append(math[0])
    for mr in literal_runs:
        rpr = mr.find(qn("m:rPr"))
        if rpr is None:
            rpr = OxmlElement("m:rPr")
            mr.insert(0, rpr)
        if rpr.find(qn("m:nor")) is None:
            rpr.insert(0, OxmlElement("m:nor"))
    # Normal text and mathematical styling are mutually exclusive in OMML.
    for rpr in doc._element.iter(qn("m:rPr")):
        if rpr.find(qn("m:nor")) is not None:
            for sty in list(rpr.findall(qn("m:sty"))):
                rpr.remove(sty)
    lang = prop(doc.styles["Normal"].element.get_or_add_rPr(), "lang", val="ru-RU")
    doc.core_properties.title = "Расчёт многоканальных СМО с отрицательными заявками"
    doc.core_properties.author = "Р. С. Хабаров; В. А. Лохвицкий"
    doc.core_properties.subject = "Редактируемая рабочая версия статьи"
    doc.save(ROOT / f"{STEM}.docx")


def main():
    CHECKS.mkdir(exist_ok=True)
    preprocess()
    args = [
        PANDOC,
        str(ROOT / f"{STEM}_source.tex"),
        "-f",
        "latex",
        "--standalone",
        "--citeproc",
        "--bibliography=" + str(SOURCE / "citations.bib"),
        "--csl=" + str(ROOT / "gost-numeric.csl"),
        "--metadata=lang:ru-RU",
        "--metadata=reference-section-title:Список литературы",
        "--metadata=link-citations:true",
    ]
    native = subprocess.run(args + ["-t", "json"], cwd=SOURCE, check=True, capture_output=True, text=True)
    (CHECKS / "document-ast.json").write_text(native.stdout)
    result = subprocess.run(
        args + ["-t", "docx", "-o", str(ROOT / f"{STEM}.docx")], cwd=SOURCE, check=True, capture_output=True, text=True
    )
    (CHECKS / "pandoc.log").write_text(native.stderr + result.stderr)
    if "Could not convert" in result.stderr:
        raise RuntimeError(result.stderr)
    doc = Document(ROOT / f"{STEM}.docx")
    finish(doc)
    (CHECKS / "conversion-changes.json").write_text(
        json.dumps(
            {
                "section_references": CORRECTIONS,
                "math_layout": "Long displays and outer arrays split into readable native display paragraphs; mathematical expressions unchanged.",
                "figure_panels": sum(len(f["panels"]) for f in FIGURES),
                "captions": CAPTIONS,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(ROOT / f"{STEM}.docx")


if __name__ == "__main__":
    main()
