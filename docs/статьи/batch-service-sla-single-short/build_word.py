"""Editable Word from the same LaTeX body, with native OMML equations.

Preprocess only cross-references, static equation/caption numbering and image
format; do not rasterize article pages. Build the concise article and its separate appendix.
Run after the LaTeX build to resolve references.
"""

import argparse
import re
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pypandoc
from lxml import etree

ROOT = Path(__file__).resolve().parent
LATEX = ROOT / "latex"
WORD = ROOT / "word"


def build(document):
    body_name = "article.tex" if document == "main" else "appendix-body.tex"
    body = (LATEX / body_name).read_text()
    table_labels = [
        re.search(r"\\label\{([^}]+)\}", match[0])[1]
        for match in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", body, re.S)
    ]
    refs = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", (LATEX / f"{document}.aux").read_text()))
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + refs[m[1]] + ")", body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: refs[m[1]], body)
    body = re.sub(r"\\input\{([^}]+)\}", lambda m: (LATEX / m[1]).read_text(), body)
    for environment, label in (("figure", "Рисунок"), ("table", "Таблица")):
        counter = 0

        def caption(match):
            nonlocal counter
            counter += 1
            return match[0].replace(r"\caption{", rf"\caption{{{label} {counter}. ", 1)

        body = re.sub(r"\\begin\{" + environment + r"\}.*?\\end\{" + environment + r"\}", caption, body, flags=re.S)
    number = 0

    def equation(match):
        nonlocal number
        content = re.sub(r"\\label\{[^}]+\}", "", match[2]).strip()
        lines = content.split(r"\\") if match[1] == "align" else [content]
        numbered = []
        for line in lines:
            if not line.strip():
                continue
            number += 1
            # Pandoc's OMML eqArr must not contain literal alignment markers.
            line = re.sub(r"&(?=\s*=)", "", line)
            line = line.replace("&", r"\qquad ")
            numbered.append(line.strip() + rf"\qquad ({number})")
        # Separate editable display equations avoid Pandoc's invalid m:count
        # output for an alignment array without alignment markers.
        return "\n" + "\n".join(r"\[" + line + r"\]" for line in numbered) + "\n"

    body = re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", equation, body, flags=re.S)
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    body = body.replace(".pdf}", ".png}").replace(r"width=\linewidth", "width=15cm")
    source = WORD / f"{document}_source.tex"
    source.write_text(body)
    target = WORD / f"{document}.docx"
    # CSL GOST does not render BibLaTeX's eid; preserve article identifiers
    # in its page field without changing the canonical shared bibliography.
    bibliography = WORD / "references.bib"
    bibliography.write_text(
        re.sub(r"(?m)^(\s*)eid(\s*=)", r"\1pages\2", (ROOT / "bibliography/library.bib").read_text())
    )
    subprocess.run(
        [
            pypandoc.get_pandoc_path(),
            str(source),
            "-f",
            "latex",
            "-t",
            "docx",
            "--standalone",
            "--citeproc",
            "--bibliography=" + str(bibliography),
            "--csl=" + str(WORD / "gost-numeric.csl"),
            "--metadata=lang:ru-RU",
            "--metadata=reference-section-title:Список литературы",
            "--resource-path=.",
            "-o",
            str(target),
        ],
        cwd=LATEX,
        check=True,
    )
    # Mechanical OOXML layout pass: A4, author margins, fonts, no clipped fixed
    # row heights, and repeated table headings. Content and OMML stay editable.
    with ZipFile(target) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    q = lambda name: f"{{{ns}}}{name}"

    def child(parent, name, **attrs):
        element = parent.find(q(name))
        if element is None:
            element = etree.Element(q(name))
            if name in ("pPr", "rPr", "trPr", "tcPr", "cantSplit") and parent.tag != q("style"):
                parent.insert(0, element)
            elif name == "keepNext":
                parent.insert(int(parent.find(q("pStyle")) is not None), element)
            else:
                parent.append(element)
        for key, value in attrs.items():
            element.set(q(key), str(value))
        return element

    document = etree.fromstring(files["word/document.xml"])
    for section in document.iter(q("sectPr")):
        size = section.find(q("pgSz"))
        if size is None:
            size = etree.SubElement(section, q("pgSz"))
        size.set(q("w"), "11906")
        size.set(q("h"), "16838")
        margin = section.find(q("pgMar"))
        if margin is not None:
            for key, value in dict(top=1417, bottom=1417, left=1701, right=1134).items():
                margin.set(q(key), str(value))
    for index, table in enumerate(document.iter(q("tbl"))):
        for row in table.findall(q("tr")):
            child(child(row, "trPr"), "cantSplit")
        rows = table.findall(q("tr"))
        child(child(rows[0], "trPr"), "tblHeader")
        for row in rows[:1] if table_labels[index] == "tab:prior" else rows[:-1]:
            for paragraph in row.iter(q("p")):
                child(child(paragraph, "pPr"), "keepNext")
        for run in table.iter(q("r")):
            child(child(run, "rPr"), "sz", val=21)
        # More room for long textual column names than equal-width defaults.
        proportions = {
            "tab:decisions": (0.17, 0.21, 0.25, 0.22, 0.15),
            "tab:choices": (0.22, 0.22, 0.22, 0.34),
            "tab:prior": (0.23, 0.37, 0.40),
            "tab:validation": (0.10, 0.08, 0.14, 0.22, 0.14, 0.22),
            "tab:diagnostics": (0.16, 0.17, 0.13, 0.28, 0.26),
            "tab:localbounds": (0.17, 0.13, 0.175, 0.175, 0.175, 0.175),
            "tab:caps": (0.12, 0.22, 0.18, 0.18, 0.30),
            "tab:inoue": (0.12, 0.12, 0.30, 0.25, 0.21),
        }[table_labels[index]]
        assert len(proportions) == len(table.find(q("tblGrid"))), table_labels[index]
        widths = [round(9071 * p) for p in proportions]
        for col, width in zip(table.find(q("tblGrid")), widths):
            col.set(q("w"), str(width))
        for row in rows:
            for cell, width in zip(row.findall(q("tc")), widths):
                child(child(cell, "tcPr"), "tcW", w=width, type="dxa")
    for paragraph in document.iter(q("p")):
        if paragraph.find(".//" + q("drawing")) is not None:
            child(child(paragraph, "pPr"), "keepNext")
        text = "".join(t.text or "" for t in paragraph.iter(q("t")))
        if text.startswith(("СВОЕВРЕМЕННОЕ НАЧАЛО", "TIMELY START OF BATCH SERVICE", "ПРИЛОЖЕНИЕ К СТАТЬЕ")):
            child(child(paragraph, "pPr"), "jc", val="center")
    content_types = etree.fromstring(files["[Content_Types].xml"])
    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    if not any(item.get("Extension") == "png" for item in content_types):
        etree.SubElement(content_types, f"{{{ct_ns}}}Default", Extension="png", ContentType="image/png")
    files["[Content_Types].xml"] = etree.tostring(
        content_types, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    files["word/document.xml"] = etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True)
    styles = etree.fromstring(files["word/styles.xml"])
    for font in styles.iter(q("rFonts")):
        for key in list(font.attrib):
            if key.lower().endswith("theme"):
                del font.attrib[key]
        font.set(q("ascii"), "PT Serif")
        font.set(q("hAnsi"), "PT Serif")
    for style in styles.findall(q("style")):
        name = style.get(q("styleId"), "")
        if name.startswith("Heading") or name in ("Title", "Subtitle"):
            color = child(child(style, "rPr"), "color")
            color.attrib.clear()
            color.set(q("val"), "000000")
            child(child(style, "rPr"), "sz", val=24)
            child(child(style, "rPr"), "b")
        if name in ("Normal", "BodyText", "FirstParagraph"):
            child(child(style, "pPr"), "spacing", before=0, after=100, line=264, lineRule="auto")
        if name in ("Caption", "TableCaption", "ImageCaption"):
            child(child(style, "rPr"), "sz", val=22)
            child(child(style, "rPr"), "i", val=0)
            child(child(style, "rPr"), "b", val=int(name == "TableCaption"))
    math_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    assert not any(t.text == "&" for t in document.iter(f"{{{math_ns}}}t"))
    files["word/styles.xml"] = etree.tostring(styles, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    print(target)
    print(
        "Numbered equations:",
        number,
        "Native maths:",
        len(document.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", choices=("main", "appendix", "all"), default="all")
    args = parser.parse_args()
    for document in ("main", "appendix") if args.document == "all" else (args.document,):
        build(document)
