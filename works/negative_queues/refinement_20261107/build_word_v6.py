"""Build the refined manuscript while retaining v5's paragraph convention."""

import importlib.util
import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent
WORD = ROOT.parent / "word"


def main():
    spec = importlib.util.spec_from_file_location("layout", ROOT.parent / "revision_20261107/build_word_v4.py")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    layout.ROOT = ROOT
    layout.build.SOURCE = ROOT
    layout.build.STEM = "negative_queues_editable_v6"
    layout.build.CHECKS = WORD / "verification_v6"
    layout.main()
    path = WORD / (layout.build.STEM + ".docx")
    doc = Document(path)
    # The two single-channel phase changes fit side by side at 12 pt.
    ps = list(doc.paragraphs)
    for a, b in zip(ps, ps[1:]):
        at = "".join(x.text or "" for x in a._p.iter(qn("m:t"))).replace(" ", "")
        bt = "".join(x.text or "" for x in b._p.iter(qn("m:t"))).replace(" ", "")
        if at.startswith("Ci(j,j+1)") and bt.startswith("Ci(j,j−1)"):
            layout.compact.equation_row(doc, [a, b], [8.25, 8.25], "RCS phase changes", [])
            break
    # Pandoc's FirstParagraph marks continuation after a display in this source.
    # An actual blank-line paragraph has BodyText and retains the first-line indent.
    blocks = [x for x in doc._element.body if x.tag in (qn("w:p"), qn("w:tbl"))]
    changes = []
    for before, current in zip(blocks, blocks[1:]):
        # Keep a formula's introductory prose with the following display.
        if before.tag == qn("w:p") and not list(before.iter(qn("m:oMathPara"))):
            if list(current.iter(qn("m:oMathPara"))):
                Paragraph(before, doc._body).paragraph_format.keep_with_next = True
        if current.tag != qn("w:p") or not list(before.iter(qn("m:oMathPara"))):
            continue
        if list(current.iter(qn("m:oMathPara"))):
            continue
        style = current.find("w:pPr/w:pStyle", current.nsmap)
        if style is None or style.get(qn("w:val")) != "FirstParagraph":
            continue
        indent = current.find("w:pPr/w:ind", current.nsmap)
        indent.set(qn("w:firstLine"), "0")
        changes.append("".join(t.text or "" for t in current.iter(qn("w:t"))))
    # Use the full 16.5 cm text width to spell out the restart semantics.
    widths = [4.4, 1.7, 3.0, 1.35, 1.7, 3.0, 1.35]
    for table in doc.tables:
        if len(table.columns) != 7:
            continue
        for col, width in zip(table.columns, widths):
            col.width = Cm(width)
        for row in table.rows:
            position = 0
            for cell in row._tr.findall(qn("w:tc")):
                prop = cell.find(qn("w:tcPr"))
                span = prop.find(qn("w:gridSpan"))
                count = int(span.get(qn("w:val"))) if span is not None else 1
                prop.find(qn("w:tcW")).set(
                    qn("w:w"), str(sum(Cm(x).twips for x in widths[position : position + count]))
                )
                position += count
    doc.save(path)
    (layout.build.CHECKS / "paragraphs.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    print("Formula continuations without indent:", len(changes))


if __name__ == "__main__":
    main()
