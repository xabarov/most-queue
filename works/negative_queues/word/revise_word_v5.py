"""Remove paragraph indents from reviewed continuations after displayed formulas.

Run with: uv run --no-project --with lxml python revise_word_v5.py
The delivered v4 remains unchanged; only w:ind changes in the v5 document.
"""

import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "negative_queues_editable_v4.docx"
OUT = ROOT / "negative_queues_editable_v5.docx"
CHECKS = ROOT / "verification_v5"
BASE_SHA256 = "06ef9792d83a251a63faea20e520a686b10f9ff0d832c48e55dc12c9893882ff"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

# Each continuation was checked against the argument and paragraph boundaries
# in revision_20261107/article_body.tex. A display alone does not imply that the
# next paragraph should lose its indent. Headings and independent paragraphs
# (including the paragraph after the list in section 2) retain their formatting.
CONTINUATIONS = {
    "При yℓ≥0": "Conditions and interpretation of the service transform.",
    "При i≥n": "Second case of the arrival-transition rule.",
    "При i>n": "Second case of the service-completion rule.",
    "Элементы с индексами": "Index limits for the preceding transition block.",
    "Например, при": "Example of the transition blocks just defined.",
    "Итерации Такахаси": "Solution of the preceding balance equations.",
    "При бесконечной сумме": "Exactness conditions for the preceding moment formula.",
    "где учитываются": "Continuation of the sentence introducing Little's law.",
    "Для непустых уровней": "Remaining blocks of the same RCS model.",
    "Для i≥2": "Remaining levels of the auxiliary-state construction.",
    "Это соответствует": "Interpretation of the physical-probability normalization.",
    "Пусть P": "Transition probabilities in the same waiting-time derivation.",
    "Вектор a здесь": "Notation and multiplication order in the waiting transform.",
    "В расчёте принимается": "Choice of the parameter in the preceding transform.",
    "Для n>1": "Scope and limits of the preceding sojourn approximation.",
    "Величина Y": "Independence and transforms of the variables just introduced.",
    "Следовательно,": "Link between successive formulas in one derivation.",
    "Вероятность завершения": "Conditional transforms derived from the same model.",
    "Индексы обозначают": "Explanation of the preceding transform subscripts.",
    "При r=0": "Zero-rate case and limits of the effective-service formula.",
    "где распределение": "Continuation of the sentence defining the closure.",
    "Поэтому при": "Averaging the preceding conditional moments.",
    "Для вещественного H2": "Existence conditions for the preceding moments.",
}


def qn(name):
    prefix, local = name.split(":")
    return "{" + NS[prefix] + "}" + local


def text(node):
    return "".join(node.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


def canonical(node):
    return E.tostring(node, method="c14n", exclusive=True)


def main():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA256
    with ZipFile(BASE) as src:
        original = E.fromstring(src.read("word/document.xml"))
        revised = copy.deepcopy(original)
        body = revised.find("w:body", NS)
        blocks = [b for b in body if b.tag in (qn("w:p"), qn("w:tbl"))]
        seen = set()
        changes = []
        for previous, paragraph in zip(blocks, blocks[1:]):
            if paragraph.tag != qn("w:p"):
                continue
            if not previous.xpath(".//m:oMathPara", namespaces=NS):
                continue
            if paragraph.xpath(".//m:oMathPara", namespaces=NS):
                continue
            style = paragraph.find("w:pPr/w:pStyle", NS)
            if style is not None and style.get(qn("w:val"), "").startswith("Heading"):
                continue
            content = text(paragraph)
            matches = [prefix for prefix in CONTINUATIONS if content.startswith(prefix)]
            assert len(matches) == 1, f"Unreviewed paragraph after a display: {content}"
            prefix = matches[0]
            assert prefix not in seen, prefix
            seen.add(prefix)
            indent = paragraph.find("w:pPr/w:ind", NS)
            assert indent is not None and indent.get(qn("w:firstLine")) == "709", prefix
            old_indent = dict(indent.attrib)
            indent.set(qn("w:firstLine"), "0")
            changes.append(
                {
                    "body_index": list(body).index(paragraph),
                    "text": content,
                    "reason": CONTINUATIONS[prefix],
                    "previous_block": E.QName(previous).localname,
                    "first_line_before_twips": 709,
                    "first_line_after_twips": 0,
                    "original_indent_attributes": old_indent,
                }
            )
        assert seen == set(CONTINUATIONS), set(CONTINUATIONS) - seen

        # Revert precisely the edited attributes in a copy: every other XML
        # node, including all formulas, tables, fields and styles, must match.
        restored = copy.deepcopy(revised)
        restored_body = restored.find("w:body", NS)
        for change in changes:
            indent = restored_body[change["body_index"]].find("w:pPr/w:ind", NS)
            indent.attrib.clear()
            indent.attrib.update(change["original_indent_attributes"])
        assert canonical(restored) == canonical(original)

        with ZipFile(OUT, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "word/document.xml":
                    data = E.tostring(revised, encoding="UTF-8", xml_declaration=True, standalone=True)
                dst.writestr(copy.copy(info), data)
        with ZipFile(OUT) as dst:
            assert dst.namelist() == src.namelist()
            assert all(src.read(name) == dst.read(name) for name in src.namelist() if name != "word/document.xml")
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA256
    CHECKS.mkdir(exist_ok=True)
    report = {
        "base_document": BASE.name,
        "base_sha256": BASE_SHA256,
        "base_unchanged": True,
        "document": OUT.name,
        "docx_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "changed_paragraphs": len(changes),
        "native_equations": len(revised.xpath("//m:oMath", namespaces=NS)),
        "only_reviewed_first_line_indents_changed": True,
        "all_other_docx_parts_byte_identical": True,
        "changes": changes,
    }
    (CHECKS / "indent-checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{OUT}: corrected {len(changes)} continuations; all content unchanged.")


if __name__ == "__main__":
    main()
