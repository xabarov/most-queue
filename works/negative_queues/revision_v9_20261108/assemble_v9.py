"""v9: assemble the LaTeX master from the v8 wrapper, the v9 body and generated tables."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V8 = ROOT.parent / "revision_v8_20261107" / "negative_queues_takahasi_takami.tex"


def main():
    wrapper = V8.read_text()
    head = wrapper[: wrapper.index(r"\begin{abstract}")]
    body = (ROOT / "article_body_v9.tex").read_text()
    tables = (ROOT / "tables_v9.tex").read_text()
    blocks = dict(re.findall(r"% (tab:[a-z_]+)\n\n(\\begin\{table\}.*?\\end\{table\})", tables, re.S))
    for label, tex in blocks.items():
        body = re.sub(rf"%%TABLE {re.escape(label)}(?![a-z_])", lambda _: tex, body)
    assert "%%TABLE" not in body, "unresolved table placeholder"
    (ROOT / "negative_queues_takahasi_takami.tex").write_text(head + body + "\n\\printbibliography\n\\end{document}\n")
    refs = {
        "tab:methods": ["tab_1", "1", ""], "tab:results": ["tab_2", "2", ""], "tab:results_fixed": ["tab_3", "3", ""],
        "tab:sd": ["tab_4", "4", ""], "tab:law": ["tab_5", "5", ""],
        "fig:rcs_transitions": ["fig_1", "1", "а"], "fig:disaster_transitions": ["fig_1", "1", "б"],
        "fig:negative_transitions": ["fig_1", "1", ""], "fig:cv": ["fig_2", "2", ""], "fig:cv_occupation": ["fig_2", "2", "а"], "fig:cv_waiting": ["fig_2", "2", "б"], "sec:times": ["sec_4", "4", ""], "sec:results": ["sec_5", "5", ""],
    }
    (ROOT / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2) + "\n")
    used = set(re.findall(r"\\ref\{([^}]+)\}", body))
    assert used <= set(refs), used - set(refs)
    print("assembled; refs used:", sorted(used))


if __name__ == "__main__":
    main()
