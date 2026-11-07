"""Build a whitelisted, checksum-verified reproducibility archive.

No dissertation, third-party paper PDFs, historical artifacts or credentials
are included. Regeneration only overwrites this revision's generated bundle.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    paths = list((REPO / "most_queue").rglob("*.py"))
    paths += [
        REPO / name
        for name in (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "examples/bulk_sla_revision.py",
            "examples/batch_service_sla_exact_tail_comparison.py",
            "examples/batch_service_optimal_bmax.py",
            "tests/units/test_bulk_service_revision.py",
        )
    ]
    init = REPO / "examples/__init__.py"
    if init.exists():
        paths.append(init)
    paths += list((REPO / "docs/research").glob("bulk-sla-revision*-2026.*"))
    paths += [
        ROOT / name
        for name in (
            "build_revision.py",
            "build_word.py",
            "build_supplement.py",
            "test_revision_export.py",
            "supplement_README.md",
            "verify_reproduction.py",
            "revision_results.json",
            "reproduction_verification.json",
            "revision_plan.md",
            "revision_audit.md",
            "style_revision.md",
            "results_revision.md",
            "corpus_revision.md",
            "verification/style-invariants.json",
            "verification/results-edit-invariants.json",
            "verification/corpus-edit-invariants.json",
            "verification/mechenaya-invariants.json",
            "verification/corpus-edit.diff",
            "experiment_protocol.md",
            "latex/main.tex",
            "latex/article.tex",
            "latex/main.pdf",
            "word/gost-numeric.csl",
            "word/main_v3_editable.docx",
        )
    ]
    paths += list((ROOT / "figures").glob("revision_*.*"))
    paths += list((ROOT / "latex/tables").glob("*.tex"))
    paths += list((ROOT / "verification/recomputed").glob("*.json"))
    paths += list((ROOT / "verification").glob("*.log"))
    paths.append(ROOT / "verification/word-render.pdf")
    paths.append(ROOT.parent / "bibliography/library.bib")
    content = {p.relative_to(REPO).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    content["REPRODUCE.md"] = (ROOT / "supplement_README.md").read_bytes()
    dependencies = ("numpy", "scipy", "matplotlib", "pytest")
    content["requirements-reproduction.txt"] = (
        "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in dependencies) + "\n"
    ).encode()
    manifest = {
        "date": "2026-10-21",
        "experiment_date": "2026-10-12",
        "starting_public_commit": "193397236dcb6e991a8c92d431c9384a4666e87f",
        "scope": "Editorial revision of 2026-10-21; numerical results and original observation hashes retained.",
        "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(content.items())},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    (ROOT / "supplement_manifest.json").write_bytes(manifest_bytes)
    content["manifest.json"] = manifest_bytes
    target = ROOT / "supplementary.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            entry = ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, data)
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, entry in manifest["files"].items():
            assert digest(archive.read(name)) == entry["sha256"], name
    (ROOT / "supplementary.sha256").write_text(digest(target.read_bytes()) + "  supplementary.zip\n")
    print(f"{len(content)} files; {target.stat().st_size} bytes; SHA-256 {digest(target.read_bytes())}")


if __name__ == "__main__":
    main()
