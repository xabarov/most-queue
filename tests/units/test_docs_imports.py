"""
Regression guard: every ``from most_queue... import ...`` line that appears in
the model catalog (docs/models.md and docs/models.ru.md) must actually import.

This keeps the documentation and the code in sync — a class rename or a moved
module that is not reflected in the catalog fails here instead of silently
shipping broken examples.
"""

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOGS = ["docs/models.md", "docs/models.ru.md"]

_IMPORT_RE = re.compile(r"^\s*from\s+most_queue[\w.]*\s+import\s+.+$", re.MULTILINE)


def _catalog_import_lines():
    lines = []
    for rel in CATALOGS:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for match in _IMPORT_RE.finditer(text):
            lines.append((rel, match.group(0).strip()))
    return lines


IMPORT_LINES = _catalog_import_lines()


def test_catalog_has_imports():
    """Sanity: the catalogs contain import examples to check."""
    assert IMPORT_LINES, "no 'from most_queue ... import' lines found in the model catalogs"


@pytest.mark.parametrize("source, statement", IMPORT_LINES, ids=[f"{s}: {st}" for s, st in IMPORT_LINES])
def test_catalog_import_executes(source, statement):
    """Every import statement quoted in the catalog must resolve."""
    try:
        exec(statement, {})  # pylint: disable=exec-used
    except Exception as exc:  # pylint: disable=broad-exception-caught
        pytest.fail(f"{source}: `{statement}` failed to import: {exc!r}")


if __name__ == "__main__":
    for src, st in IMPORT_LINES:
        exec(st, {})  # pylint: disable=exec-used
    print(f"all {len(IMPORT_LINES)} catalog imports OK")
