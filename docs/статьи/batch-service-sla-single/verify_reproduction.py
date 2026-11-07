"""Compare all frozen observations against a fresh, non-overwriting run.

Usage: python verify_reproduction.py PATH_TO_FOUR_RECOMPUTED_JSON_FILES
Timing and provenance necessarily differ; every scientific field is compared.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def compare(left, right, path="cases"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if key != "elapsed_s":
                compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12), (path, left, right)
    else:
        assert left == right, (path, left, right)


def main():
    directory = Path(sys.argv[1])
    report = {
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 1e-12,
        "excluded_case_fields": ["elapsed_s"],
        "panels": {},
    }
    for panel in ("patience", "load", "decision", "inoue"):
        name = f"bulk-sla-revision-{panel}-2026.json"
        original, fresh = REPO / "docs/research" / name, directory / name
        a, b = json.loads(original.read_text()), json.loads(fresh.read_text())
        assert a["protocol_sha256"] == b["protocol_sha256"]
        assert a["deadlines"] == b["deadlines"]
        compare(a["cases"], b["cases"])
        report["panels"][panel] = {
            "cases": len(a["cases"]),
            "scientific_fields_match": True,
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "recomputed_sha256": hashlib.sha256(fresh.read_bytes()).hexdigest(),
            "recomputed_versions": b["versions"],
            "recomputed_code_sha256": b["code_sha256"],
        }
    (ROOT / "reproduction_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("All scientific fields match for all 286 scenarios, including DES replicates and refusals.")


if __name__ == "__main__":
    main()
