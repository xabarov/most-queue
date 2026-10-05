"""Independent EPIC-065 prefix-lag/donor-pool audit; writes nothing.

Reuses the already-independent EPIC-059 raw parser (works/feature_service/audit.py)
for job/submit-range reconstruction; does not import this epic's own
availability_prefix or prepare(). Cross-checks protocol.json's reported
prefix-lag/unresolved/donor-pool-size numbers directly from the cached raw
sources, and the stale completed-prefix numbers against the same source of
truth EPIC-062's own audit already uses.
"""

import argparse
import hashlib
import json
from pathlib import Path

from works.feature_service import audit as base


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/real_trace"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/availability_aware_arrivals"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    for filename, expected in manifest["implementation_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((args.output_dir / artifact["file"]).read_bytes()).hexdigest() == artifact["sha256"]
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    checked = {}
    for name, filename in (("sdsc", "SDSC-SP2-1998-4.2-cln.swf"), ("kalos", "acme-kalos.csv")):
        path = args.cache_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sources"][name]["sha256"]
        jobs, _ = base.load_source(path, name)
        jobs.sort(key=lambda j: j["submit"])
        for block in protocol["blocks"][name]:
            cutoff = block["split"]["cutoff"]
            prefix = [j for j in jobs if j["submit"] < cutoff]
            assert len(prefix) >= 2 and all(a["submit"] <= b["submit"] for a, b in zip(prefix, prefix[1:]))
            unresolved = sum(1 for j in prefix if base.end(j) >= cutoff)
            lag = cutoff - prefix[-1]["submit"]
            reported = block["prefix_audit_availability"]
            assert len(prefix) == reported["prefix_jobs"], (name, block["fraction"], len(prefix), reported)
            assert unresolved == reported["unresolved_at_cutoff"]
            assert abs(lag - reported["prefix_lag"]) < 1e-6
            # The old (stale) construction never uses a job past the first
            # one still open at cutoff; it can never see more donors than the
            # availability-aware prefix built from the exact same cutoff.
            assert block["prefix_audit_stale"]["prefix_jobs"] <= len(prefix)
            assert block["prefix_audit_stale"]["prefix_lag"] >= lag - 1e-6
            checked[f"{name}-{block['fraction']}"] = {
                "prefix_jobs": len(prefix),
                "unresolved_at_cutoff": unresolved,
                "prefix_lag": lag,
            }
    receipt = {
        "checked": checked,
        "primary_protocol_sha256": digest(protocol),
        "verification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Independent re-derivation of prefix-lag/donor-pool-size numbers from raw sources; "
        "no replay, no reimplementation of availability_prefix or the scheduler.",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
