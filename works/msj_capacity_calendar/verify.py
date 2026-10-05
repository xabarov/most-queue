"""Independent pandas re-derivation of the EPIC-064 Helios scenario inputs.

Does not import examples.msj_capacity_calendar_experiment. Re-parses the same
cached archive and re-derives, per scenario: the job count, the needs set and
the three calendars' breakpoint capacities. Does not re-run the scheduler
(that dispatch logic is covered by tests/units/test_msj_capacity_calendar.py).
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TERMINAL = ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL")
DAY = 86400


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/resource_observability"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/msj_capacity_calendar"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    payload = (args.cache_dir / "helios-data.zip").read_bytes()
    assert sha(payload) == manifest["archive_sha256"]
    for item in manifest["inputs"]:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            raw = archive.read(item["file"])
        assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
    content = (args.output_dir / manifest["output"]).read_bytes()
    assert sha(content) == manifest["output_sha256"]
    result = json.loads(content)
    cluster = result["cluster"]

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        frame = pd.read_csv(archive.open(f"data/{cluster}/cluster_log.csv"))
        caps = pd.read_csv(archive.open(f"data/{cluster}/cluster_gpu_number.csv"))
    # Ordinal days (proleptic Gregorian, matching examples/resource_observability_audit.py's
    # `timestamp()`/`read_capacities`), NOT pandas' unix-epoch day numbering.
    caps.index = pd.to_datetime(caps.pop("date"), format="%Y-%m-%d").map(lambda ts: ts.toordinal())

    begin_day, end_day = result["window_days"]
    window = list(range(begin_day, end_day + 1))
    assert list(caps.index[(caps.index >= begin_day) & (caps.index <= end_day)]) == window
    begin, end = window[0] * DAY, (window[-1] + 1) * DAY
    assert [begin, end] == result["window_seconds"]

    for kind in ("submit", "start", "end"):
        parsed = pd.to_datetime(frame[f"{kind}_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
        ordinal_days = parsed.dt.normalize().map(lambda ts: ts.toordinal())
        seconds_of_day = (parsed - parsed.dt.normalize()).dt.total_seconds()
        frame[kind] = (ordinal_days * DAY + seconds_of_day).astype(np.int64)
    gpu = frame.gpu_num > 0
    closed = frame.submit <= frame.start
    closed &= frame.start < frame.end
    terminal = frame.state.isin(TERMINAL)
    windowed = (frame.submit >= begin) & (frame.submit < end)
    valid = gpu & closed & terminal & windowed
    selected = frame[valid]
    assert len(selected) == result["window_job_count"]

    vc_counts = selected.vc.value_counts()
    known_vcs = set(caps.columns) - {"total"}
    vc_counts = vc_counts[vc_counts.index.isin(known_vcs)]
    largest_vc = vc_counts.sort_index().idxmax()
    assert largest_vc == result["isolated_vc"]

    checked = {}
    for name in ("fixed_aggregate", "daily_aggregate", "isolated_vc"):
        subset = selected if name != "isolated_vc" else selected[selected.vc == largest_vc]
        needs = sorted(int(n) for n in subset.gpu_num.unique())
        scenario = result["scenarios"][name]
        assert needs == scenario["needs"]
        assert len(subset) == scenario["job_count"]
        if name == "fixed_aggregate":
            capacities = [int(caps.loc[window[0], "total"])]
        elif name == "daily_aggregate":
            capacities = [int(caps.loc[day, "total"]) for day in window]
        else:
            capacities = [int(caps.loc[day, largest_vc]) for day in window]
        assert capacities == scenario["calendar"]["capacities"]
        checked[name] = {"job_count": int(len(subset)), "needs": needs, "capacities": capacities}

    receipt = {
        "cluster": cluster,
        "window_days": result["window_days"],
        "isolated_vc": largest_vc,
        "scenarios": checked,
        "primary_output_sha256": sha(content),
        "verification_sha256": sha(Path(__file__).read_bytes()),
        "pandas": pd.__version__,
        "scope": "Independent pandas re-derivation of scenario job sets and calendar capacities; no replay.",
    }
    (args.output_dir / "verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v["job_count"] for k, v in checked.items()}))


if __name__ == "__main__":
    main()
