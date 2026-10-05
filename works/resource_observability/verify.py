"""Independent full-data checks using pandas and vectorized event integrals.

Does not import the primary audit. Verifies raw clocks, identities, states,
daily configurations, requested work, peak and hypothetical daily excess.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integrate(frame, caps):
    starts = frame["start"].to_numpy()
    ends = frame["end"].to_numpy()
    needs = frame["gpu_num"].to_numpy()
    days = caps.index.to_numpy()
    stamps = np.r_[starts, ends, days, days + 86400]
    delta = np.r_[needs, -needs, np.zeros(2 * len(days), dtype=np.int64)]
    times, indices = np.unique(stamps, return_inverse=True)
    differences = np.zeros(len(times), dtype=np.int64)
    np.add.at(differences, indices, delta)
    levels = np.cumsum(differences)
    assert not len(levels) or (levels.min() >= 0 and levels[-1] == 0)
    widths = np.diff(times)
    day = times[:-1] // 86400 * 86400
    values = caps.reindex(day).to_numpy()
    known = np.isfinite(values)
    total = int(np.dot(widths, levels[:-1]))
    covered = int(np.dot(widths[known], levels[:-1][known]))
    excess = np.maximum(levels[:-1][known] - values[known], 0)
    work = int(np.dot(widths[known], excess))
    seconds = int(widths[known][excess > 0].sum())
    return {
        "full_gpu_seconds": total,
        "covered_gpu_seconds": covered,
        "excess_gpu_seconds": work,
        "excess_seconds": seconds,
        "peak_requested_gpu": int(levels.max()) if len(levels) else 0,
    }


def check_cluster(archive, cluster, result):
    frame = pd.read_csv(archive.open(f"data/{cluster}/cluster_log.csv"))
    caps = pd.read_csv(archive.open(f"data/{cluster}/cluster_gpu_number.csv"))
    c = result["counts"]
    assert len(frame) == c["rows"] and frame.job_id.nunique() == c["unique_job_ids"]
    assert int(frame.job_id.duplicated().sum()) == c["duplicate_job_ids"]
    assert frame.state.value_counts().to_dict() == result["states"]
    assert not frame.isna().any().any()
    for column in ("gpu_num", "cpu_num", "node_num"):
        assert np.all(frame[column] >= 0) and np.all(frame[column] == frame[column].astype(np.int64))
    gpu = frame.gpu_num > 0
    assert int(gpu.sum()) == c["gpu_jobs"] and int((~gpu).sum()) == c["cpu_only"]
    assert frame[gpu].state.value_counts().to_dict() == result["gpu_states"]
    for kind in ("submit", "start", "end"):
        parsed = pd.to_datetime(frame[f"{kind}_time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
        frame[kind] = parsed.to_numpy().astype("datetime64[s]").astype(np.int64)
        assert str(parsed.min()) == result["source_clock_bounds"][f"first_{kind}"]
        assert str(parsed.max()) == result["source_clock_bounds"][f"last_{kind}"]
    assert np.all(frame.start >= frame.submit) and np.all(frame.end > frame.start)
    assert np.all(frame.duration == frame.end - frame.start) and np.all(frame.queue == frame.start - frame.submit)
    derived = result["derived_fields"]
    assert derived["duration_not_execution"] == derived["queue_not_wait"] == 0
    assert derived["duration_checked"] == derived["queue_checked"] == len(frame)
    assert derived["duration_equals_sojourn"] == int((frame.duration == frame.end - frame.submit).sum())
    dates = pd.to_datetime(caps.pop("date"), format="%Y-%m-%d")
    caps.index = dates.to_numpy().astype("datetime64[s]").astype(np.int64)
    assert caps.index.is_unique and caps.index.is_monotonic_increasing
    conf = result["configuration"]
    assert len(caps) == conf["days"]
    assert dates.min().strftime("%Y-%m-%d") == conf["first_date"]
    assert dates.max().strftime("%Y-%m-%d") == conf["last_date"]
    assert (caps.index[-1] - caps.index[0]) // 86400 + 1 - len(caps) == conf["missing_days"]
    assert int((caps.drop(columns="total").sum(axis=1) != caps.total).sum()) == conf["sum_vc_not_total_days"]
    assert caps.total.min() == conf["total_min"] and caps.total.max() == conf["total_max"]
    assert int((caps.total.diff().iloc[1:] != 0).sum()) == conf["total_changes"]
    assert int((caps.drop(columns="total").diff().iloc[1:] != 0).sum().sum()) == conf["vc_day_changes"]
    assert int((caps.drop(columns="total") == 0).sum().sum()) == conf["zero_vc_days"]
    assert frame.vc.nunique() == c["observed_vcs"]
    assert len(set(frame.vc) - set(caps.columns)) == c["unknown_vcs"]
    joined = caps.drop(columns="total").stack()
    for stage in ("submit", "start"):
        selected = frame[gpu]
        job_days = selected[stage] // 86400 * 86400
        index = pd.MultiIndex.from_arrays([job_days, selected.vc])
        limits = joined.reindex(index).to_numpy()
        present = np.isfinite(limits)
        assert int(present.sum()) == c.get(f"gpu_{stage}_compared", 0)
        assert int((~job_days.isin(caps.index)).sum()) == c.get(f"gpu_{stage}_missing_date", 0)
        assert int((limits == 0).sum()) == c.get(f"gpu_{stage}_zero_count", 0)
        assert int((selected.gpu_num.to_numpy()[present] > limits[present]).sum()) == c.get(
            f"gpu_{stage}_need_above_daily_count", 0
        )
    terminal = frame[gpu & frame.state.isin(["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"])]
    assert len(terminal) == c["gpu_closed_terminal"]
    observed = result["closed_terminal_occupancy"]
    summary = integrate(terminal, caps.total)
    assert all(observed[k] == v for k, v in summary.items())
    assert summary["full_gpu_seconds"] == int(((terminal.end - terminal.start) * terminal.gpu_num).sum())
    per_vc = [
        integrate(jobs, caps[vc] if vc in caps else pd.Series(dtype=float)) for vc, jobs in terminal.groupby("vc")
    ]
    assert sum(v["excess_gpu_seconds"] for v in per_vc) == observed["vc_excess_gpu_seconds"]
    assert sum(v["excess_seconds"] for v in per_vc) == observed["vc_excess_seconds_sum"]
    assert sum(v["excess_seconds"] > 0 for v in per_vc) == observed["vcs_with_excess"]
    assert sum(v["covered_gpu_seconds"] for v in per_vc) == observed["vc_covered_gpu_seconds"]
    assert len(per_vc) == observed["vcs_with_intervals"]
    done = terminal[terminal.state == "COMPLETED"]
    desc = result["completed_gpu_descriptive"]
    assert len(done) == desc["count"]
    assert int((done.queue > 0).sum()) == desc["positive_recorded_waits"]
    np.testing.assert_allclose(
        [done.duration.mean(), done.duration.quantile(0.99), done.queue.mean()],
        [desc["mean_s"], desc["p99_s"], desc["mean_recorded_w"]],
        rtol=1e-12,
    )
    duplicates = frame[frame.job_id.duplicated(keep=False)]
    return {
        "rows_verified": len(frame),
        "gpu_rows_verified": int(gpu.sum()),
        "duplicate_id_rows": len(duplicates),
        "duplicate_id_gpu_rows": int((duplicates.gpu_num > 0).sum()),
        "duplicate_id_exact_rows": int(duplicates.duplicated().sum()),
        "unknown_vc_gpu_rows": int((gpu & ~frame.vc.isin(caps.columns)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/resource_observability"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/resource_observability"))
    args = parser.parse_args()
    folder = args.output_dir
    manifest = json.loads((folder / "manifest.json").read_text())
    payload = (args.cache_dir / "helios-data.zip").read_bytes()
    assert sha(payload) == manifest["archive_sha256"]
    assert sha(Path(manifest["implementation"]).read_bytes()) == manifest["implementation_sha256"]
    content = (folder / manifest["output"]).read_bytes()
    assert sha(content) == manifest["output_sha256"]
    result = json.loads(content)
    verified = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for item in result["inputs"]:
            raw = archive.read(item["file"])
            assert len(raw) == item["bytes"] and sha(raw) == item["sha256"]
        for cluster, report in result["clusters"].items():
            verified[cluster] = check_cluster(archive, cluster, report)
            print(json.dumps({cluster: verified[cluster]}), flush=True)
    receipt = {
        "clusters": verified,
        "primary_output_sha256": sha(content),
        "verification_sha256": sha(Path(__file__).read_bytes()),
        "pandas": pd.__version__,
        "scope": "Independent pandas parsing, raw counts, clocks, daily counts, work and occupancy integrals; no replay.",
    }
    (folder / "verification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
