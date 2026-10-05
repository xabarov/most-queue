"""Offline analytic checks of EPIC-063 accounting, dates and source boundaries."""

import io
import json
import sys
import zipfile
from datetime import date

import pytest

from examples import resource_observability_audit as audit

CAPS = "date,vcA,vcB,total\n2020-01-01,2,0,2\n2020-01-02,1,1,2\n"


def row(job="a", **changes):
    result = dict(
        zip(
            audit.FIELDS,
            [
                job,
                "user",
                "vcA",
                "1",
                "1",
                "1",
                "COMPLETED",
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:01",
                "2020-01-01 00:00:04",
                "3",
                "1",
            ],
        )
    )
    result.update(changes)
    return ",".join(result[k] for k in audit.FIELDS)


def log(*rows):
    return io.StringIO(",".join(audit.FIELDS) + "\n" + "\n".join(rows) + "\n")


def fixture_archive(extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for cluster in audit.CLUSTERS:
            archive.writestr(f"data/{cluster}/cluster_gpu_number.csv", CAPS)
            archive.writestr(f"data/{cluster}/cluster_log.csv", log(row()).getvalue())
        if extra:
            archive.writestr(extra, "not executed")
    return stream.getvalue()


def test_daily_changes_totals_and_missing_dates():
    capacities, report = audit.read_capacities(io.StringIO(CAPS))
    assert len(capacities) == 2 and report["vc_day_changes"] == 2
    assert report["total_changes"] == report["sum_vc_not_total_days"] == report["missing_days"] == 0
    assert report["zero_vc_days"] == 1
    _, report = audit.read_capacities(io.StringIO(CAPS.replace("2020-01-02,1,1,2", "2020-01-03,1,1,3")))
    assert report["missing_days"] == report["sum_vc_not_total_days"] == report["total_changes"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "date,total\n",
        "date,vcA,vcA,total\n",
        "date,bad,total\n",
        "date,vcA,total\n",
        "date,vcA,total\n2020-01-01,-1,1\n",
        CAPS + "2020-01-01,1,1,2\n",
        CAPS + "2020-01-03,1,2\n",
        CAPS + "2020-01-03,1,2,3,4\n",
        CAPS.replace("2020-01-02", "bad"),
    ],
)
def test_invalid_capacity_tables_fail(text):
    with pytest.raises(ValueError):
        audit.read_capacities(io.StringIO(text))


@pytest.mark.parametrize("value", ["", "nan", "-1", "1.5", "1.0", "True"])
def test_resource_counts_are_not_silently_coerced(value):
    assert audit.integer(value) is None


@pytest.mark.parametrize(
    "value",
    ["", "Unknown", "2020-13-01 00:00:00", "2020-01-01T00:00:00", "2020-01-01 00:00:00+08:00", "2020-01-01 00:00:60"],
)
def test_unknown_or_ambiguous_clock_is_not_imputed(value):
    assert audit.timestamp(value) is None


def test_naive_clock_difference_and_same_date_lookup():
    caps, _ = audit.read_capacities(io.StringIO(CAPS))
    t = audit.timestamp("2020-01-01 23:59:59")
    assert audit.timestamp("2020-01-02 00:00:00") - t == 1
    assert audit.capacity_at(caps, "vcA", t, 2)["need_above_daily_count"] == 0
    assert audit.capacity_at(caps, "vcA", t + 1, 2)["need_above_daily_count"] == 1
    assert audit.capacity_at(caps, "vcB", t, 1)["zero_count"] == 1
    assert audit.capacity_at(caps, "total", t, 1) == {"unknown_vc": 1}
    assert audit.capacity_at(caps, "missing", t, 1) == {"unknown_vc": 1}
    assert audit.capacity_at(caps, "vcA", t - 2 * audit.DAY, 1) == {"missing_date": 1}
    assert audit.capacity_at(caps, "vcA", None, 1) == {"invalid_time": 1}


def test_half_open_occupancy_work_and_simultaneous_events():
    events = [(0, 2), (10, -2), (5, 1), (20, -1)]
    report = audit.occupancy(events, {0: 2})
    assert report["peak_requested_gpu"] == 3
    assert report["full_gpu_seconds"] == report["covered_gpu_seconds"] == 35
    assert report["excess_gpu_seconds"] == report["excess_seconds"] == 5
    report = audit.occupancy([(0, 2), (5, -2), (5, 2), (10, -2)], {0: 2})
    assert report["peak_requested_gpu"] == 2 and report["excess_seconds"] == 0


def test_cross_midnight_capacity_change_and_missing_day_not_filled():
    d = audit.DAY
    report = audit.occupancy([(d - 2, 2), (d + 3, -2)], {0: 4, 1: 1})
    assert report["full_gpu_seconds"] == report["covered_gpu_seconds"] == 10
    assert report["excess_gpu_seconds"] == report["excess_seconds"] == 3
    report = audit.occupancy([(0, 2), (3 * d, -2)], {0: 1, 2: 1})
    assert report["full_gpu_seconds"] == 6 * d and report["covered_gpu_seconds"] == 4 * d
    assert report["excess_gpu_seconds"] == report["excess_seconds"] == 2 * d
    assert audit.occupancy([(0, 1), (1, -1)], {})["covered_gpu_seconds"] == 0


@pytest.mark.parametrize("events", [[(0, -1)], [(0, 1)]])
def test_occupancy_rejects_unbalanced_events(events):
    with pytest.raises(ValueError):
        audit.occupancy(events, {})


def test_all_states_and_invalid_jobs_stay_in_accounting():
    caps, _ = audit.read_capacities(io.StringIO(CAPS))
    result = audit.audit_jobs(
        log(
            row(),
            row("b", state="FAILED"),
            row("c", gpu_num="0"),
            row("d", state="SUSPENDED"),
            row("e", start_time="Unknown"),
            row("f", end_time="2020-01-01 00:00:01", duration="0"),
            row("a", start_time="2019-12-31 23:59:59", duration="5", queue="-1"),
        ),
        caps,
    )
    c = result["counts"]
    assert c["rows"] == 7 and c["gpu_jobs"] == 6 and c["cpu_only"] == 1
    assert c["duplicate_job_ids"] == 1 and c["unique_job_ids"] == 6
    assert c["gpu_closed_terminal"] == 2 and c["gpu_excluded_from_occupancy"] == 4
    assert c["negative_wait"] == c["invalid_start"] == c["zero_execution"] == 1
    assert result["states"]["SUSPENDED"] == 1
    assert result["closed_terminal_occupancy"]["full_gpu_seconds"] == 6
    assert result["completed_gpu_descriptive"] == {
        "count": 1,
        "mean_s": 3.0,
        "p99_s": 3.0,
        "mean_recorded_w": 1.0,
        "positive_recorded_waits": 1,
    }


def test_duration_sojourn_confusion_and_unknown_vc_are_flagged():
    caps, _ = audit.read_capacities(io.StringIO(CAPS))
    result = audit.audit_jobs(log(row(duration="4", queue="4", vc="vcMissing")), caps)
    assert result["derived_fields"]["duration_not_execution"] == 1
    assert result["derived_fields"]["duration_equals_sojourn"] == 1
    assert result["derived_fields"]["queue_not_wait"] == 1
    assert result["counts"]["unknown_vcs"] == 1
    assert result["closed_terminal_occupancy"]["full_gpu_seconds"] == 3
    assert result["closed_terminal_occupancy"]["vc_covered_gpu_seconds"] == 0


def test_malformed_rows_and_nonfinite_numbers_counted_not_repaired():
    caps, _ = audit.read_capacities(io.StringIO(CAPS))
    result = audit.audit_jobs(log(row(duration="inf", queue="NaN", gpu_num="-1"), "bad,width"), caps)
    for key in ("invalid_duration", "invalid_queue", "invalid_gpu", "malformed_width"):
        assert result["counts"][key] == 1
    assert result["completed_gpu_descriptive"]["count"] == 0
    with pytest.raises(ValueError, match="schema"):
        audit.audit_jobs(io.StringIO("wrong,header\n"), caps)


def test_allowlisted_zip_exact_hashes_and_repetition():
    payload = fixture_archive()
    first = audit.audit_archive(payload)
    assert first == audit.audit_archive(payload)
    assert len(first["inputs"]) == 8
    assert sum(r["counts"]["rows"] for r in first["clusters"].values()) == 4
    with pytest.raises(ValueError, match="allowlist"):
        audit.audit_archive(fixture_archive("../../escape.py"))


def test_cache_hash_check_opt_in_and_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "data.zip"
    with pytest.raises(FileNotFoundError):
        audit.verified_archive(path)
    payload = fixture_archive()
    monkeypatch.setattr(audit, "SHA256", audit.digest(payload))
    monkeypatch.setattr(audit.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    assert audit.verified_archive(path, True) == payload
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA-256"):
        audit.verified_archive(path, True)
    assert path.read_bytes() == b"bad"


def test_cli_outputs_aggregates_with_no_user_or_job_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "verified_archive", lambda *a: fixture_archive())
    monkeypatch.setattr(sys, "argv", ["audit", "--output-dir", str(tmp_path)])
    audit.main()
    result = (tmp_path / "helios-audit.json").read_bytes()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["scheduler_runs"] == 0 and manifest["output_sha256"] == audit.digest(result)
    assert b'"job_id"' not in result and b'"user"' not in result
    assert audit.timestamp("2020-01-01 00:00:00") == date(2020, 1, 1).toordinal() * audit.DAY
