"""GPU-free tests of the observational pinned-vLLM trace wrapper."""

import json
from types import SimpleNamespace

import pytest

from tools.runpod_vllm_server import install_trace, validate_timestamps


def metrics(**changes):
    """Return the external attribute-based pinned statistics contract."""
    return SimpleNamespace(
        **{
            "queued_ts": 1.0,
            "scheduled_ts": 1.1,
            "first_token_ts": 1.2,
            "last_token_ts": 1.5,
            "arrival_time": 1777777777.0,
            "num_generation_tokens": 32,
            "num_preemptions": 0,
            **changes,
        }
    )


def test_trace_observes_only_final_outputs_and_preserves_identity(tmp_path):
    """Wrapping must neither mutate nor replace any ordinary output object."""

    class State:
        """Minimal fake of the pinned RequestState output hook."""

        def make_request_output(self, result):
            """Return the caller's exact object."""
            return result

    path = tmp_path / "trace.jsonl"
    install_trace(State, path)
    state = State()
    partial = SimpleNamespace(finished=False)
    assert state.make_request_output(None) is None
    assert state.make_request_output(partial) is partial
    assert path.read_text() == ""
    stats = metrics()
    final = SimpleNamespace(finished=True, request_id="cmpl-mqcal-1-0", metrics=stats)
    assert state.make_request_output(final) is final
    assert final.metrics is stats
    row = json.loads(path.read_text())
    assert row["request_id"] == final.request_id
    assert row["timestamp_clock"] == "monotonic"
    assert row["arrival_time_clock"] == "wall"
    assert row["num_generation_tokens"] == 32
    with pytest.raises(ValueError, match="Duplicate final"):
        state.make_request_output(final)
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"queued_ts": 0},
        {"first_token_ts": float("nan")},
        {"last_token_ts": float("inf")},
        {"last_token_ts": -1},
        {"first_token_ts": 1.6},
        {"scheduled_ts": 0.5},
    ],
)
def test_invalid_engine_timestamps_fail(changes):
    """No invalid timestamp can silently become a calibration observation."""
    with pytest.raises(ValueError):
        validate_timestamps(vars(metrics(**changes)))


def test_trace_refuses_existing_file(tmp_path):
    """Independent server starts cannot mix evidence."""
    path = tmp_path / "trace.jsonl"
    path.touch()
    with pytest.raises(FileExistsError):
        install_trace(object, path)
