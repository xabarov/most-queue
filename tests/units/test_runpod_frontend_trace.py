"""GPU-free contracts for optional frontend boundary instrumentation."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tools import runpod_frontend_trace as trace


def scope(request_id=b"mqfront-1", **changes):
    """Return a minimal ASGI scope with a safe synthetic identifier."""
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": [] if request_id is None else [(b"x-mq-trace-id", request_id)],
        **changes,
    }


def setup_trace(tmp_path):
    """Configure a new trace and return its path."""
    path = tmp_path / "frontend.jsonl"
    trace.configure_trace(path)
    return path


def completion_class():
    """Create an isolated fake of the two pinned async completion methods."""

    class Completion:
        """Preserve all caller objects and expose an ordinary async generator."""

        async def create_completion(self, request, raw_request=None):
            """Return a stream by identity after rendering the same request."""
            assert raw_request is request.raw
            assert await self.render_completion_request(request) is request
            return request.stream

        async def render_completion_request(self, request):
            """Yield once to exercise task-local state isolation."""
            await asyncio.sleep(0)
            return request

    trace.install_frontend_hooks(Completion)
    return Completion


def test_marks_chunks_identity_and_interleaved_context(tmp_path):
    """Observe boundary timestamps without replacing messages, scopes, or streams."""
    path = setup_trace(tmp_path)
    completion = completion_class()()

    async def run_one(number):
        request_scope = scope(f"mqfront-{number}".encode())
        chunks = [
            {"type": "http.request", "body": b"PRIVATE PROMPT", "more_body": True},
            {"type": "http.request", "body": b"SECRET", "more_body": False},
        ]
        outputs = [
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.body", "body": b"", "more_body": True},
            {"type": "http.response.body", "body": b"answer", "more_body": False},
        ]
        received = iter(chunks)
        sent = []

        async def stream():
            yield "value"

        request = SimpleNamespace(request_id=f"mqfront-{number}", raw=object(), stream=stream())

        async def receive():
            return next(received)

        async def send(message):
            sent.append(message)

        async def app(actual_scope, actual_receive, actual_send):
            assert actual_scope is request_scope
            assert await actual_receive() is chunks[0]
            assert await actual_receive() is chunks[1]
            result = await completion.create_completion(request, request.raw)
            assert result is request.stream
            assert await anext(result) == "value"
            await result.aclose()
            for message in outputs:
                await actual_send(message)

        await trace.FrontendTraceMiddleware(app)(request_scope, receive, send)
        assert all(actual is expected for actual, expected in zip(sent, outputs))
        assert trace._STATE.get() is None  # pylint: disable=protected-access

    async def concurrent():
        await asyncio.gather(run_one(1), run_one(2))

    asyncio.run(concurrent())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row["request_id"] for row in rows} == {"mqfront-1", "mqfront-2"}
    fields = [
        "http_enter_ts",
        "body_received_ts",
        "handler_enter_ts",
        "render_enter_ts",
        "render_exit_ts",
        "first_response_body_ts",
    ]
    for row in rows:
        marks = [row[field] for field in fields]
        assert marks == sorted(marks)
        assert row["schema_version"] == 1
        assert row["timestamp_clock"] == "monotonic"
        assert row["render_success"] is True
        assert row["completed_response_body"] is True
        assert row["response_status"] == 200
        assert row["exception"] is None
        assert "instrumentation_error" not in row
    assert "PRIVATE" not in path.read_text()
    assert "SECRET" not in path.read_text()
    assert "answer" not in path.read_text()


@pytest.mark.parametrize("request_id", [b"secret-token", b"mqfront-1\xff", b"mqfront-1\n"])
def test_malformed_present_identifier_is_an_invalid_record(tmp_path, request_id):
    """Retain a bounded contract error without retaining untrusted header data."""
    path = setup_trace(tmp_path)

    async def app(_scope, _receive, _send):
        return

    asyncio.run(trace.FrontendTraceMiddleware(app)(scope(request_id), None, None))
    record = json.loads(path.read_text())
    assert record["request_id"] is None
    assert record["instrumentation_error"] == ["missing_or_invalid_trace_id"]
    assert "secret-token" not in path.read_text()


def test_id_mismatch_and_duplicate_marks_preserve_behavior(tmp_path):
    """Hook contract failures invalidate the trace without breaking the completion."""
    path = setup_trace(tmp_path)
    completion = completion_class()()
    request = SimpleNamespace(request_id="mqfront-9", raw=None, stream=object())

    async def app(_scope, _receive, _send):
        assert await completion.create_completion(request) is request.stream
        assert await completion.create_completion(request) is request.stream

    asyncio.run(trace.FrontendTraceMiddleware(app)(scope(), None, None))
    row = json.loads(path.read_text())
    assert "handler_request_id_mismatch" in row["instrumentation_error"]
    assert "duplicate_mark:handler_enter_ts" in row["instrumentation_error"]
    assert "duplicate_mark:render_enter_ts" in row["instrumentation_error"]
    assert "duplicate_mark:render_exit_ts" in row["instrumentation_error"]


@pytest.mark.parametrize("failure", [ValueError("PRIVATE ERROR"), asyncio.CancelledError("PRIVATE CANCEL")])
def test_errors_and_cancellation_propagate_and_reset_context(tmp_path, failure):
    """Exceptional outcomes preserve exception identity and log type alone."""
    path = setup_trace(tmp_path)

    class Completion:
        """Throw from rendering to exercise both hook and ASGI finally blocks."""

        async def create_completion(self, request, raw_request=None):
            """Forward to rendering."""
            assert raw_request is None
            return await self.render_completion_request(request)

        async def render_completion_request(self, _request):
            """Raise the exact test exception."""
            raise failure

    trace.install_frontend_hooks(Completion)

    async def app(_scope, _receive, _send):
        await Completion().create_completion(SimpleNamespace(request_id="mqfront-1"))

    async def run():
        with pytest.raises(type(failure)) as caught:
            await trace.FrontendTraceMiddleware(app)(scope(), None, None)
        assert caught.value is failure
        assert trace._STATE.get() is None  # pylint: disable=protected-access

    asyncio.run(run())
    row = json.loads(path.read_text())
    assert row["exception"] == type(failure).__name__
    assert row["render_exception"] == type(failure).__name__
    assert "render_exit_ts" in row
    assert row["completed_response_body"] is False
    assert "PRIVATE" not in path.read_text()


@pytest.mark.parametrize("changes", [{"type": "websocket"}, {"method": "GET"}, {"path": "/health"}, {"headers": []}])
def test_unrelated_and_uninstrumented_requests_bypass(tmp_path, changes):
    """Controls receive their original callables and produce no trace record."""
    path = setup_trace(tmp_path)
    incoming = scope(**changes)
    receive, send = object(), object()

    async def app(actual_scope, actual_receive, actual_send):
        assert actual_scope is incoming
        assert actual_receive is receive
        assert actual_send is send
        assert trace._STATE.get() is None  # pylint: disable=protected-access

    asyncio.run(trace.FrontendTraceMiddleware(app)(incoming, receive, send))
    assert path.read_text() == ""


def test_exclusive_configuration_and_missing_configuration(tmp_path, monkeypatch):
    """Refuse reused evidence paths and an unconfigured middleware instance."""
    path = setup_trace(tmp_path)
    with pytest.raises(FileExistsError):
        trace.configure_trace(path)
    monkeypatch.setattr(trace, "_WRITER", None)
    with pytest.raises(RuntimeError, match="configure_trace"):
        trace.FrontendTraceMiddleware(None)


def test_duplicate_ids_and_write_errors_are_visible(tmp_path, monkeypatch):
    """Do not silently count duplicate observations or failed writes as evidence."""
    path = setup_trace(tmp_path)

    async def app(_scope, _receive, _send):
        return

    middleware = trace.FrontendTraceMiddleware(app)
    asyncio.run(middleware(scope(), None, None))
    with pytest.raises(ValueError, match="Duplicate frontend"):
        asyncio.run(middleware(scope(), None, None))
    assert len(path.read_text().splitlines()) == 1
    monkeypatch.setattr(trace.os, "write", lambda *_args: 0)
    with pytest.raises(OSError, match="Incomplete frontend"):
        asyncio.run(middleware(scope(b"mqfront-2"), None, None))


def test_duplicate_headers_are_rejected_without_retaining_values(tmp_path):
    """Ambiguous trace headers must never be silently selected."""
    path = setup_trace(tmp_path)

    async def app(_scope, _receive, _send):
        return

    incoming = scope(headers=[(b"X-MQ-Trace-ID", b"mqfront-1"), (b"x-mq-trace-id", b"mqfront-2")])
    asyncio.run(trace.FrontendTraceMiddleware(app)(incoming, None, None))
    row = json.loads(path.read_text())
    assert row["request_id"] is None
    assert row["instrumentation_error"] == ["missing_or_invalid_trace_id"]


def test_send_failure_is_not_a_completed_response(tmp_path):
    """A failed final transport send must not be recorded as complete."""
    path = setup_trace(tmp_path)
    failure = ConnectionError("PRIVATE TRANSPORT")

    async def send(_message):
        raise failure

    async def app(_scope, _receive, actual_send):
        await actual_send({"type": "http.response.body", "body": b"private", "more_body": False})

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(trace.FrontendTraceMiddleware(app)(scope(), None, send))
    assert caught.value is failure
    row = json.loads(path.read_text())
    assert row["completed_response_body"] is False
    assert row["exception"] == "ConnectionError"
    assert "first_response_body_ts" in row
    assert "PRIVATE" not in path.read_text()


def test_bootstrap_controls_middleware_before_gpu_import(monkeypatch, tmp_path):
    """Reject unsupported CLI profiles without requiring vLLM or GPU dependencies."""
    from tools import runpod_frontend_server  # pylint: disable=import-outside-toplevel

    for middleware_arg in (["--middleware", "arbitrary.Hook"], ["--middleware=arbitrary.Hook"]):
        monkeypatch.setattr(
            "sys.argv", ["bootstrap", "--trace-path", str(tmp_path / "engine.jsonl"), "--", "model", *middleware_arg]
        )
        with pytest.raises(SystemExit) as caught:
            runpod_frontend_server.main()
        assert caught.value.code == 2


def test_bootstrap_pinned_imports_alias_and_cli(monkeypatch, tmp_path):
    """Resolve the exact pinned modules and share state with CLI middleware imports."""
    import sys  # pylint: disable=import-outside-toplevel
    from types import ModuleType  # pylint: disable=import-outside-toplevel

    from tools import runpod_frontend_server, runpod_vllm_server  # pylint: disable=import-outside-toplevel

    calls = []
    modules = {}
    for name in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.cli",
        "vllm.entrypoints.cli.main",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.completion",
        "vllm.entrypoints.openai.completion.serving",
        "vllm.v1",
        "vllm.v1.engine",
        "vllm.v1.engine.output_processor",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["vllm"].__version__ = "0.32.0"
    modules["vllm.entrypoints.cli.main"].main = lambda: calls.append(("cli", sys.argv))
    completion = object()
    request_state = object()
    modules["vllm.entrypoints.openai.completion.serving"].OpenAIServingCompletion = completion
    modules["vllm.v1.engine.output_processor"].RequestState = request_state
    monkeypatch.setattr(trace, "configure_trace", lambda path: calls.append(("frontend", path)))
    monkeypatch.setattr(trace, "install_frontend_hooks", lambda cls: calls.append(("hooks", cls)))
    monkeypatch.setattr(runpod_vllm_server, "install_trace", lambda cls, path: calls.append(("engine", cls, path)))
    engine_path = tmp_path / "engine.jsonl"
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--trace-path", str(engine_path), "--", "model", "--port", "8000"])
    # Restore the basename alias after the test, even when the bootstrap creates it.
    monkeypatch.setitem(sys.modules, "runpod_frontend_trace", trace)
    runpod_frontend_server.main()
    assert sys.modules["runpod_frontend_trace"] is trace
    assert sys.modules["tools.runpod_frontend_trace"] is trace
    assert calls == [
        ("frontend", tmp_path / "frontend_timestamps.jsonl"),
        ("hooks", completion),
        ("engine", request_state, engine_path),
        (
            "cli",
            [
                "vllm",
                "serve",
                "model",
                "--port",
                "8000",
                "--middleware",
                "runpod_frontend_trace.FrontendTraceMiddleware",
            ],
        ),
    ]
    modules["vllm"].__version__ = "0.32.1"
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--trace-path", str(engine_path), "--", "model"])
    with pytest.raises(RuntimeError, match="Expected vLLM 0.32.0"):
        runpod_frontend_server.main()
    assert len(calls) == 4


def test_write_failure_preserves_active_application_exception(tmp_path, monkeypatch):
    """Expose a failed write as cause while propagating the original exception."""
    path = setup_trace(tmp_path)
    failure = asyncio.CancelledError("private")

    async def app(_scope, _receive, _send):
        raise failure

    monkeypatch.setattr(trace.os, "write", lambda *_args: 0)

    async def run():
        # Check the middleware boundary directly: Python 3.10's asyncio.run
        # can replace CancelledError while transferring a cancelled Task result.
        with pytest.raises(asyncio.CancelledError) as caught:
            await trace.FrontendTraceMiddleware(app)(scope(), None, None)
        assert caught.value is failure
        assert isinstance(failure.__cause__, OSError)

    asyncio.run(run())
    assert path.read_text() == ""
