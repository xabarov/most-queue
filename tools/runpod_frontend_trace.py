"""Observe frontend request boundaries without inspecting completion payloads."""

import functools
import json
import os
import re
import threading
import time
from contextvars import ContextVar
from pathlib import Path

_STATE = ContextVar("mq_frontend_trace", default=None)
_WRITER = None
_SAFE_ID = re.compile(rb"mqfront-[0-9]+\Z")


class _Writer:
    """Append one atomic record per request to an exclusively created trace."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8"):
            pass
        self.lock = threading.Lock()
        self.seen = set()

    def append(self, record):
        """Fail visibly on duplicate IDs, serialization errors, or incomplete writes."""
        encoded = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
        request_id = record["request_id"]
        with self.lock:
            if request_id is not None and request_id in self.seen:
                raise ValueError("Duplicate frontend request ID")
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                if os.write(descriptor, encoded) != len(encoded):
                    raise OSError("Incomplete frontend trace write")
            finally:
                os.close(descriptor)
            if request_id is not None:
                self.seen.add(request_id)


def configure_trace(path):
    """Create a fresh trace at startup before constructing any middleware."""
    global _WRITER  # pylint: disable=global-statement
    _WRITER = _Writer(path)


def _error(state, message):
    state.setdefault("instrumentation_error", []).append(message)


def _mark(state, name):
    if name in state:
        _error(state, "duplicate_mark:" + name)
    else:
        state[name] = time.monotonic()


def install_frontend_hooks(completion_class):
    """Observe validated-handler and render boundaries on the pinned completion class."""
    original_create = completion_class.create_completion
    original_render = completion_class.render_completion_request

    @functools.wraps(original_create)
    async def create(self, request, raw_request=None):
        state = _STATE.get()
        if state is not None:
            _mark(state, "handler_enter_ts")
            request_id = getattr(request, "request_id", None)
            if state["request_id"] is None or request_id != state["request_id"]:
                _error(state, "handler_request_id_mismatch")
        return await original_create(self, request, raw_request)

    @functools.wraps(original_render)
    async def render(self, request):
        state = _STATE.get()
        if state is not None:
            _mark(state, "render_enter_ts")
        try:
            result = await original_render(self, request)
        except BaseException as exc:
            if state is not None:
                state["render_exception"] = type(exc).__name__
            raise
        else:
            if state is not None:
                state["render_success"] = True
            return result
        finally:
            if state is not None:
                _mark(state, "render_exit_ts")

    completion_class.create_completion = create
    completion_class.render_completion_request = render
    return original_create, original_render


class FrontendTraceMiddleware:
    """Observe POST completions while forwarding original ASGI objects unchanged."""

    def __init__(self, app):
        if _WRITER is None:
            raise RuntimeError("configure_trace must run before middleware initialization")
        self.app = app
        self.writer = _WRITER

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != "/v1/completions":
            await self.app(scope, receive, send)
            return
        values = [value for key, value in scope.get("headers", ()) if key.lower() == b"x-mq-trace-id"]
        if not values:
            await self.app(scope, receive, send)
            return
        state = {
            "schema_version": 1,
            "timestamp_clock": "monotonic",
            "request_id": None,
            "http_enter_ts": time.monotonic(),
            "response_status": None,
            "completed_response_body": False,
            "exception": None,
        }
        if len(values) == 1 and _SAFE_ID.fullmatch(values[0]):
            state["request_id"] = values[0].decode("ascii")
        else:
            _error(state, "missing_or_invalid_trace_id")
        token = _STATE.set(state)

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                _mark(state, "body_received_ts")
            return message

        async def observed_send(message):
            if message["type"] == "http.response.start":
                state["response_status"] = message["status"]
            elif message["type"] == "http.response.body":
                if message.get("body") and "first_response_body_ts" not in state:
                    _mark(state, "first_response_body_ts")
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                state["completed_response_body"] = True

        request_error = None
        try:
            await self.app(scope, observed_receive, observed_send)
        except BaseException as exc:
            request_error = exc
            state["exception"] = type(exc).__name__
            raise
        finally:
            _STATE.reset(token)
            try:
                self.writer.append(state)
            except BaseException as write_error:
                if request_error is not None:
                    raise request_error from write_error
                raise
