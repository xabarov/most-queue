"""Exact-resource Runpod lifecycle primitives and a detached cleanup supervisor.

This module uses only the standard library. The supervisor never allocates a pod
and never edits the campaign ledger. An unresolved create remains unresolved even
when a fresh inventory is empty: a delayed create can still materialize later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Exact type checks intentionally reject booleans as PID/status/deadline numbers.
# pylint: disable=unidiomatic-typecheck


API_BASE = "https://api.runpod.io/v2"
_NAME = re.compile(r"most-queue-pilot-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_POD_FIELDS = ("id", "name", "image", "cost", "gpu", "cloud", "disk", "mounts")


class LifecycleError(Exception):
    """A lifecycle operation could not establish its required safety condition."""


class ApiError(LifecycleError):
    """A sanitized HTTP failure without provider content or request data."""

    def __init__(self, message="request failed", status=None, retry_after=None):
        del message
        self.status = status if type(status) is int else None
        self.retry_after = retry_after if type(retry_after) is int and 0 <= retry_after <= 60 else None
        # Deliberately do not retain even a caller-supplied message: it may contain
        # response bodies, URLs or credentials from a lower-level exception.
        super().__init__(f"Runpod API request failed (HTTP {self.status})")


class _OwnershipError(LifecycleError):
    """Identity verification failed; no mutation is permitted."""


def _env_value(path, variable):
    """Read one literal assignment; shell evaluation and interpolation are forbidden."""
    values = []
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(variable) + r"\s*=\s*(.*?)\s*$")
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                match = pattern.match(line)
                if match:
                    value = match.group(1)
                    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
                        value = value[1:-1]
                    values.append(value)
    except (OSError, UnicodeError):
        raise LifecycleError("Cannot read selected environment file") from None
    if len(values) != 1 or not values[0] or any(c in values[0] for c in "\r\n\x00"):
        raise LifecycleError("Selected environment assignment is missing or invalid")
    return values[0]


def load_api_key(project_env: Path) -> str:
    """Load only the selected external file's RUNPOD_API_KEY into memory."""
    pointer = os.environ.get("RUNPOD_ENV_FILE") or _env_value(project_env, "RUNPOD_ENV_FILE")
    external = Path(pointer).expanduser()
    if not external.is_absolute():
        external = Path(project_env).resolve().parent / external
    return _env_value(external, "RUNPOD_API_KEY")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ApiError(status=code)


class RestClient:
    """Minimal JSON client; credentials never follow an HTTP redirect."""

    def __init__(self, api_key, base_url=API_BASE):
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method, path, payload=None, timeout=20):
        """Return a JSON object or null, raising a sanitized ApiError on failure."""
        if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
            raise LifecycleError("API path must be relative to the fixed API base")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ApiError(status=response.status)
                result = json.loads(raw) if raw.strip() else None
                if result is not None and not isinstance(result, dict):
                    raise ApiError(status=response.status)
                return result
        except urllib.error.HTTPError as error:
            try:
                retry_after = int(error.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
            error.close()
            raise ApiError(status=error.code, retry_after=retry_after) from None
        except (OSError, ValueError, urllib.error.URLError):
            raise ApiError() from None


def atomic_json(path, data):
    """Durably replace a private JSON file, using a unique temporary file."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path) -> dict:
    """Read a JSON object without leaking file content through exceptions."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError
        return result
    except (OSError, ValueError, UnicodeError):
        raise LifecycleError("Cannot read valid lifecycle JSON object") from None


def process_start_ticks(pid) -> int | None:
    """Return Linux process birth ticks, treating a zombie as absent."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat[stat.rindex(")") + 2 :].split()
        return None if fields[0] in {"Z", "X", "x"} else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def owner_alive(pid, ticks):
    """Check process identity as well as PID, preventing PID reuse mistakes."""
    return type(ticks) is int and ticks > 0 and process_start_ticks(pid) == ticks


def _valid_identity(pod):
    return isinstance(pod, dict) and all(isinstance(pod.get(k), str) and pod[k] for k in ("id", "name", "image"))


def _validate_intent(name, image):
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise _OwnershipError("Expected a complete unique Most-Queue pilot name")
    if not isinstance(image, str) or not image:
        raise _OwnershipError("Expected image is missing")


def list_pods(client) -> list:
    """Return a complete, validated inventory; partial pagination is an error."""
    pods, seen_ids, seen_cursors = [], set(), set()
    path = "/pods?includeClusterPods=true&limit=1000"
    for _ in range(1000):
        page = client.request("GET", path)
        if not isinstance(page, dict) or not isinstance(page.get("pods"), list):
            raise LifecycleError("Malformed pod inventory")
        pagination = page.get("pagination")
        if not isinstance(pagination, dict) or "nextCursor" not in pagination:
            raise LifecycleError("Incomplete pod pagination")
        cursor = pagination["nextCursor"]
        more = pagination.get("hasNextPage")
        if type(more) is not bool or (more and (not isinstance(cursor, str) or not cursor)):
            raise LifecycleError("Malformed pod pagination")
        if not more and cursor is not None:
            raise LifecycleError("Inconsistent pod pagination")
        for pod in page["pods"]:
            if not _valid_identity(pod) or pod["id"] in seen_ids:
                raise LifecycleError("Malformed or duplicated inventory identity")
            seen_ids.add(pod["id"])
            pods.append(pod)
        if not more:
            return pods
        if cursor in seen_cursors:
            raise LifecycleError("Pod pagination loop")
        seen_cursors.add(cursor)
        path = "/pods?includeClusterPods=true&limit=1000&cursor=" + urllib.parse.quote(cursor, safe="")
    raise LifecycleError("Pod pagination limit exceeded")


def _assert_identity(pod, pod_id, name, image):
    if not _valid_identity(pod) or (pod["id"], pod["name"], pod["image"]) != (pod_id, name, image):
        raise _OwnershipError("Pod identity differs from the recorded owned resource")


def _pod_path(pod_id):
    if not isinstance(pod_id, str) or not pod_id or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise _OwnershipError("Invalid recorded pod identifier")
    return "/pods/" + pod_id


def exact_pod(client, pod_id, expected_name, image) -> dict | None:
    """Freshly verify all ownership fields; a 404 requires full inventory proof."""
    _validate_intent(expected_name, image)
    try:
        pod = client.request("GET", _pod_path(pod_id))
    except ApiError as error:
        if error.status != 404:
            raise
        matching = [pod for pod in list_pods(client) if pod["id"] == pod_id]
        if not matching:
            return None
        pod = matching[0]
    _assert_identity(pod, pod_id, expected_name, image)
    return pod


def cleanup_exact(client, pod_id, expected_name, image, attempts=5, sleep=2):
    """Delete only a freshly verified owned ID, then prove fresh absence."""
    _validate_intent(expected_name, image)
    path = _pod_path(pod_id)
    failures = 0
    for index in range(attempts):
        try:
            pod = exact_pod(client, pod_id, expected_name, image)
            if pod is None:
                return {"status": "cleanup_confirmed", "confirmed_absent_epoch": time.time()}
            try:
                client.request("DELETE", path)
            except ApiError:
                failures += 1
                if failures >= 2:
                    # Stop is a temporary cost mitigation, never a cleanup proof.
                    if exact_pod(client, pod_id, expected_name, image) is not None:
                        try:
                            client.request("POST", path + "/action", {"action": "stop"})
                        except ApiError:
                            pass
            if exact_pod(client, pod_id, expected_name, image) is None:
                return {"status": "cleanup_confirmed", "confirmed_absent_epoch": time.time()}
        except ApiError:
            pass
        if index + 1 < attempts:
            time.sleep(sleep)
    raise LifecycleError("Exact pod cleanup remains unconfirmed")


def _optional_json(path):
    return read_json(path) if Path(path).exists() else None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _recover_pod(attempt_dir, pod):
    """Publish recovered metadata atomically without overwriting a controller ID."""
    target = attempt_dir / "pod.json"
    recovered = {key: pod.get(key) for key in _POD_FIELDS}
    descriptor, temporary = tempfile.mkstemp(prefix=".recovered-pod.", dir=attempt_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(recovered, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
            _sync_directory(attempt_dir)
        except FileExistsError:
            try:
                existing = read_json(target)
            except LifecycleError:
                existing = None
            if not _valid_identity(existing):
                # Preserve the corrupt controller record for diagnosis. A second
                # durable record authorizes only this freshly reconciled ID.
                atomic_json(attempt_dir / "recovered-pod.json", recovered)
                return recovered
            _assert_identity(existing, pod["id"], pod["name"], pod["image"])
    finally:
        os.unlink(temporary)
    return read_json(target)


def watchdog(attempt_dir: Path, client, poll_seconds=2, *, provider_poll_seconds=15, max_iterations=None):
    """Supervise one immutable intent until exact cleanup is confirmed.

    ``max_iterations`` is solely a deterministic in-process test hook. The CLI
    never sets it; API outages and ambiguous creates do not make the worker exit.
    """
    attempt_dir = Path(attempt_dir)
    intent = read_json(attempt_dir / "attempt.json")
    receipt = {
        "schema_version": 1,
        "status": "monitoring",
        "watchdog_pid": os.getpid(),
        "watchdog_start_ticks": process_start_ticks(os.getpid()),
        "pod_id": None,
    }
    try:
        _validate_intent(intent.get("name"), intent.get("image"))
        if intent.get("schema_version") != 1 or not all(
            _number(intent.get(key)) for key in ("created_epoch", "work_deadline_epoch", "hard_deadline_epoch")
        ):
            raise _OwnershipError("Invalid immutable lifecycle intent")
        if not intent["created_epoch"] <= intent["work_deadline_epoch"] <= intent["hard_deadline_epoch"]:
            raise _OwnershipError("Invalid lifecycle deadline order")
        if type(intent.get("owner_pid")) is not int or type(intent.get("owner_start_ticks")) is not int:
            raise _OwnershipError("Invalid owner process identity")
    except _OwnershipError:
        receipt.update(status="ownership_error", heartbeat_epoch=time.time(), error="Invalid lifecycle intent")
        atomic_json(attempt_dir / "watchdog.json", receipt)
        raise

    received_signal = []
    previous_handlers = {}

    def on_signal(signum, _frame):
        received_signal.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[signum] = signal.signal(signum, on_signal)
        except ValueError:  # Tests may run the supervisor in a worker thread.
            pass
    start_wall, start_monotonic = time.time(), time.monotonic()
    iteration = 0
    cleanup_reason = None
    stage_guards = {}
    last_owned_pod: dict | None = None
    next_provider_read = 0.0
    # This durable acknowledgement is deliberately before the first cloud read.
    receipt["heartbeat_epoch"] = start_wall
    atomic_json(attempt_dir / "watchdog.json", receipt)
    try:
        while True:
            effective_now = max(time.time(), start_wall + time.monotonic() - start_monotonic)
            receipt["heartbeat_epoch"] = time.time()
            receipt.pop("error", None)
            try:
                try:
                    if read_json(attempt_dir / "attempt.json") != intent:
                        cleanup_reason = cleanup_reason or "immutable_intent_changed"
                except LifecycleError:
                    cleanup_reason = cleanup_reason or "immutable_intent_unreadable"
                if received_signal:
                    cleanup_reason = cleanup_reason or "supervisor_signal"
                if not owner_alive(intent["owner_pid"], intent["owner_start_ticks"]):
                    cleanup_reason = cleanup_reason or "owner_dead"
                if effective_now >= intent["work_deadline_epoch"] or effective_now >= intent["hard_deadline_epoch"]:
                    cleanup_reason = cleanup_reason or "deadline"
                try:
                    requested = _optional_json(attempt_dir / "cleanup.json")
                    if requested and requested.get("requested") is True:
                        cleanup_reason = cleanup_reason or "controller_requested"
                    stage = _optional_json(attempt_dir / "stage.json")
                    if stage:
                        for key in ("deadline_epoch", "progress_deadline_epoch"):
                            deadline = stage.get(key)
                            if not _number(deadline):
                                cleanup_reason = cleanup_reason or "invalid_stage_deadline"
                            else:
                                # Anchor each observed deadline to monotonic time too.
                                marker = (str(stage.get("stage")), key, deadline)
                                guard = stage_guards.setdefault(
                                    marker, time.monotonic() + max(0, deadline - time.time())
                                )
                                if effective_now >= deadline or time.monotonic() >= guard:
                                    cleanup_reason = cleanup_reason or "stage_deadline"
                except LifecycleError:
                    # Corrupt controller status must not prevent cleanup of the
                    # independently recorded, freshly verified resource.
                    cleanup_reason = cleanup_reason or "invalid_control_state"
                try:
                    pod: dict | None = _optional_json(attempt_dir / "pod.json")
                    if pod is not None:
                        _assert_identity(pod, pod.get("id"), intent["name"], intent["image"])
                    if last_owned_pod is not None and (pod is None or pod.get("id") != last_owned_pod.get("id")):
                        raise _OwnershipError("Recorded pod ID changed")
                except (LifecycleError, TypeError):
                    cleanup_reason = cleanup_reason or "invalid_pod_record"
                    pod = last_owned_pod
                if pod:
                    receipt["pod_id"] = pod.get("id")
                    if cleanup_reason:
                        receipt.update(status="cleanup_pending", reason=cleanup_reason)
                        atomic_json(attempt_dir / "watchdog.json", receipt)
                        result = cleanup_exact(
                            client, pod.get("id"), intent["name"], intent["image"], attempts=3, sleep=poll_seconds
                        )
                        receipt.update(result)
                    elif time.monotonic() >= next_provider_read:
                        current = exact_pod(client, pod.get("id"), intent["name"], intent["image"])
                        next_provider_read = time.monotonic() + provider_poll_seconds
                        if current is None:
                            receipt.update(
                                status="cleanup_confirmed", confirmed_absent_epoch=time.time(), reason="pod_absent"
                            )
                        else:
                            last_owned_pod = {key: current.get(key) for key in _POD_FIELDS}
                            receipt["status"] = "monitoring"
                elif cleanup_reason:
                    receipt.update(status="cleanup_pending", reason=cleanup_reason)
                    inventory = list_pods(client)
                    matches = [candidate for candidate in inventory if candidate["name"] == intent["name"]]
                    if len(matches) > 1 or (matches and matches[0]["image"] != intent["image"]):
                        raise _OwnershipError("Unique create intent does not identify exactly one owned resource")
                    if matches:
                        pod = _recover_pod(attempt_dir, matches[0])
                        receipt["pod_id"] = pod.get("id")
                        atomic_json(attempt_dir / "watchdog.json", receipt)
                        receipt.update(
                            cleanup_exact(
                                client, pod.get("id"), intent["name"], intent["image"], attempts=3, sleep=poll_seconds
                            )
                        )
                    else:
                        result = _optional_json(attempt_dir / "create-result.json") or {}
                        rejected = (
                            result.get("outcome") == "rejected"
                            and type(result.get("status")) is int
                            and result["status"] in (400, 401, 403, 404, 422)
                        )
                        if result.get("outcome") == "not_sent" or rejected:
                            receipt.update(status="cleanup_confirmed", confirmed_absent_epoch=time.time())
                        else:
                            receipt["status"] = "ambiguous_create"
                else:
                    receipt["status"] = "monitoring"
            except _OwnershipError:
                receipt.update(status="ownership_error", error="Resource ownership verification failed")
                cleanup_reason = cleanup_reason or "ownership_error"
            except ApiError:
                receipt.update(
                    status="cleanup_pending" if cleanup_reason else "monitoring",
                    error="Provider temporarily unavailable; retrying",
                )
                next_provider_read = time.monotonic() + provider_poll_seconds
            except (LifecycleError, OSError, ValueError, TypeError):
                receipt.update(
                    status="cleanup_pending" if cleanup_reason else "monitoring",
                    error="Lifecycle check failed; retrying",
                )
                # A broken state file is unsafe to keep running against indefinitely.
                cleanup_reason = cleanup_reason or "lifecycle_check_failed"
            atomic_json(attempt_dir / "watchdog.json", receipt)
            if receipt["status"] == "cleanup_confirmed":
                return receipt
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                return receipt
            time.sleep(poll_seconds)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def start_watchdog(attempt_dir, project_env: Path) -> subprocess.Popen:
    """Launch a detached worker without inheriting credentials or other settings."""
    attempt_dir = Path(attempt_dir).resolve()
    attempt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = attempt_dir / "watchdog.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if os.environ.get("RUNPOD_ENV_FILE"):
        # Preserve the selected source pointer, never the key or unrelated env.
        environment["RUNPOD_ENV_FILE"] = os.environ["RUNPOD_ENV_FILE"]
    with os.fdopen(descriptor, "ab") as log:
        return subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "watch",
                "--attempt-dir",
                str(attempt_dir),
                "--project-env",
                str(Path(project_env).resolve()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )


def main(argv=None):
    """Run the persistent supervisor using only the selected local key source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["watch"])
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--project-env", type=Path, required=True)
    args = parser.parse_args(argv)
    # Key loading can fail transiently too. The process must stay available so a
    # corrected selected file can restore cleanup without another agent turn.
    while True:
        try:
            client = RestClient(load_api_key(args.project_env))
            watchdog(args.attempt_dir, client)
            return 0
        except (LifecycleError, OSError, ValueError):
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
