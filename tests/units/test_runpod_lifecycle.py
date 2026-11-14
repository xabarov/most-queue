"""Offline lifecycle and real detached-process cleanup regressions."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tools import runpod_lifecycle as life

NAME = "most-queue-pilot-10d7886b-e9c6-4ed4-baf6-4cf503c329cf"
IMAGE = "example.invalid/image@sha256:" + "a" * 64
POD = {"id": "owned_123", "name": NAME, "image": IMAGE, "cost": 0.59, "gpu": {"count": 1}}
NEIGHBOR = {"id": "neighbor_456", "name": NAME + "-other", "image": IMAGE}


def inventory(pods, cursor=None, more=False):
    return {"pods": pods, "pagination": {"nextCursor": cursor, "hasNextPage": more}}


class FakeClient:
    """Small stateful provider, retaining every mutation for assertions."""

    def __init__(self, pods=None, failures=0, delete_effect=True):
        self.pods = {p["id"]: dict(p) for p in (pods if pods is not None else [POD, NEIGHBOR])}
        self.calls = []
        self.failures = failures
        self.delete_effect = delete_effect
        self.list_error = None
        self.get_404 = False

    def request(self, method, path, payload=None, timeout=20):
        self.calls.append((method, path, payload))
        if method == "GET" and path.startswith("/pods?"):
            if self.list_error is not None:
                return self.list_error
            return inventory(list(self.pods.values()))
        pod_id = path.split("/")[2]
        if method == "GET":
            if pod_id not in self.pods or self.get_404:
                raise life.ApiError(status=404)
            return self.pods[pod_id]
        if method == "DELETE":
            if self.failures:
                self.failures -= 1
                raise life.ApiError("body containing a secret", status=503)
            if self.delete_effect:
                self.pods.pop(pod_id, None)
            return None
        if method == "POST":
            self.pods[pod_id]["status"] = "EXITED"
            return self.pods[pod_id]
        raise AssertionError("Unexpected fake API call")


def intent(tmp_path, **overrides):
    data = {
        "schema_version": 1,
        "name": NAME,
        "image": IMAGE,
        "owner_pid": os.getpid(),
        "owner_start_ticks": life.process_start_ticks(os.getpid()),
        "created_epoch": time.time() - 1,
        "work_deadline_epoch": time.time() + 60,
        "hard_deadline_epoch": time.time() + 120,
        "project_env_file": str(tmp_path / ".env"),
        "create_body": {"name": NAME, "image": IMAGE},
    }
    data.update(overrides)
    life.atomic_json(tmp_path / "attempt.json", data)
    return data


def test_key_reads_only_named_literal_assignments(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_ENV_FILE", raising=False)
    external = tmp_path / "selected.env"
    external.write_text("OTHER=irrelevant\nexport RUNPOD_API_KEY = 'dummy-local-token'\n")
    project = tmp_path / ".env"
    project.write_text('UNRELATED=$(touch forbidden)\nexport RUNPOD_ENV_FILE = "selected.env"\n')
    assert life.load_api_key(project) == "dummy-local-token"
    monkeypatch.setenv("RUNPOD_ENV_FILE", str(external))
    assert life.load_api_key(tmp_path / "missing") == "dummy-local-token"
    external.write_text("RUNPOD_API_KEY=first\nRUNPOD_API_KEY=second\n")
    with pytest.raises(life.LifecycleError, match="assignment"):
        life.load_api_key(project)


def test_atomic_private_json_and_no_temporary_collision(tmp_path):
    path = tmp_path / "state" / "receipt.json"
    threads = [threading.Thread(target=life.atomic_json, args=(path, {"number": number})) for number in range(12)]
    for worker in threads:
        worker.start()
    for worker in threads:
        worker.join()
    assert life.read_json(path)["number"] in range(12)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert list(path.parent.iterdir()) == [path]


def test_proc_birth_ticks_reuse_and_zombie():
    ticks = life.process_start_ticks(os.getpid())
    assert life.owner_alive(os.getpid(), ticks)
    assert not life.owner_alive(os.getpid(), ticks + 1)
    assert not life.owner_alive(-1, ticks)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 5
        while life.process_start_ticks(child.pid) is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert life.process_start_ticks(child.pid) is None
    finally:
        child.wait(timeout=5)


def test_paginated_inventory_includes_all_clusters():
    class Pages:
        def request(self, method, path):
            assert "includeClusterPods=true" in path
            if "cursor=" not in path:
                return inventory([NEIGHBOR], "cursor + /", True)
            assert parse_qs(urlsplit(path).query)["cursor"] == ["cursor + /"]
            return inventory([POD])

    assert life.list_pods(Pages()) == [NEIGHBOR, POD]


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"pods": []},
        {"pods": [], "pagination": {}},
        inventory([{}]),
        inventory([], "bad", False),
        inventory([], None, True),
        {"pods": [], "pagination": {"nextCursor": None}},
    ],
)
def test_incomplete_inventory_never_proves_absence(response):
    client = FakeClient([])
    client.list_error = response if response is not None else []
    with pytest.raises(life.LifecycleError):
        life.exact_pod(client, POD["id"], NAME, IMAGE)


def test_inventory_cursor_loop_is_rejected():
    class Loop:
        def request(self, method, path):
            return inventory([], "same-cursor", True)

    with pytest.raises(life.LifecycleError, match="loop"):
        life.list_pods(Loop())


def test_404_with_present_inventory_is_not_absence():
    client = FakeClient()
    client.get_404 = True
    assert life.exact_pod(client, POD["id"], NAME, IMAGE) == POD


@pytest.mark.parametrize("field,value", [("name", NAME + "-neighbor"), ("image", "different"), ("id", "another_id")])
def test_ownership_mismatch_never_mutates(field, value):
    client = FakeClient()
    client.pods[POD["id"]][field] = value
    with pytest.raises(life.LifecycleError, match="identity"):
        life.cleanup_exact(client, POD["id"], NAME, IMAGE, sleep=0)
    assert all(method == "GET" for method, _, _ in client.calls)


def test_bad_expected_name_never_calls_provider():
    client = FakeClient()
    with pytest.raises(life.LifecycleError, match="unique"):
        life.cleanup_exact(client, POD["id"], "most-queue-pilot", IMAGE)
    assert not client.calls


def test_cleanup_retry_stop_and_fresh_proof_only():
    client = FakeClient(failures=2)
    result = life.cleanup_exact(client, POD["id"], NAME, IMAGE, sleep=0)
    assert result["status"] == "cleanup_confirmed"
    assert result["confirmed_absent_epoch"] <= time.time()
    assert client.pods == {NEIGHBOR["id"]: NEIGHBOR}
    assert ("POST", "/pods/owned_123/action", {"action": "stop"}) in client.calls
    assert all("neighbor_456" not in path for method, path, _ in client.calls if method != "GET")
    assert client.calls[-1][1].startswith("/pods?")


def test_delete_success_and_exited_are_not_cleanup_proof():
    client = FakeClient(delete_effect=False)
    client.pods[POD["id"]]["status"] = "EXITED"
    with pytest.raises(life.LifecycleError, match="unconfirmed"):
        life.cleanup_exact(client, POD["id"], NAME, IMAGE, attempts=2, sleep=0)


@pytest.mark.parametrize("trigger", ["request", "owner_dead", "pid_reuse", "deadline", "stage", "progress"])
def test_watchdog_cleanup_triggers(tmp_path, trigger):
    data = intent(tmp_path)
    if trigger == "request":
        life.atomic_json(tmp_path / "cleanup.json", {"requested": True, "reason": "test"})
    elif trigger in {"owner_dead", "pid_reuse"}:
        data["owner_start_ticks"] += 1
        if trigger == "owner_dead":
            data["owner_pid"] = 2147483647
        life.atomic_json(tmp_path / "attempt.json", data)
    elif trigger == "deadline":
        data["created_epoch"] -= 5
        data["work_deadline_epoch"] = time.time() - 2
        life.atomic_json(tmp_path / "attempt.json", data)
    else:
        stage = {"stage": "testing", "deadline_epoch": time.time() + 60, "progress_deadline_epoch": time.time() + 60}
        stage["deadline_epoch" if trigger == "stage" else "progress_deadline_epoch"] = time.time() - 1
        life.atomic_json(tmp_path / "stage.json", stage)
    life.atomic_json(tmp_path / "pod.json", POD)
    client = FakeClient()
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "cleanup_confirmed"
    assert result["pod_id"] == POD["id"]
    assert NEIGHBOR["id"] in client.pods


def test_monitoring_acknowledged_before_any_network_read(tmp_path):
    intent(tmp_path)
    client = FakeClient([])
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "monitoring"
    assert result["watchdog_pid"] == os.getpid()
    assert not client.calls


def test_pod_receipt_requires_fresh_identity(tmp_path):
    intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)
    result = life.watchdog(tmp_path, FakeClient(), poll_seconds=0, max_iterations=1)
    assert result["status"] == "monitoring"
    assert result["pod_id"] == POD["id"]


def test_ambiguous_create_empty_inventory_then_delayed_recovery(tmp_path):
    intent(tmp_path, owner_start_ticks=1)
    life.atomic_json(tmp_path / "create-result.json", {"outcome": "ambiguous", "status": None})
    client = FakeClient([NEIGHBOR])
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "ambiguous_create"
    assert "confirmed_absent_epoch" not in result
    client.pods[POD["id"]] = dict(POD, env={"SECRET": "must-not-persist"})
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "cleanup_confirmed"
    recovered = life.read_json(tmp_path / "pod.json")
    assert recovered["id"] == POD["id"]
    assert "SECRET" not in (tmp_path / "pod.json").read_text()
    assert "env" not in recovered


@pytest.mark.parametrize("outcome,status", [("not_sent", None), ("rejected", 400), ("rejected", 403)])
def test_definitive_no_create_finalizes_only_after_inventory(tmp_path, outcome, status):
    intent(tmp_path, owner_start_ticks=1)
    life.atomic_json(tmp_path / "create-result.json", {"outcome": outcome, "status": status})
    client = FakeClient([NEIGHBOR])
    assert life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=1)["status"] == "cleanup_confirmed"
    assert client.calls and client.calls[0][1].startswith("/pods?")


@pytest.mark.parametrize("pods", [[dict(POD, image="unexpected")], [POD, dict(POD, id="duplicate")]])
def test_create_gap_ownership_error_never_deletes(tmp_path, pods):
    intent(tmp_path, owner_start_ticks=1)
    client = FakeClient(pods)
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "ownership_error"
    assert all(method == "GET" for method, _, _ in client.calls)


def test_api_outage_stays_pending_without_false_success(tmp_path):
    intent(tmp_path, owner_start_ticks=1)
    life.atomic_json(tmp_path / "pod.json", POD)

    class Outage:
        def request(self, *args, **kwargs):
            raise life.ApiError("private response", status=503)

    result = life.watchdog(tmp_path, Outage(), poll_seconds=0, max_iterations=2)
    assert result["status"] == "cleanup_pending"
    assert "confirmed_absent_epoch" not in result
    assert "private" not in (tmp_path / "watchdog.json").read_text()


@pytest.fixture
def http_provider():
    state = {"pods": {POD["id"]: dict(POD), NEIGHBOR["id"]: dict(NEIGHBOR)}, "calls": [], "mode": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, value=None, headers=None):
            self.send_response(status)
            for key, value_ in (headers or {}).items():
                self.send_header(key, value_)
            self.end_headers()
            if value is not None:
                self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            state["calls"].append(("GET", self.path, self.headers.get("Authorization")))
            if state["mode"] == "redirect":
                return self.respond(302, {"secret": "body"}, {"Location": base + "/credential-target"})
            if state["mode"] == "error":
                return self.respond(503, {"detail": "private-provider-secret"})
            path = urlsplit(self.path).path
            if path == "/v2/pods":
                return self.respond(200, inventory(list(state["pods"].values())))
            pod = state["pods"].get(path.rsplit("/", 1)[-1])
            self.respond(200, pod) if pod else self.respond(404, {"detail": "private-provider-secret"})

        def do_DELETE(self):
            state["calls"].append(("DELETE", self.path, self.headers.get("Authorization")))
            state["pods"].pop(self.path.rsplit("/", 1)[-1], None)
            self.respond(204)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base + "/v2", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_errors_and_redirects_never_expose_provider_content(http_provider):
    base, state = http_provider
    client = life.RestClient("dummy-key", base_url=base)
    for mode, status in [("error", 503), ("redirect", 302)]:
        state["mode"] = mode
        with pytest.raises(life.ApiError) as caught:
            client.request("GET", "/pods")
        assert caught.value.status == status
        assert "secret" not in str(caught.value)
        assert "dummy-key" not in str(caught.value)
    assert all("credential-target" not in path for _, path, _ in state["calls"])
    assert life.ApiError("dummy-key and response body", status=503).args == ("Runpod API request failed (HTTP 503)",)


def _wait_receipt(path, expected, seconds=12):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            result = life.read_json(path)
            if result["status"] == expected:
                return result
        time.sleep(0.03)
    raise AssertionError(f"Timed out waiting for {expected}")


@pytest.mark.parametrize("trigger", ["owner_death", "signal"])
def test_detached_watchdog_survives_owner_and_handles_signal(tmp_path, http_provider, trigger):
    base, state = http_provider
    intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)
    # The bootstrap injects a local fake HTTP client; no production CLI flag can
    # override the provider endpoint. It also records the child's actual env.
    bootstrap = tmp_path / "fake-python"
    bootstrap.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(Path(life.__file__).resolve().parent.parent)!r})\n"
        "from tools import runpod_lifecycle as m\n"
        "real = m.RestClient\n"
        f"m.RestClient = lambda key: real(key, base_url={base!r})\n"
        f"m.atomic_json({str(tmp_path / 'child-env.json')!r}, {{'env':dict(os.environ),'sid':os.getsid(0),'pid':os.getpid()}})\n"
        "raise SystemExit(m.main(sys.argv[2:]))\n"
    )
    bootstrap.chmod(0o700)
    external = tmp_path / "selected.env"
    external.write_text("RUNPOD_API_KEY=dummy-http-token\nUNRELATED=irrelevant\n")
    project = tmp_path / ".env"
    project.write_text(f"RUNPOD_ENV_FILE='{external}'\n")
    owner_script = (
        "import os, sys, time\n"
        f"sys.path.insert(0, {str(Path(life.__file__).resolve().parent.parent)!r})\n"
        "from tools import runpod_lifecycle as m\n"
        "from pathlib import Path\n"
        f"p=Path({str(tmp_path)!r})\n"
        "a=m.read_json(p/'attempt.json'); a.update(owner_pid=os.getpid(), owner_start_ticks=m.process_start_ticks(os.getpid()))\n"
        "m.atomic_json(p/'attempt.json', a)\n"
        f"sys.executable={str(bootstrap)!r}\n"
        "os.environ['RUNPOD_API_KEY']='do-not-inherit'; os.environ['UNRELATED_SECRET']='do-not-inherit'\n"
        f"os.environ['RUNPOD_ENV_FILE']={str(external)!r}\n"
        "worker=m.start_watchdog(p,p/'.env')\n"
        "print(worker.pid, flush=True)\n" + ("os._exit(0)\n" if trigger == "owner_death" else "time.sleep(30)\n")
    )
    owner = subprocess.Popen([sys.executable, "-c", owner_script], stdout=subprocess.PIPE, text=True)
    watchdog_pid = int(owner.stdout.readline())
    try:
        assert os.getsid(watchdog_pid) == watchdog_pid
        if trigger == "signal":
            _wait_receipt(tmp_path / "watchdog.json", "monitoring")
            os.kill(watchdog_pid, signal.SIGTERM)
        else:
            owner.wait(timeout=5)
        result = _wait_receipt(tmp_path / "watchdog.json", "cleanup_confirmed")
        assert result["pod_id"] == POD["id"]
        assert state["pods"] == {NEIGHBOR["id"]: NEIGHBOR}
        child = life.read_json(tmp_path / "child-env.json")
        assert child["pid"] == child["sid"] == watchdog_pid
        assert "RUNPOD_API_KEY" not in child["env"]
        assert "UNRELATED_SECRET" not in child["env"]
        assert child["env"]["RUNPOD_ENV_FILE"] == str(external)
        assert all(auth == "Bearer dummy-http-token" for _, _, auth in state["calls"])
        assert "dummy-http-token" not in (tmp_path / "watchdog.log").read_text()
    finally:
        if owner.poll() is None:
            owner.terminate()
        owner.wait(timeout=5)
        if life.process_start_ticks(watchdog_pid) is not None:
            os.kill(watchdog_pid, signal.SIGTERM)


@pytest.mark.parametrize("filename", ["stage.json", "cleanup.json"])
def test_corrupt_control_file_still_cleans_exact_owned_pod(tmp_path, filename):
    intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)
    (tmp_path / filename).write_text("{broken")
    client = FakeClient()
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "cleanup_confirmed"
    assert result["reason"] == "invalid_control_state"
    assert state_ids(client) == {NEIGHBOR["id"]}


def state_ids(client):
    return set(client.pods)


def test_monotonic_guard_triggers_deadline_after_wall_clock_rollback(tmp_path, monkeypatch):
    intent(tmp_path, created_epoch=90, work_deadline_epoch=110, hard_deadline_epoch=120)
    life.atomic_json(tmp_path / "pod.json", POD)
    wall = iter([100, 80])
    monotonic = iter([100, 121])
    monkeypatch.setattr(life.time, "time", lambda: next(wall, 80))
    monkeypatch.setattr(life.time, "monotonic", lambda: next(monotonic, 121))
    result = life.watchdog(tmp_path, FakeClient(), poll_seconds=0, max_iterations=1)
    assert result["status"] == "cleanup_confirmed"
    assert result["reason"] == "deadline"


def test_recovered_pod_cannot_overwrite_conflicting_controller_id(tmp_path):
    life.atomic_json(tmp_path / "pod.json", dict(POD, id="controller_different_id"))
    with pytest.raises(life.LifecycleError, match="identity"):
        life._recover_pod(tmp_path, POD)
    assert life.read_json(tmp_path / "pod.json")["id"] == "controller_different_id"


@pytest.mark.parametrize("filename", ["stage.json", "cleanup.json", "pod.json", "attempt.json"])
def test_corruption_after_monitoring_cleans_original_identity(tmp_path, monkeypatch, filename):
    intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)
    client = FakeClient()
    sleeps = []

    def corrupt_after_first_poll(_seconds):
        if not sleeps:
            (tmp_path / filename).write_text("{corrupted")
        sleeps.append(1)

    monkeypatch.setattr(life.time, "sleep", corrupt_after_first_poll)
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "cleanup_confirmed"
    assert result["pod_id"] == POD["id"]
    assert set(client.pods) == {NEIGHBOR["id"]}


def test_changed_intent_uses_original_name_and_id(tmp_path, monkeypatch):
    original = intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)
    client = FakeClient()

    def mutate(_seconds):
        life.atomic_json(tmp_path / "attempt.json", dict(original, name=NEIGHBOR["name"], image="other"))
        life.atomic_json(tmp_path / "pod.json", NEIGHBOR)

    monkeypatch.setattr(life.time, "sleep", mutate)
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=2)
    assert result["status"] == "cleanup_confirmed"
    assert result["pod_id"] == POD["id"]
    assert set(client.pods) == {NEIGHBOR["id"]}


def test_first_observation_corrupt_pod_reconciles_original_unique_intent(tmp_path):
    intent(tmp_path)
    (tmp_path / "pod.json").write_text("{corrupt")
    client = FakeClient()
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=1)
    assert result["status"] == "cleanup_confirmed"
    assert life.read_json(tmp_path / "recovered-pod.json")["id"] == POD["id"]
    assert (tmp_path / "pod.json").read_text() == "{corrupt"
    assert set(client.pods) == {NEIGHBOR["id"]}


def test_local_heartbeat_does_not_flood_provider(tmp_path):
    intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)
    client = FakeClient()
    result = life.watchdog(tmp_path, client, poll_seconds=0, max_iterations=10)
    assert result["status"] == "monitoring"
    assert len(client.calls) == 1


def test_transport_error_retries_monitoring_until_deadline(tmp_path):
    intent(tmp_path)
    life.atomic_json(tmp_path / "pod.json", POD)

    class Outage:
        def request(self, *args, **kwargs):
            raise life.ApiError(status=503)

    result = life.watchdog(tmp_path, Outage(), poll_seconds=0, max_iterations=2)
    assert result["status"] == "monitoring"
    assert "confirmed_absent_epoch" not in result


@pytest.mark.parametrize("status", [408, 429, 500, None])
def test_nondefinitive_create_failure_remains_ambiguous(tmp_path, status):
    intent(tmp_path, owner_start_ticks=1)
    life.atomic_json(tmp_path / "create-result.json", {"outcome": "rejected", "status": status})
    result = life.watchdog(tmp_path, FakeClient([]), poll_seconds=0, max_iterations=1)
    assert result["status"] == "ambiguous_create"
    assert "confirmed_absent_epoch" not in result
