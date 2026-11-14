"""Offline controller contracts: accounting, immutable bounds and safe artifacts."""

import base64
import copy
import hashlib
import io
import json
import math
import tarfile

import pytest

from tools import runpod_pilot as pilot
from tools.runpod_lifecycle import LifecycleError, atomic_json


@pytest.fixture(name="inputs")
def campaign_inputs():
    """Provide a small authorized campaign without depending on campaign files."""
    policy = {
        "authorized": True,
        "campaign_budget_usd": 5.0,
        "max_compute_usd_per_hour": 0.75,
        "max_total_usd_per_hour": 0.77,
        "max_campaign_seconds": 14400,
        "cleanup_reserve_seconds": 900,
        "max_allocations": 2,
        "max_concurrent_pods": 1,
        "allowed_gpu_ids": ["test-gpu"],
        "allowed_cloud": "SECURE",
        "gpu_count": 1,
        "max_container_disk_gb": 80,
        "max_persistent_disk_gb": 0,
        "allow_network_volume": False,
    }
    profile = {
        "campaign_id": "test-campaign",
        "image": "test/image@sha256:" + "a" * 64,
        "min_cuda_version": "13.0",
        "work_seconds": 2700,
        "container_disk_gb": 80,
        "storage_usd_per_hour": 0.01,
    }
    plan = {
        "gpu_id": "test-gpu",
        "cloud": "SECURE",
        "gpu_count": 1,
        "compute_usd_per_hour": 0.5,
        "storage_usd_per_hour": 0.01,
        "work_seconds": 2700,
        "container_disk_gb": 80,
        "persistent_disk_gb": 0,
        "network_volume_ids": [],
    }
    ledger = {"spent_usd": 0.0, "allocated_seconds": 0, "allocations": 0, "active_pod_ids": []}
    return policy, profile, plan, ledger


# Each argument controls an independent ledger observation.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def saved_attempt(state, name, created, finished, cost=0.5, status="cleanup_confirmed"):
    """Persist one synthetic failed or successful allocation for ledger recovery."""
    directory = state / "attempts" / name
    atomic_json(directory / "create-started.json", {"epoch": created})
    atomic_json(
        directory / "attempt.json",
        {"created_epoch": created, "plan": {"compute_usd_per_hour": 0.5, "storage_usd_per_hour": 0.01}},
    )
    atomic_json(directory / "pod.json", {"id": name, "cost": cost})
    atomic_json(directory / "watchdog.json", {"status": status, "pod_id": name, "confirmed_absent_epoch": finished})
    atomic_json(directory / "outcome.json", {"successful": False})
    return directory


def test_reconcile_charges_all_attempts_and_never_resets_on_retry(tmp_path):
    """Retry and failure still consume aggregate time and conservative cost."""
    saved_attempt(tmp_path, "first", 1000, 2800, cost=0.4)
    first = pilot.reconcile_ledger(tmp_path)
    saved_attempt(tmp_path, "retry", 3000, 3900.1, cost=0.7)
    ledger = pilot.reconcile_ledger(tmp_path)
    expected = math.ceil(((1800 * 0.51 + 901 * 0.71) / 3600) * 1_000_000) / 1_000_000
    assert ledger["allocations"] == 2
    assert ledger["allocated_seconds"] == 2701
    assert ledger["spent_usd"] == expected
    assert ledger["spent_usd"] > first["spent_usd"]
    assert json.loads((tmp_path / "ledger.json").read_text()) == ledger
    assert pilot.reconcile_ledger(tmp_path) == ledger


@pytest.mark.parametrize("status", ["monitoring", "cleanup_unknown", "cleanup_requested"])
def test_reconcile_unconfirmed_cleanup_blocks(tmp_path, status):
    """No new allocation can proceed with unresolved prior ownership."""
    saved_attempt(tmp_path, "unresolved", 1000, 2000, status=status)
    with pytest.raises(LifecycleError, match="Unresolved"):
        pilot.reconcile_ledger(tmp_path)


def test_reconcile_definitively_rejected_create_has_no_charge(tmp_path):
    """A rejected create counts an attempt but has no billable resource lifetime."""
    directory = saved_attempt(tmp_path, "rejected", 1000, 1010)
    (directory / "pod.json").unlink()
    atomic_json(directory / "create-result.json", {"outcome": "rejected", "status": 422})
    atomic_json(
        directory / "watchdog.json", {"status": "cleanup_confirmed", "pod_id": None, "confirmed_absent_epoch": 1010}
    )
    ledger = pilot.reconcile_ledger(tmp_path)
    assert ledger["allocations"] == 1
    assert ledger["spent_usd"] == 0
    assert ledger["allocated_seconds"] == 0


def test_reconcile_recovered_identity_preserves_charge(tmp_path):
    """A corrupt original record cannot reset the cost of a recovered resource."""
    directory = saved_attempt(tmp_path, "recovered", 1000, 2800)
    recovered = json.loads((directory / "pod.json").read_text())
    atomic_json(directory / "recovered-pod.json", recovered)
    (directory / "pod.json").write_text("{")
    ledger = pilot.reconcile_ledger(tmp_path)
    assert ledger["allocated_seconds"] == 1800
    assert ledger["spent_usd"] == 0.255


def test_reconcile_different_cleanup_identity_blocks(tmp_path):
    """Another resource's absence cannot discharge the recorded allocation."""
    directory = saved_attempt(tmp_path, "owned", 1000, 2800)
    atomic_json(
        directory / "watchdog.json",
        {
            "status": "cleanup_confirmed",
            "pod_id": "neighbor",
            "confirmed_absent_epoch": 2800,
        },
    )
    with pytest.raises(LifecycleError, match="identities"):
        pilot.reconcile_ledger(tmp_path)


def allocation(inputs, tmp_path):
    """Prepare an intent and its matching provider response."""
    policy, profile, plan, ledger = inputs
    _, attempt = pilot.prepare_attempt(tmp_path, policy, profile, plan, {"public": "offer"}, ledger)
    pod = copy.deepcopy(attempt["create_body"])
    pod.update(id="test-pod", cost=0.5, cudaVersion="13.1")
    return attempt, pod


def test_valid_allocation_matches_intent(inputs, tmp_path):
    """Exact resources and compatible host CUDA are accepted."""
    attempt, pod = allocation(inputs, tmp_path)
    pilot.validate_allocated(pod, attempt, inputs[0], inputs[3])


def test_running_allocation_waits_for_container_exec(inputs, tmp_path, monkeypatch):
    """A RUNNING pod can answer exec409 until its image/container is ready."""
    attempt, pod = allocation(inputs, tmp_path)
    pod["status"] = "RUNNING"
    instance = pilot.Pilot(None, tmp_path, attempt)
    instance.pod = pod
    monkeypatch.setattr(instance, "check", lambda: None)
    monkeypatch.setattr(pilot, "exact_pod", lambda *_args: pod)
    sleeps = []
    monkeypatch.setattr(pilot.time, "sleep", sleeps.append)
    responses = [pilot.ApiError(status=409, retry_after=7), pilot.ApiError(status=503), "ready\n"]

    def remote(command, timeout):
        assert command == ["python3", "-c", "print('ready')"] and timeout == 10
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(instance, "remote", remote)
    assert instance.wait_running() == pod
    assert sleeps == [7, 5]
    assert not responses


@pytest.mark.parametrize(
    "change",
    [
        {"cost": 0.8},
        {"cloud": "COMMUNITY"},
        {"disk": 81},
        {"mounts": {"/data": "volume"}},
        {"gpu": {"id": "different-gpu", "count": 1}},
        {"gpu": {"id": "test-gpu", "count": 2}},
        {"image": "different/image"},
        {"name": "neighbor-pod"},
    ],
)
def test_allocated_resource_or_rate_mismatch_is_rejected(inputs, tmp_path, change):
    """Provider drift cannot silently expand authorized resources or hourly rate."""
    attempt, pod = allocation(inputs, tmp_path)
    pod.update(change)
    with pytest.raises(LifecycleError):
        pilot.validate_allocated(pod, attempt, inputs[0], inputs[3])


@pytest.mark.parametrize("cuda", [None, "12.9", "invalid", "13.beta"])
def test_incompatible_or_invalid_cuda_rejected(inputs, tmp_path, cuda):
    """CUDA host capability must be known and compatible before workload execution."""
    attempt, pod = allocation(inputs, tmp_path)
    pod["cudaVersion"] = cuda
    with pytest.raises(LifecycleError):
        pilot.validate_allocated(pod, attempt, inputs[0], inputs[3], require_cuda=True)


def test_initial_allocation_may_defer_cuda_until_readiness(inputs, tmp_path):
    """Creation can lack host metadata, while the readiness check remains strict."""
    attempt, pod = allocation(inputs, tmp_path)
    pod.pop("cudaVersion")
    pilot.validate_allocated(pod, attempt, inputs[0], inputs[3], require_cuda=False)
    with pytest.raises(LifecycleError):
        pilot.validate_allocated(pod, attempt, inputs[0], inputs[3], require_cuda=True)


@pytest.mark.parametrize("constraint", ["planned", "time", "money"])
def test_prepare_attempt_bounds_deadlines_and_reserves_cleanup(inputs, tmp_path, monkeypatch, constraint):
    """The earliest work, aggregate time or aggregate money bound always wins."""
    policy, profile, plan, ledger = inputs
    monkeypatch.setattr(pilot.time, "time", lambda: 10000.0)
    expected_total = 3600
    if constraint == "time":
        ledger["allocated_seconds"] = policy["max_campaign_seconds"] - 2100
        expected_total = 2100
    elif constraint == "money":
        ledger["spent_usd"] = 4.7
        expected_total = math.floor((5.0 - 4.7) / 0.77 * 3600)
    directory, attempt = pilot.prepare_attempt(tmp_path, policy, profile, plan, {"availability": "HIGH"}, ledger)
    assert attempt["hard_deadline_epoch"] == 10000 + expected_total
    assert attempt["work_deadline_epoch"] == attempt["hard_deadline_epoch"] - 900
    assert json.loads((directory / "attempt.json").read_text()) == attempt
    assert attempt["ledger_before"] == ledger


def test_prepare_attempt_cannot_spend_cleanup_reserve(inputs, tmp_path):
    """No work starts if remaining authorized time is only the cleanup reserve."""
    policy, profile, plan, ledger = inputs
    ledger["allocated_seconds"] = policy["max_campaign_seconds"] - policy["cleanup_reserve_seconds"]
    with pytest.raises(LifecycleError, match="cleanup reserve"):
        pilot.prepare_attempt(tmp_path, policy, profile, plan, {}, ledger)


def test_prepare_request_has_no_credential_or_external_ports(inputs, tmp_path, monkeypatch):
    """Environment secrets never enter pod requests or persisted public intent."""
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-secret-not-a-real-key")
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-secret")
    policy, profile, plan, ledger = inputs
    directory, attempt = pilot.prepare_attempt(tmp_path, policy, profile, plan, {}, ledger)
    request = attempt["create_body"]
    assert request["ports"] == []  # pylint: disable=use-implicit-booleaness-not-comparison
    assert request["mounts"] == {}  # pylint: disable=use-implicit-booleaness-not-comparison
    assert "RUNPOD_API_KEY" not in request["env"]
    assert "HF_TOKEN" not in request["env"]
    for path in directory.iterdir():
        assert "synthetic-secret" not in path.read_text()
        assert "synthetic-hf-secret" not in path.read_text()


def archive_file(path, member_name, kind=tarfile.REGTYPE):
    """Create a tiny synthetic archive for path and member-type validation."""
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.type = kind
        if kind == tarfile.SYMTYPE:
            member.linkname = "../../outside"
        if kind == tarfile.REGTYPE:
            member.size = 7
            archive.addfile(member, io.BytesIO(b"public\n"))
        else:
            archive.addfile(member)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside", tarfile.REGTYPE),
        ("/tmp/absolute-artifact", tarfile.REGTYPE),
        ("nested/link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
    ],
)
def test_extract_refuses_traversal_and_links(tmp_path, name, kind):
    """Exported archives cannot write outside their destination or install links."""
    archive = tmp_path / "artifacts.tar.gz"
    archive_file(archive, name, kind)
    with pytest.raises(LifecycleError, match="Unsafe"):
        pilot.extract_artifacts(archive, tmp_path / "artifacts")
    assert not (tmp_path / "outside").exists()


def test_extract_valid_nested_artifact(tmp_path):
    """Ordinary nested files are exported faithfully."""
    archive = tmp_path / "artifacts.tar.gz"
    archive_file(archive, "measurements/requests.jsonl")
    destination = tmp_path / "artifacts"
    pilot.extract_artifacts(archive, destination)
    assert (destination / "measurements/requests.jsonl").read_bytes() == b"public\n"


class FakeClient:
    """Fake provider records read-only preflight calls and forbids all mutations."""

    def __init__(self, pods=None):
        self.calls = []
        self.pods = pods or []

    def request(self, method, path, payload=None, timeout=20):  # pylint: disable=unused-argument
        """Serve only deterministic account, inventory and catalog reads."""
        self.calls.append((method, path))
        assert method == "GET", "Preflight must never create or mutate a resource"
        if path == "/account":
            return {"id": "selected-account"}
        if path.startswith("/pods?"):
            return {"pods": self.pods, "pagination": {"nextCursor": None, "hasNextPage": False}}
        if path.startswith("/catalog/gpus/"):
            return {"availability": "HIGH", "price": {"secure": 0.5}}
        raise AssertionError(f"Unexpected provider path: {path}")


def test_preflight_never_calls_create(inputs, tmp_path):
    """A successful preflight reads live stock and price without paid actions."""
    policy, profile, _, _ = inputs
    client = FakeClient()
    assert pilot.run_campaign(client, tmp_path, policy, profile, tmp_path / "unused.env", perform=False) == 0
    assert all(method == "GET" for method, _ in client.calls)
    assert not (tmp_path / "attempts").exists()
    assert (tmp_path / "preflight.json").exists()


@pytest.mark.parametrize("change", ["account", "policy", "closed"])
def test_campaign_identity_or_closed_state_blocks_preflight(inputs, tmp_path, change):
    """An authorization cannot migrate accounts, policy or completed campaigns."""
    policy, profile, _, _ = inputs
    metadata = {"account_id": "selected-account", "policy": copy.deepcopy(policy), "closed": False}
    if change == "account":
        metadata["account_id"] = "different-account"
    elif change == "policy":
        metadata["policy"]["campaign_budget_usd"] = 6
    else:
        metadata["closed"] = True
    atomic_json(tmp_path / "campaign.json", metadata)
    client = FakeClient()
    with pytest.raises(LifecycleError):
        pilot.run_campaign(client, tmp_path, policy, profile, tmp_path / "unused.env", perform=False)
    assert client.calls == [("GET", "/account")]
    assert not (tmp_path / "preflight.json").exists()


def test_existing_most_queue_pod_blocks_preflight(inputs, tmp_path):
    """An existing campaign resource must be reconciled before new plans proceed."""
    policy, profile, _, _ = inputs
    client = FakeClient([{"id": "existing-pod", "name": "most-queue-pilot-existing", "image": profile["image"]}])
    with pytest.raises(LifecycleError, match="existing Most-Queue"):
        pilot.run_campaign(client, tmp_path, policy, profile, tmp_path / "unused.env", perform=False)
    assert not any(path.startswith("/catalog/") for _, path in client.calls)


@pytest.fixture(name="serving_profile")
def fixture_serving_profile():
    """Read the prepared public calibration profile for offline command checks."""
    return json.loads((pilot.ROOT / "configs/runpod/calibration-profile.json").read_text())


@pytest.mark.parametrize("configured", [False, True])
def test_upload_preserves_configured_base_names_and_legacy_name(tmp_path, monkeypatch, serving_profile, configured):
    """Every public source is sent once with a matching content checksum."""
    profile = serving_profile if configured else {}
    instance = pilot.Pilot(None, tmp_path, {"profile": profile, "work_deadline_epoch": 1e20})
    commands = []
    monkeypatch.setattr(instance, "remote", commands.append)
    instance.upload_workload()
    expected_names = (
        {"runpod_calibration_workload.py", "runpod_workload.py", "calibration-workload.json", "runpod_vllm_server.py"}
        if configured
        else {"workload.py"}
    )
    assert {command[3].rsplit("/", 1)[1] for command in commands} == expected_names
    files = {name: source for source, name in pilot.workload_files(profile)}
    for command in commands:
        assert command[:2] == ["python3", "-c"]
        assert command[3].startswith(pilot.REMOTE + "/")
        content = base64.b64decode("".join(command[5:]), validate=True)
        assert hashlib.sha256(content).hexdigest() == command[4]
        assert content == files[command[3].rsplit("/", 1)[1]].read_bytes()
        compile(command[2], "<upload>", "exec")


@pytest.mark.parametrize("configured", [False, True])
def test_server_argv_bootstrap_or_legacy(tmp_path, monkeypatch, serving_profile, configured):
    """Bootstrap wraps the same pinned model/revision/server args on loopback."""
    profile = copy.deepcopy(serving_profile)
    if not configured:
        for field in (
            "workload_script",
            "workload_support",
            "workload_config",
            "server_bootstrap",
            "measurement_timeout_seconds",
        ):
            profile.pop(field)
    instance = pilot.Pilot(None, tmp_path, {"profile": profile, "work_deadline_epoch": 1e20})
    commands = []
    stages = []
    monkeypatch.setattr(instance, "remote", commands.append)
    monkeypatch.setattr(instance, "set_stage", lambda name, seconds: stages.append((name, seconds)))
    monkeypatch.setattr(instance, "probe", lambda: {"healthy": True})
    instance.start_server()
    command = json.loads(commands[-1][4])
    if configured:
        assert command[:6] == [
            "python3",
            pilot.REMOTE + "/runpod_vllm_server.py",
            "--trace-path",
            pilot.REMOTE + "/server_timestamps.jsonl",
            "--",
            profile["model"],
        ]
        assert "serve" not in command
    else:
        assert command[:3] == ["vllm", "serve", profile["model"]]
    assert command[-len(profile["server_args"]) :] == profile["server_args"]
    assert command[command.index("--revision") + 1] == profile["revision"]
    assert command[command.index("--tokenizer-revision") + 1] == profile["revision"]
    assert command[command.index("--seed") + 1] == str(profile["seed"])
    assert stages == [("model_load_and_warmup", 1200)]


@pytest.mark.parametrize("configured", [False, True])
def test_measurement_argv_and_bounded_margins(tmp_path, monkeypatch, serving_profile, configured):
    """Calibration gets 2700 seconds; old profiles keep 930/950 wrapper/stage."""
    profile = copy.deepcopy(serving_profile)
    if not configured:
        for field in (
            "workload_script",
            "workload_support",
            "workload_config",
            "server_bootstrap",
            "measurement_timeout_seconds",
        ):
            profile.pop(field)
    instance = pilot.Pilot(None, tmp_path, {"profile": profile, "work_deadline_epoch": 1e20})
    commands = []
    stages = []
    monkeypatch.setattr(instance, "remote", commands.append)
    monkeypatch.setattr(instance, "set_stage", lambda name, seconds: stages.append((name, seconds)))
    monkeypatch.setattr(instance, "probe", lambda: {"workload_exit": "0"})
    assert instance.measure()
    launch = commands[0]
    command = json.loads(launch[5])
    assert command[:2] == [
        "python3",
        pilot.REMOTE + ("/runpod_calibration_workload.py" if configured else "/workload.py"),
    ]
    assert command[2:7] == [pilot.REMOTE + "/measurements", "--model", profile["model"], "--seed", str(profile["seed"])]
    if configured:
        assert command[7:] == [
            "--config",
            pilot.REMOTE + "/calibration-workload.json",
            "--server-trace",
            pilot.REMOTE + "/server_timestamps.jsonl",
        ]
        assert "timeout=2730" in launch[3]
        assert stages == [("measurements", 2750)]
    else:
        assert len(command) == 7
        assert "timeout=930" in launch[3]
        assert stages == [("measurements", 950)]
    compile(launch[2], "<launcher>", "exec")
    compile(launch[3], "<runtime-wrapper>", "exec")


def test_configured_stage_still_clips_to_existing_work_deadline(tmp_path, serving_profile):
    """Configurable duration cannot expand the immutable spending work window."""
    instance = pilot.Pilot(None, tmp_path, {"profile": serving_profile, "work_deadline_epoch": 1.0})
    instance.set_stage("measurements", pilot.measurement_timeout(serving_profile) + 50)
    assert instance.stage_deadline == 1.0
    assert json.loads((tmp_path / "stage.json").read_text())["deadline_epoch"] == 1.0


@pytest.mark.parametrize("seconds", [None, True, 0, -1, float("nan"), float("inf"), "2700"])
def test_invalid_measurement_window_fails_before_provider_reads(inputs, tmp_path, seconds):
    """Invalid local settings cannot reach paid operations or provider calls."""
    policy, profile, _, _ = inputs
    profile["measurement_timeout_seconds"] = seconds
    client = FakeClient()
    with pytest.raises(LifecycleError, match="finite and positive"):
        pilot.run_campaign(client, tmp_path, policy, profile, tmp_path / "unused.env", perform=True)
    assert not client.calls


def test_missing_workload_fails_before_provider_reads(inputs, tmp_path):
    """All configured source files must be present before allocation."""
    policy, profile, _, _ = inputs
    profile["workload_script"] = "tools/does-not-exist.py"
    client = FakeClient()
    with pytest.raises(LifecycleError, match="exist inside"):
        pilot.run_campaign(client, tmp_path, policy, profile, tmp_path / "unused.env", perform=True)
    assert not client.calls


def test_upload_rejects_outside_source_and_colliding_base_names(tmp_path, monkeypatch):
    """Base-name uploads cannot overwrite another configured source."""
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "one").mkdir()
    (repository / "two").mkdir()
    (repository / "one/script.py").write_text("one")
    (repository / "two/script.py").write_text("two")
    (tmp_path / "outside.py").write_text("outside")
    monkeypatch.setattr(pilot, "ROOT", repository)
    with pytest.raises(LifecycleError, match="colliding"):
        pilot.workload_files({"workload_script": "one/script.py", "workload_support": ["two/script.py"]})
    with pytest.raises(LifecycleError, match="exist inside"):
        pilot.workload_files({"workload_script": "../outside.py"})


@pytest.mark.parametrize("configured", [False, True])
def test_policy_cli_explicit_or_legacy_default(tmp_path, monkeypatch, serving_profile, configured):
    """The selected policy reaches the same controller authorization checks."""
    selected = tmp_path / "policy.json" if configured else pilot.ROOT / "configs/runpod/pilot.json"
    profile_path = tmp_path / "profile.json"
    policy = {"authorized": False}
    paths = []
    calls = []

    def read(path):
        paths.append(path)
        return policy if path == selected else serving_profile

    # Preserve the controller's independent inputs in the CLI test double.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def run(client, state, selected_policy, profile, project_env, perform):
        calls.append((client, state, selected_policy, profile, project_env, perform))
        return 0

    argv = ["runpod_pilot.py", "preflight", "--state-dir", str(tmp_path / "state"), "--profile", str(profile_path)]
    if configured:
        argv += ["--policy", str(selected)]
    monkeypatch.setattr(pilot.sys, "argv", argv)
    monkeypatch.setattr(pilot.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(pilot, "read_json", read)
    monkeypatch.setattr(pilot, "load_api_key", lambda _path: "synthetic-key")
    monkeypatch.setattr(pilot, "RestClient", lambda _key: "fake-client")
    monkeypatch.setattr(pilot, "run_campaign", run)
    assert pilot.main() == 0
    assert paths == [selected, profile_path]
    assert calls[0][2] is policy
    assert calls[0][3] is serving_profile
    assert calls[0][-1] is False
