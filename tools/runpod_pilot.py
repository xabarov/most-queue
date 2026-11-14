"""Run one bounded serving pilot; every paid allocation has detached cleanup."""

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import tarfile
import time
import urllib.parse
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Direct execution needs the repository on sys.path before local imports.
from tools.runpod_budget import assess_plan  # pylint: disable=wrong-import-position
from tools.runpod_lifecycle import (  # pylint: disable=wrong-import-position
    ApiError,
    LifecycleError,
    RestClient,
    atomic_json,
    cleanup_exact,
    exact_pod,
    list_pods,
    load_api_key,
    process_start_ticks,
    read_json,
    start_watchdog,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path.home() / ".local/state/most-queue/runpod/serving-pilot-2026-11-14"
REMOTE = "/tmp/most-queue-pilot"
POD_FIELDS = ("id", "name", "image", "cost", "gpu", "cloud", "disk", "mounts", "cudaVersion", "dataCenterId")


def event(message, **values):
    """Print only deliberate non-secret operational facts."""
    print(json.dumps({"event": message, **values}), flush=True)


def money_rate(value):
    """Reject absent, negative, boolean or non-finite hourly rates."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise LifecycleError("Invalid provider hourly rate")
    return float(value)


def reconcile_ledger(state):
    """Charge every allocation through confirmed absence, including failed runs."""
    ledger = {"spent_usd": 0.0, "allocated_seconds": 0, "allocations": 0, "active_pod_ids": []}
    for directory in sorted((state / "attempts").glob("*")):
        if not (directory / "create-started.json").exists():
            continue
        ledger["allocations"] += 1
        attempt = read_json(directory / "attempt.json")
        receipt = read_json(directory / "watchdog.json") if (directory / "watchdog.json").exists() else {}
        try:
            pod = read_json(directory / "pod.json") if (directory / "pod.json").exists() else {}
        except LifecycleError:
            pod = read_json(directory / "recovered-pod.json")
        if not pod and (directory / "recovered-pod.json").exists():
            pod = read_json(directory / "recovered-pod.json")
        if receipt.get("status") != "cleanup_confirmed":
            raise LifecycleError("Unresolved previous allocation; recover exact intent before any new create")
        if not pod:
            if receipt.get("pod_id"):
                raise LifecycleError("Cleanup receipt has no durable pod identity")
            continue
        if pod.get("id") != receipt.get("pod_id"):
            raise LifecycleError("Cleanup and accounting pod identities differ")
        finished = receipt.get("confirmed_absent_epoch")
        if type(finished) not in (int, float) or not math.isfinite(finished):
            raise LifecycleError("Missing confirmed absence timestamp")
        elapsed = math.ceil(max(0, finished - attempt["created_epoch"]))
        rate = max(money_rate(pod.get("cost")), attempt["plan"]["compute_usd_per_hour"])
        rate += attempt["plan"]["storage_usd_per_hour"]
        ledger["allocated_seconds"] += elapsed
        ledger["spent_usd"] += elapsed * rate / 3600
    # Round upward so an exported ledger does not silently gain fractional budget.
    ledger["spent_usd"] = math.ceil(ledger["spent_usd"] * 1_000_000) / 1_000_000
    atomic_json(state / "ledger.json", ledger)
    return ledger


def choose_plan(client, policy, profile, ledger):
    """Read live capacity and price using the owner's selected account."""
    for gpu_id in policy["allowed_gpu_ids"]:
        query = urllib.parse.urlencode(
            {
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": "SECURE",
                "count": 1,
                "minCudaVersion": profile["min_cuda_version"],
            }
        )
        offer = client.request("GET", "/catalog/gpus/" + urllib.parse.quote(gpu_id, safe="") + "?" + query)
        if offer.get("availability") not in ("LOW", "MEDIUM", "HIGH"):
            continue
        plan = {
            "gpu_id": gpu_id,
            "cloud": "SECURE",
            "gpu_count": 1,
            "compute_usd_per_hour": money_rate(offer.get("price", {}).get("secure")),
            "storage_usd_per_hour": profile["storage_usd_per_hour"],
            "work_seconds": profile["work_seconds"],
            "container_disk_gb": profile["container_disk_gb"],
            "persistent_disk_gb": 0,
            "network_volume_ids": [],
        }
        decision = assess_plan(policy, plan, ledger)
        if decision["allowed"]:
            return plan, offer, decision
    raise LifecycleError("No available approved GPU fits the remaining campaign limits")


def validate_allocated(pod, attempt, policy, ledger, require_cuda=False):
    """Reject a changed allocation before installing or running the workload."""
    requested = attempt["create_body"]
    for field in ("name", "image", "cloud", "disk"):
        if pod.get(field) != requested[field]:
            raise LifecycleError("Allocated pod differs from requested " + field)
    if pod.get("mounts") not in ({}, None) or pod.get("gpu", {}).get("id") != requested["gpu"]["id"]:
        raise LifecycleError("Unexpected storage or GPU allocated")
    if pod.get("gpu", {}).get("count") != 1:
        raise LifecycleError("Unexpected GPU count")
    actual = dict(attempt["plan"], compute_usd_per_hour=money_rate(pod.get("cost")))
    decision = assess_plan(policy, actual, ledger)
    if not decision["allowed"]:
        raise LifecycleError("Allocated rate exceeds remaining limits: " + "; ".join(decision["reasons"]))
    cuda = pod.get("cudaVersion")
    floor = attempt["profile"]["min_cuda_version"]
    if cuda is None and not require_cuda:
        return
    try:
        compatible = isinstance(cuda, str) and tuple(map(int, cuda.split("."))) >= tuple(map(int, floor.split(".")))
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        raise LifecycleError("Allocated host CUDA is incompatible")


class Pilot:
    """Own the exact pod until its absence has been independently observed."""

    def __init__(self, client, directory, attempt):
        self.client = client
        self.directory = directory
        self.attempt = attempt
        self.pod = None
        self.supervisor = None
        self.progress = None
        self.last_progress = time.time()
        self.stage_name = "creating"
        self.stage_deadline = min(time.time() + 120, attempt["work_deadline_epoch"])

    def set_stage(self, name, seconds):
        """Bound each stage inside the existing work window."""
        self.stage_name = name
        self.stage_deadline = min(time.time() + seconds, self.attempt["work_deadline_epoch"])
        self.last_progress = time.time()
        self.progress = None
        self.save_stage()
        event("stage", stage=name)

    def save_stage(self):
        """Persist only observed progress, never extend the fixed stage deadline."""
        atomic_json(
            self.directory / "stage.json",
            {
                "stage": self.stage_name,
                "deadline_epoch": self.stage_deadline,
                "progress_deadline_epoch": min(self.last_progress + 600, self.stage_deadline),
            },
        )

    def check(self):
        """Detect lost supervision or exhaustion before each remote operation."""
        if time.time() >= min(self.stage_deadline, self.last_progress + 600):
            raise LifecycleError("Stage deadline or progress deadline reached")
        if self.supervisor is None or self.supervisor.poll() is not None:
            raise LifecycleError("Cleanup supervisor exited")
        receipt = read_json(self.directory / "watchdog.json")
        if receipt.get("status") != "monitoring":
            raise LifecycleError("Cleanup supervisor is no longer monitoring")
        if time.time() - (self.directory / "watchdog.json").stat().st_mtime > 45:
            raise LifecycleError("Cleanup supervisor heartbeat is stale")

    def remote(self, command, timeout=20):
        """Run an argv command via the provider, without SSH or exposed ports."""
        self.check()
        result = self.client.request(
            "POST",
            "/pods/" + self.pod["id"] + "/exec",
            {"command": command, "timeoutSeconds": min(timeout, 25)},
            timeout=timeout + 10,
        )
        if result.get("timedOut") or result.get("truncated") or result.get("exitCode") != 0:
            atomic_json(self.directory / "last-exec-error.json", result)
            raise LifecycleError("Remote command failed; evidence saved locally")
        return result["stdout"]

    def wait_watchdog(self, require_pod=False):
        """Require a running detached supervisor before any billable operation."""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.supervisor.poll() is not None:
                raise LifecycleError("Cleanup supervisor failed to start")
            path = self.directory / "watchdog.json"
            if path.exists():
                receipt = read_json(path)
                if receipt.get("status") == "monitoring":
                    if not require_pod or receipt.get("pod_id") == self.pod["id"]:
                        return
            time.sleep(0.2)
        raise LifecycleError("Cleanup supervisor did not acknowledge ownership")

    def allocate(self, project_env):
        """Persist intent and supervision before the single non-retried POST."""
        self.save_stage()
        self.supervisor = start_watchdog(self.directory, project_env)
        self.wait_watchdog()
        atomic_json(self.directory / "create-started.json", {"epoch": time.time()})
        try:
            pod = self.client.request("POST", "/pods", self.attempt["create_body"], timeout=25)
        except ApiError as error:
            outcome = "rejected" if error.status in (400, 401, 403, 404, 422) else "ambiguous"
            atomic_json(self.directory / "create-result.json", {"outcome": outcome, "status": error.status})
            raise
        if not isinstance(pod, dict) or not pod.get("id"):
            atomic_json(self.directory / "create-result.json", {"outcome": "ambiguous", "status": None})
            raise LifecycleError("Create response did not contain a durable identity")
        self.pod = {key: pod.get(key) for key in POD_FIELDS}
        atomic_json(self.directory / "pod.json", self.pod)
        atomic_json(self.directory / "create-result.json", {"outcome": "created", "status": 201})
        self.wait_watchdog(require_pod=True)
        event("allocated", pod_id=self.pod["id"], compute_usd_per_hour=self.pod.get("cost"))

    def wait_running(self):
        """Wait at most ten minutes for container readiness."""
        self.set_stage("infrastructure_ready", 600)
        while True:
            self.check()
            pod = exact_pod(self.client, self.pod["id"], self.attempt["name"], self.attempt["image"])
            if pod is None:
                raise LifecycleError("Pod disappeared before readiness")
            if pod.get("status") == "RUNNING":
                try:
                    # RUNNING describes the allocation; image pull/container start
                    # can still be in progress. This harmless probe is retryable.
                    if self.remote(["python3", "-c", "print('ready')"], timeout=10).strip() != "ready":
                        raise LifecycleError("Container readiness probe returned unexpected output")
                    return pod
                except ApiError as error:
                    if error.status not in (None, 409, 429, 503):
                        raise
                    time.sleep(error.retry_after if error.retry_after is not None else 5)
                    continue
            if pod.get("status") in ("EXITED", "FAILED", "ERROR"):
                raise LifecycleError("Pod failed before readiness")
            time.sleep(5)

    def upload_workload(self):
        """Upload only the public measurement program, with a checksum."""
        raw = (ROOT / "tools/runpod_workload.py").read_bytes()
        encoded = base64.b64encode(raw).decode()
        script = (
            "import base64,hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True,exist_ok=True); b=base64.b64decode(''.join(sys.argv[3:])); "
            "assert hashlib.sha256(b).hexdigest()==sys.argv[2]; p.write_bytes(b)"
        )
        self.remote(
            ["python3", "-c", script, REMOTE + "/workload.py", hashlib.sha256(raw).hexdigest()]
            + [encoded[i : i + 4000] for i in range(0, len(encoded), 4000)]
        )

    def start_server(self):
        """Start pinned vLLM on loopback with no externally accessible endpoint."""
        self.set_stage("model_load_and_warmup", 1200)
        self.upload_workload()
        profile = self.attempt["profile"]
        command = [
            "vllm",
            "serve",
            profile["model"],
            "--revision",
            profile["revision"],
            "--tokenizer-revision",
            profile["revision"],
            "--seed",
            str(profile["seed"]),
        ]
        command += profile["server_args"]
        script = (
            "import json,pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); "
            "p.mkdir(parents=True,exist_ok=True); f=(p/'server.log').open('wb'); "
            "s=subprocess.Popen(json.loads(sys.argv[2]),stdin=subprocess.DEVNULL,stdout=f,"
            "stderr=subprocess.STDOUT,start_new_session=True); "
            "(p/'server.pid').write_text(str(s.pid)); print(s.pid)"
        )
        self.remote(["python3", "-c", script, REMOTE, json.dumps(command)])
        while True:
            status = self.probe()
            if status["healthy"]:
                event("server_ready")
                return
            time.sleep(5)

    def probe(self):
        """Collect bounded progress evidence without exposing environment values."""
        script = """import hashlib,json,pathlib,urllib.request
p=pathlib.Path('/tmp/most-queue-pilot'); log=p/'server.log'
tail=''
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-6000)); tail=f.read().decode(errors='replace')
healthy=False
try:
 with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r: healthy=r.status==200
except Exception: pass
progress=p/'measurements/progress.json'
print(json.dumps({'healthy':healthy,'log_tail':tail,'log_size':log.stat().st_size if log.exists() else 0,
 'progress':json.loads(progress.read_text()) if progress.exists() else None,
 'workload_exit':(p/'workload.exit').read_text().strip() if (p/'workload.exit').exists() else None}))
"""
        data = json.loads(self.remote(["python3", "-c", script], timeout=10))
        atomic_json(self.directory / "remote-progress.json", data)
        # During measurements only completed requests count as progress; logs cannot mask a stalled workload.
        observed = data["progress"] if self.stage_name == "measurements" else data["log_size"]
        if observed != self.progress:
            self.progress = observed
            self.last_progress = time.time()
            self.save_stage()
        return data

    def measure(self):
        """Collect warmup and fixed burst batches, preserving every result."""
        self.set_stage("measurements", 950)
        profile = self.attempt["profile"]
        command = [
            "python3",
            REMOTE + "/workload.py",
            REMOTE + "/measurements",
            "--model",
            profile["model"],
            "--seed",
            str(profile["seed"]),
        ]
        wrapper = (
            "import json,pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); "
            "f=(p/'workload.log').open('wb'); r=subprocess.run(json.loads(sys.argv[2]),"
            "stdout=f,stderr=subprocess.STDOUT,timeout=930); (p/'workload.exit').write_text(str(r.returncode))"
        )
        launcher = (
            "import subprocess,sys; subprocess.Popen(['python3','-c',sys.argv[1],sys.argv[2],sys.argv[3]],"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)"
        )
        self.remote(["python3", "-c", launcher, wrapper, REMOTE, json.dumps(command)])
        while True:
            data = self.probe()
            if data["workload_exit"] is not None:
                return data["workload_exit"] == "0"
            time.sleep(5)

    def export(self):
        """Download a bounded checksum-verified archive before terminating."""
        self.set_stage("export", 180)
        script = """import hashlib,json,pathlib,tarfile
p=pathlib.Path('/tmp/most-queue-pilot'); a=p.parent/'most-queue-pilot.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for f in sorted(p.rglob('*')):
  if f.is_file() and not f.is_symlink(): t.add(f,arcname=str(f.relative_to(p)),recursive=False)
print(json.dumps({'size':a.stat().st_size,'sha256':hashlib.sha256(a.read_bytes()).hexdigest()}))
"""
        metadata = json.loads(self.remote(["python3", "-c", script]))
        if not 0 < metadata["size"] <= 50 * 1024 * 1024:
            raise LifecycleError("Export exceeds bounded artifact size")
        target = self.directory / "artifacts.tar.gz"
        with target.open("wb") as handle:
            for offset in range(0, metadata["size"], 128 * 1024):
                script = (
                    "import base64,sys; f=open('/tmp/most-queue-pilot.tar.gz','rb'); "
                    "f.seek(int(sys.argv[1])); print(base64.b64encode(f.read(131072)).decode())"
                )
                handle.write(
                    base64.b64decode(self.remote(["python3", "-c", script, str(offset)]).strip(), validate=True)
                )
            handle.flush()
            os.fsync(handle.fileno())
        if (
            target.stat().st_size != metadata["size"]
            or hashlib.sha256(target.read_bytes()).hexdigest() != metadata["sha256"]
        ):
            raise LifecycleError("Export checksum or length mismatch")
        extract_artifacts(target, self.directory / "artifacts")
        atomic_json(self.directory / "export.json", metadata)
        event("export_verified", bytes=metadata["size"])

    def cleanup(self, reason):
        """Request detached cleanup and provide a controller fallback."""
        if not (self.directory / "create-started.json").exists():
            atomic_json(self.directory / "create-result.json", {"outcome": "not_sent", "status": None})
            if self.supervisor is None or self.supervisor.poll() is not None:
                receipt = {
                    "status": "cleanup_confirmed",
                    "pod_id": None,
                    "confirmed_absent_epoch": time.time(),
                    "reason": "create_was_never_sent",
                }
                atomic_json(self.directory / "watchdog.json", receipt)
                return receipt
        atomic_json(self.directory / "cleanup.json", {"requested": True, "reason": reason})
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            receipt_path = self.directory / "watchdog.json"
            receipt = read_json(receipt_path) if receipt_path.exists() else {}
            if receipt.get("status") == "cleanup_confirmed":
                event("cleanup_confirmed", pod_id=receipt.get("pod_id"))
                return receipt
            stale = receipt_path.exists() and time.time() - receipt_path.stat().st_mtime > 60
            if self.supervisor is None or self.supervisor.poll() is not None or stale:
                if self.pod:
                    receipt = cleanup_exact(self.client, self.pod["id"], self.attempt["name"], self.attempt["image"])
                    receipt["pod_id"] = self.pod["id"]
                    atomic_json(receipt_path, receipt)
                    event("cleanup_confirmed", pod_id=self.pod["id"], fallback=True)
                    return receipt
                raise LifecycleError("Supervisor failed with unresolved create; exact intent recovery required")
            time.sleep(2)
        raise LifecycleError("Cleanup not yet confirmed; detached supervisor remains active")


def extract_artifacts(archive, destination):
    """Extract only regular files and directories contained in the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        if sum(member.size for member in members) > 100 * 1024 * 1024:
            raise LifecycleError("Expanded artifacts exceed size bound")
        for member in members:
            path = destination / member.name
            if not (member.isfile() or member.isdir()) or not path.resolve().is_relative_to(destination.resolve()):
                raise LifecycleError("Unsafe archive member")
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, path.open("wb") as target:
                    target.write(source.read())


# Explicit inputs keep the persisted spending decision reviewable.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def prepare_attempt(state, policy, profile, plan, offer, ledger):
    """Write the unique intent, exact resource request and immutable deadlines."""
    run_id = str(uuid.uuid4())
    directory = state / "attempts" / run_id
    name = "most-queue-pilot-" + run_id
    body = {
        "name": name,
        "image": profile["image"],
        "cloud": plan["cloud"],
        "gpu": {"id": plan["gpu_id"], "count": 1, "minCudaVersion": profile["min_cuda_version"]},
        "disk": plan["container_disk_gb"],
        "mounts": {},
        "ports": [],
        "entrypoint": ["/bin/sleep"],
        "cmd": ["infinity"],
        "env": {
            "HF_HOME": "/tmp/most-queue-hf",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        },
    }
    now = time.time()
    # Use the authorized hourly ceiling for deadlines even if the returned price exceeds the quote.
    reserve = policy["cleanup_reserve_seconds"]
    total_seconds = min(
        plan["work_seconds"] + reserve,
        policy["max_campaign_seconds"] - ledger["allocated_seconds"],
        math.floor((policy["campaign_budget_usd"] - ledger["spent_usd"]) / policy["max_total_usd_per_hour"] * 3600),
    )
    if total_seconds <= reserve:
        raise LifecycleError("No work time remains after cleanup reserve")
    attempt = {
        "schema_version": 1,
        "name": name,
        "image": profile["image"],
        "profile": profile,
        "owner_pid": os.getpid(),
        "owner_start_ticks": process_start_ticks(os.getpid()),
        "created_epoch": now,
        "work_deadline_epoch": now + total_seconds - reserve,
        "hard_deadline_epoch": now + total_seconds,
        "create_body": body,
        "plan": plan,
        "ledger_before": ledger,
    }
    atomic_json(directory / "attempt.json", attempt)
    atomic_json(directory / "catalog.json", offer)
    return directory, attempt


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def run_campaign(client, state, policy, profile, project_env, perform=False):
    """Preflight without side effects on Runpod; allocate only in explicit run mode."""
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LifecycleError("Another controller owns this campaign") from error
        account = client.request("GET", "/account")
        if not isinstance(account, dict) or not account.get("id"):
            raise LifecycleError("Cannot verify selected account")
        meta_path = state / "campaign.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta.get("closed"):
                raise LifecycleError("This authorized campaign is already complete")
            if meta.get("policy") != policy or meta.get("account_id") != account["id"]:
                raise LifecycleError("Campaign policy or account changed; reconcile before continuing")
        else:
            meta = {
                "campaign_id": profile["campaign_id"],
                "policy": policy,
                "account_id": account["id"],
                "closed": False,
            }
            atomic_json(meta_path, meta)
        ledger = reconcile_ledger(state)
        if any(pod.get("name", "").startswith("most-queue-pilot-") for pod in list_pods(client)):
            raise LifecycleError("An existing Most-Queue allocation needs reconciliation first")
        plan, offer, decision = choose_plan(client, policy, profile, ledger)
        atomic_json(
            state / "preflight.json", {"plan": plan, "decision": decision, "profile": profile, "epoch": time.time()}
        )
        event(
            "preflight_allowed",
            gpu=plan["gpu_id"],
            compute_usd_per_hour=plan["compute_usd_per_hour"],
            projected_campaign_usd=decision["projected_total_usd"],
            paid_create=perform,
        )
        if not perform:
            return 0
        directory, attempt = prepare_attempt(state, policy, profile, plan, offer, ledger)
        pilot = Pilot(client, directory, attempt)
        outcome = {"successful": False, "error": None}
        try:
            pilot.allocate(project_env)
            validate_allocated(pilot.pod, attempt, policy, ledger)
            pod = pilot.wait_running()
            validate_allocated(pod, attempt, policy, ledger, require_cuda=True)
            runtime = pilot.remote(
                [
                    "python3",
                    "-c",
                    "import json,torch,vllm; print(json.dumps({'torch':torch.__version__,"
                    "'vllm':vllm.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),"
                    "'memory_bytes':torch.cuda.get_device_properties(0).total_memory}))",
                ],
                timeout=25,
            )
            atomic_json(directory / "runtime.json", json.loads(runtime))
            pilot.start_server()
            outcome["successful"] = pilot.measure()
            pilot.export()
        # Every workload failure must reach exact-resource cleanup.
        except (Exception, KeyboardInterrupt) as error:  # pylint: disable=broad-exception-caught
            outcome["successful"] = False
            outcome["error"] = type(error).__name__ + ": " + str(error)
            event("pilot_failed", error=outcome["error"])
            if pilot.pod:
                try:
                    pilot.export()
                except (Exception, KeyboardInterrupt):  # pylint: disable=broad-exception-caught
                    outcome["export_incomplete"] = True
        finally:
            atomic_json(directory / "outcome.json", outcome)
            pilot.cleanup("finished" if outcome["successful"] else "failed_or_interrupted")
        ledger = reconcile_ledger(state)
        event("campaign_accounted", **ledger)
        if outcome["successful"] and not outcome.get("export_incomplete"):
            meta.update(closed=True, completed_epoch=time.time())
            atomic_json(meta_path, meta)
            return 0
        return 1


def stop_campaign(state):
    """Signal only owned active attempts; their supervisors perform exact cleanup."""
    for directory in (state / "attempts").glob("*"):
        receipt_path = directory / "watchdog.json"
        receipt = read_json(receipt_path) if receipt_path.exists() else {}
        if receipt.get("status") != "cleanup_confirmed":
            atomic_json(directory / "cleanup.json", {"requested": True, "reason": "explicit_stop"})
    event("cleanup_requested")


def main():
    """Expose explicit preflight/run/stop/status commands with no secret arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "stop", "status"))
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--project-env", type=Path, default=ROOT / ".env")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/runpod/serving-profile.json")
    args = parser.parse_args()
    if args.command == "stop":
        stop_campaign(args.state_dir)
        return 0
    if args.command == "status":
        event("ledger", **reconcile_ledger(args.state_dir))
        return 0

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        policy = read_json(ROOT / "configs/runpod/pilot.json")
        profile = read_json(args.profile)
        if "@sha256:" not in profile["image"]:
            raise LifecycleError("Container image must be pinned by digest")
        return run_campaign(
            RestClient(load_api_key(args.project_env)),
            args.state_dir,
            policy,
            profile,
            args.project_env.resolve(),
            perform=args.command == "run",
        )
    except (Exception, KeyboardInterrupt) as error:  # pylint: disable=broad-exception-caught
        event("controller_error", error=type(error).__name__ + ": " + str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
