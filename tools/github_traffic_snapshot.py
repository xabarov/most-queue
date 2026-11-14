#!/usr/bin/env python3
"""Save a manual, read-only GitHub traffic snapshot outside the checkout."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


class NoRedirect(request.HTTPRedirectHandler):
    """Do not forward credentials or request unapproved endpoints on redirects."""

    # Signature is imposed by urllib.request.HTTPRedirectHandler.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def repository(value):
    """Accept only an owner/repository pair, never a URL or query string."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError("expected owner/repository")
    if value.split("/")[1] in {".", ".."}:
        raise argparse.ArgumentTypeError("invalid repository name")
    return value


def get_token(repo):
    """Read environment credentials first, then Git's noninteractive helper."""
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/true", SSH_ASKPASS="/bin/true")
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input=f"protocol=https\nhost=github.com\npath={repo}\n\n",
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return fields.get("password") or None


def fetch(opener, endpoint, token):
    """Return parsed JSON or a fixed, credential-free error; retry network failures once."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "most-queue-traffic-snapshot",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(2):
        try:
            req = request.Request(f"https://api.github.com{endpoint}", headers=headers, method="GET")
            with opener.open(req, timeout=15) as response:
                return json.load(response), None
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            message = {
                401: "authentication failed",
                403: "access forbidden or rate limited; traffic requires repository access",
                404: "repository or traffic unavailable to these credentials",
                429: "rate limit reached",
            }.get(status, "GitHub request failed")
            return None, {"http_status": status, "message": message}
        except (error.URLError, TimeoutError, OSError):
            if attempt == 0:
                time.sleep(1)
                continue
            return None, {"http_status": None, "message": "network request failed after 2 attempts"}
        except (ValueError, UnicodeError):
            return None, {"http_status": None, "message": "invalid JSON response"}
    raise RuntimeError("unreachable")


def redact(value, token):
    """Prevent credential text from being persisted even if returned in API data."""
    if not token:
        return value
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]")
    if isinstance(value, list):
        return [redact(item, token) for item in value]
    if isinstance(value, dict):
        return {redact(key, token): redact(item, token) for key, item in value.items()}
    return value


def main(argv=None):
    """Fetch repository counters and save one private local snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=repository, default="xabarov/most-queue")
    parser.add_argument("--output-dir", type=Path, default=Path("~/.local/share/most-queue/traffic"))
    args = parser.parse_args(argv)
    token = get_token(args.repo)
    if not token:
        print(
            "No GitHub credential found; trying unauthenticated GET requests. Traffic may be unavailable.",
            file=sys.stderr,
        )
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo": args.repo,
        "stars": None,
        "forks": None,
        "open_issues_count": None,
        "traffic": {},
        "errors": {},
    }
    opener = request.build_opener(NoRedirect())
    endpoints = {
        "repository": "",
        "views": "/traffic/views",
        "clones": "/traffic/clones",
        "referrers": "/traffic/popular/referrers",
        "paths": "/traffic/popular/paths",
    }
    for name, suffix in endpoints.items():
        data, failure = fetch(opener, f"/repos/{args.repo}{suffix}", token)
        if failure:
            snapshot["errors"][name] = failure
            status = failure["http_status"]
            label = f"HTTP {status}" if status is not None else "request error"
            print(f"{name}: {label}: {failure['message']}", file=sys.stderr)
        elif name == "repository":
            if not isinstance(data, dict):
                snapshot["errors"][name] = {"http_status": None, "message": "invalid repository response"}
                continue
            snapshot.update(
                stars=data.get("stargazers_count"),
                forks=data.get("forks_count"),
                open_issues_count=data.get("open_issues_count"),
            )
        else:
            snapshot["traffic"][name] = data
    output_dir = args.output_dir.expanduser()
    try:
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = output_dir / f"{args.repo.replace('/', '--')}-{stamp}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(redact(snapshot, token), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except OSError:
        print("Could not save the snapshot; check the output directory and its permissions.", file=sys.stderr)
        return 2
    print(f"Saved snapshot: {path}")
    return 1 if snapshot["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
