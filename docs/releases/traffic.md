# Manual GitHub traffic snapshots

Run the read-only snapshot script manually, for example once a week:

```bash
python tools/github_traffic_snapshot.py
python tools/github_traffic_snapshot.py --repo xabarov/most-queue --output-dir ~/private/most-queue-traffic
```

The default destination is `~/.local/share/most-queue/traffic`, outside the
checkout. Newly created directories are private (mode `0700`); snapshots use
mode `0600`. Keep these local files private. No scheduler or publishing step is
installed. The script uses only the Python standard library.

Credentials come from `GH_TOKEN`, then `GITHUB_TOKEN`, then the configured Git
credential helper for `github.com`. Git terminal prompting is disabled. The
script does not load `.env` files, print credentials or HTTP headers, or pass a
token as a command-line argument. Without credentials it attempts anonymous
requests, which normally cannot retrieve repository traffic.

Only five GET endpoints are queried: repository metadata, traffic views, clones,
popular referrers, and popular paths. Each timestamped JSON file contains its UTC
collection time, repository, stars, forks, GitHub's `open_issues_count` (which
includes pull requests), and available traffic data. GitHub exposes a rolling
14-day traffic window; weekly manual snapshots preserve overlapping observations,
so totals from different snapshots should not be added together.

Traffic requires repository write access; fine-grained tokens need repository
Administration read permission. Consult the official
[GitHub traffic API documentation](https://docs.github.com/en/rest/metrics/traffic).
Partial results are still saved. Failures appear under `errors` with HTTP status
codes and fixed messages, without response bodies or credentials. The script
prints a short status to stderr, uses a 15-second request timeout, retries a
network failure once, and does not retry HTTP failures. Redirects are rejected.

Exit codes: `0` for a complete snapshot, `1` for a saved partial snapshot, and
`2` for an invalid command or inability to save the file.
