# Interrupted operational attempt — excluded from accuracy validation

This first paid attempt ended after an error reading the Runpod API response
(`ApiError`, HTTP status unavailable). The exact transport/parser cause was
not retained by the API client. The controller exported a live snapshot and
terminated its exact owned pod; a subsequent provider read confirmed absence.
The cumulative ledger charged 1256 seconds and an estimated $0.209722 including
the configured container-disk rate. No new model was fitted on this snapshot.

The snapshot is intentionally preserved unchanged and is NOT a complete
validation cohort: requests.jsonl.gz contains 361 completed successful records,
server_timestamps.jsonl.gz contains 362, and the producer summary says 362.
The workload was still writing during export and never finalized its plan or
summary. The actual number of offered/in-flight requests at interruption cannot
be reconstructed exactly from this snapshot. Manifest counts describe exported
records only. Remaining planned requests must not be mislabeled as observed
failures or used to shrink an SLO denominator. The complete-plan auditor must
reject this attempt. A second attempt uses the same frozen scientific plan;
the operational correction only retries safe status reads within existing bounds.

The copied analysis-plan.md retains its pre-authorization freeze-time status.

# Synthetic Qwen3-8B reservation-v2 validation

Measurements from 2026-11-14 are held out from **both fixed isolated-training
fits** in `frozen-fit.json`; this packaging step performs no fitting.
The prespecified method is preserved in `analysis-plan.md`, with the source
freeze in `scientific-freeze.json`. No accuracy or cleanup conclusion is made here.

Raw record audit counts: planned 690, completed 361,
offered 361, successful 361, failed 0.
`manifest.json` records raw status/split/mode counts and exact file SHA256 values.
`plan.json`, `series.json` and `summary.json` retain the planned schedule,
actual series outcomes and the producer's original completeness fields.
All failures are retained; none are filtered from `requests.jsonl.gz`.
The standalone completeness auditor must separately verify the full identities,
counts, offered status and summary consistency against the generated plan.

Prompts use synthetic token IDs, not user-provided text. Request seeds, lengths,
prompt SHA256, raw SSE responses, token usage, metrics and client clocks are
preserved. `server_timestamps.jsonl.gz` retains the exact engine trace. Both gzip
files use deterministic metadata (mtime zero, no original filename).
No logs, model weights, environment files, credentials, account details,
authorization files or other private controller state are packaged.

Exact recorded hardware/software: GPU `NVIDIA A40`, memory
`47699722240` bytes, CUDA `13.0`, PyTorch
`2.13.0+cu130`, vLLM `0.32.0`. `runtime.json` retains those exact
fields; `profile.json` pins model revision, image and server flags.
The dataset uses the repository owner's copied MIT `LICENSE`.

Offline reproduction from the repository root (allocates no resources):

```sh
fixture=tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_reservation_v2_interrupted_20261114
PYTHONPATH=. .venv/bin/python examples/serving_reservation_slo.py \
  --evaluation-fixture "$fixture" --frozen-fit "$fixture/frozen-fit.json" \
  --output /tmp/reservation-v2-report.json
.venv/bin/python docs/figures/plot_reservation_validation.py \
  --report /tmp/reservation-v2-report.json --output /tmp/reservation-v2.png
```

From the fixture directory, verify packaged file hashes:

```sh
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path('.')
manifest = json.loads((root / 'manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('Packaged file hashes verified; accuracy and provider cleanup not checked.')
PY
```
