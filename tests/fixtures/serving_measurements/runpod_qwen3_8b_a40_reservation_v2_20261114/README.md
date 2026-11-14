# Synthetic Qwen3-8B reservation-v2 validation

Measurements from 2026-11-14 are held out from **both fixed isolated-training
fits** in `frozen-fit.json`; this packaging step performs no fitting.
The prespecified method is preserved in `analysis-plan.md`, with the source
freeze in `scientific-freeze.json`. The [results report](../../../../docs/research/serving-reservation-v2-results.md)
records all 14 criteria: 13 pass, isolated server-TTFT maximum error fails.
The complete 690-request fixture passes the completeness audit. Exact pod
absence was checked through subsequent provider GET and inventory reads;
`completion.json` retains the sanitized proof and aggregate campaign accounting.

Raw record audit counts: planned 690, completed 690,
offered 690, successful 690, failed 0.
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
fixture=tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_reservation_v2_20261114
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

The archival `analysis-plan.md` predates authorization. The
[operational addendum](../../../../docs/research/serving-reservation-v2-operations-addendum.md)
records approval, two interrupted attempts and the one approved extra allocation.
Neither the scientific files nor fixed fits changed. The interrupted first
snapshot is preserved separately and excluded from accuracy evaluation.

`analysis-plan.md` is a byte-for-byte archival copy. Its relative links are
relative to the original documentation directory; use the
[canonical plan](../../../../docs/research/serving-reservation-v2-plan.md)
to follow them. The archive is not rewritten, preserving its frozen hash.
