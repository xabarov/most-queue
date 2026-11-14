# Frontend v1 diagnostic fixture

These measurements use synthetic prompts and paired instrumentation on/off controls.
The fixture preserves raw measurements and failures. Counts describe records only;
completion, model accuracy, cost, and provider cleanup require separate evidence.
The scope is diagnostic: no fitting and no v3 validation. Measurement code commit:
`64bbecf`.

Client, frontend, and engine timestamps use monotonic clocks. Interpret differences
only as prescribed by the scientific freeze, analysis plan, and clock checks in the
repository auditor; do not infer cross-host clock alignment from raw timestamps.

`analysis-plan.md` is an archival byte-exact copy. Its relative links resolve against
the canonical repository document `docs/research/serving-frontend-v1-plan.md`, not
this fixture directory. Its historical authorization/status text is archival.
Do not rewrite this hashed copy to repair links or update campaign state.

`producer-manifest.json` preserves the producer's original measurement manifest.
The top-level `manifest.json` inventories this package; it excludes itself.
The runtime file contains only gpu, torch, vllm, cuda, and memory_bytes.

## Reproduce the audit in a separate scratch directory

The current repository auditor expects an uncompressed measurements directory and
both timestamp traces in its parent. Keep this fixture unchanged. From its root:

```bash
fixture="$PWD"
scratch=$(mktemp -d)
mkdir "$scratch/measurements"
cp "$fixture/plan.json" "$fixture/series.json" "$fixture/summary.json" \
   "$fixture/progress.json" "$scratch/measurements/"
cp "$fixture/producer-manifest.json" "$scratch/measurements/manifest.json"
gzip -dc "$fixture/requests.jsonl.gz" > "$scratch/measurements/requests.jsonl"
gzip -dc "$fixture/server_timestamps.jsonl.gz" > "$scratch/server_timestamps.jsonl"
gzip -dc "$fixture/frontend_timestamps.jsonl.gz" > "$scratch/frontend_timestamps.jsonl"
# Run from the canonical repository checkout (or use its absolute tool path):
python tools/audit_frontend_measurements.py "$scratch/measurements" \
    --output "$scratch/frontend-audit.json"
```

Verify compressed files against the package SHA256 inventory and decompressed
bytes against `traces` in the package manifest before reproducing the audit.


## Completed study and resource accounting

All 613 planned requests completed successfully; the standalone auditor checks
613 engine and 533 frontend records, with no condition failures. This is a
diagnostic study with fixed v2 comparisons, no model fitting or v3 validation.
[Results and limitations](../../../../docs/research/serving-frontend-v1-results.md).

The first export API read failed, but the fallback export was verified by size
and SHA256. The original controller error and shell status are preserved in
`completion.json`; completion is established by the full audit and fresh provider
absence proof. One allocation, 1597 seconds, estimated $0.266660 including disk.
The pod is confirmed absent, no persistent storage retained, campaign closed.
Provider billing still returned no records; these costs are local estimates.

Reproduce the complete descriptive report and plots from the repository root:

```bash
PYTHONPATH=. python examples/serving_frontend_diagnostics.py \
  --output frontend-v1-results.json
python docs/figures/plot_frontend_diagnostics.py --report frontend-v1-results.json
```

The original frozen plan is accessible at
[its canonical location](../../../../docs/research/serving-frontend-v1-plan.md).
The `outside_engine` output field is a residual outside QUEUED-to-first-token;
it does not imply that all of that delay occurs outside the EngineCore process.
