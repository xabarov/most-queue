# Synthetic Qwen3-8B serving pilot

Self-generated on 2026-11-14 with Qwen/Qwen3-8B on one NVIDIA A40.
The 81 records contain 3 sequential warmups and 78 measured requests: two
repetitions of input lengths 128/1024/4096 at burst concurrency 1/4/8, each
requesting 128 output tokens. The server allows at most 4 simultaneous sequences.
All records succeeded. This finite-burst smoke measurement does not validate a
steady queueing model.

`requests.jsonl.gz` preserves the original JSONL bytes with deterministic gzip
metadata (`mtime=0`). Records retain raw SSE payloads, reported usage and server
metrics, per-event monotonic client elapsed times, seeds and prompt SHA256.
Synthetic prompt token IDs are reproducible from each record's seed with
`random.Random(seed).randint(1000, 20000)` repeated `input_tokens` times; the
prompt hash uses UTF-8 JSON with separators `(',', ':')`.

`client_ttft_s` measures client arrival to the first nonempty content event over
loopback HTTP. Raw `metrics.time_to_first_token_ms` measures server scheduling
to first generated output; `metrics.queue_time_ms` separately measures scheduler
queue wait. Text chunk counts are not token counts.

`profile.json` freezes model revision and the vLLM 0.32.0 image digest, along
with server arguments. `runtime.json` records the observed GPU/software runtime;
`summary.json` is the exported workload summary. No model weights are included.
The repository MIT license is copied as `LICENSE`.

From this directory, inspect the records, reproduce the first synthetic prompt,
and verify all packaged checksums with Python's standard library:

```sh
python3 - <<'PYTHON'
import gzip, hashlib, json, pathlib, random
p = pathlib.Path('.')
m = json.loads((p / 'manifest.json').read_text())
for name, expected in m['sha256'].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
raw = gzip.decompress((p / 'requests.jsonl.gz').read_bytes())
assert hashlib.sha256(raw).hexdigest() == m['uncompressed_requests_sha256']
rows = [json.loads(line) for line in raw.splitlines()]
r = rows[0]
g = random.Random(r['seed'])
ids = [g.randint(1000, 20000) for _ in range(r['input_tokens'])]
b = json.dumps(ids, separators=(',', ':')).encode()
assert hashlib.sha256(b).hexdigest() == r['prompt_sha256']
print(len(rows), 'verified records; first prompt reproduced')
PYTHON
```
