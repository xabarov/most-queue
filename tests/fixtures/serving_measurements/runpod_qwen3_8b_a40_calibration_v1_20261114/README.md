# Synthetic Qwen3-8B serving calibration v1

Self-generated on 2026-11-14 with Qwen/Qwen3-8B on one NVIDIA A40.
All 721 requests succeeded across 90 sequential, drained series:

| Split | Requests | Purpose |
| --- | ---: | --- |
| Warmup | 21 | Nine isolated cells and three four-request bursts |
| Training | 27 | Three repetitions of nine isolated input/output cells |
| Isolated held-out validation | 9 | Fourth repetition of the nine cells |
| Burst held-out validation | 108 | Two repetitions, nine cells, concurrency 2 and 4 |
| Open-loop held-out validation | 556 | Six initially empty 180-second series |

The cells cross input lengths 128/1024/4096 with output lengths 32/128/512.
Open-loop requests use input 1024 and output 128 at prescribed exponential
arrival rates 0.2/0.5/0.8 requests per second, with seeds 54101 and 54102.
The server permits four simultaneous sequences. The workload seed is 2026111402.

`plan.json` preserves the complete predetermined schedule and offered status;
`summary.json` and `series.json` preserve actual completion counts, series
boundaries and outcomes. `analysis-plan.md` is a byte-identical snapshot of the
[original analysis plan](../../../../docs/research/serving-calibration-v1-plan.md);
relative links inside that snapshot retain the original document's location.
`requests.jsonl.gz` preserves original request JSONL bytes, including raw SSE
payloads, actual token usage, server metrics, client clocks, seeds and prompt
SHA256. `server_timestamps.jsonl.gz` preserves original engine trace bytes.
Both gzip files use deterministic metadata (`mtime=0`). Synthetic prompt token
IDs reproduce from `random.Random(seed).randint(1000, 20000)` repeated
`input_tokens` times; hashes use UTF-8 JSON with separators `(',', ':')`.

The pinned vLLM `RequestState.make_request_output` observational hook records
final `RequestStateStats` timestamps and counts with one append per completed
request. It leaves scheduling, statistics and returned output objects unchanged;
its instrumentation overhead was not isolated in a separate measurement.
Each client record joins the exact engine request ID ending in `-0` and checks
token counts, absence of preemptions, timestamp order and metric differences.
Engine queued/scheduled/first-token/last-token timestamps are monotonic;
`arrival_time` is separately labelled wall clock. Client TTFT runs from the
child's start to first nonempty SSE content over loopback HTTP. Derived server
TTFT starts at engine queue entry; raw `metrics.time_to_first_token_ms` starts
at scheduling. Raw `metrics.queue_time_ms` retains the preceding queue wait.
Text chunk counts are not token counts.

`profile.json` pins model revision
`b968826d9c46dd6066d109eabc6255188de91218` and the vLLM 0.32.0 image
`vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9`,
including BF16, eager execution, disabled prefix caching/chunked prefill and
the remaining server arguments. `runtime.json` records NVIDIA A40, CUDA 13.0,
PyTorch 2.13.0+cu130 and vLLM 0.32.0. No model weights or credentials are included.
The synthetic dataset is distributed under the copied MIT `LICENSE`.

The [results report](../../../../docs/research/serving-calibration-v1-results.md)
and [analysis JSON](../../../../docs/research/serving-calibration-v1-results.json)
report the frozen gates as passed for isolated validation and all six prescribed
arrival series. Isolated median absolute relative errors are 6.099% for client
TTFT and 0.348% for TPOT. Across the six loaded series, mean client TTFT error
is at most 17.417%, mean server TPOT error at most 1.163%, and normalized mean
queue error at most 19.329%. Queue normalization is
`abs(predicted_mean - observed_mean) / max(observed_mean, 0.05 seconds)`.
At the rarest load the actual queue is about 24–26 microseconds while prediction
is 7.5–9.2 milliseconds: passing the 50 ms floor criterion does not imply a
small ordinary relative queue error.

Burst validation has no acceptance gate. Its maximum mean client TTFT error is
35.387% and maximum mean server TPOT error is 28.147%, an explicit limitation
of the surrogate under bursts. This experiment does not establish universal
vLLM accuracy, stationarity, p99 guarantees or multi-GPU generalization.
The manifest's `analysis_correction` records an earlier mistaken claim of rank
deficiency about the older pilot: its design actually has rank four. This
campaign's method, splits, schedule and criteria were unchanged.

Measurements took 1726.9179 seconds. The allocation estimate is 1876 seconds
and $0.313247; provider billing was still pending with zero returned records.
Cleanup was confirmed by a fresh provider absence read before packaging.
These are historical measurements; reproducing the offline analysis allocates
no resources.

From the repository root, use this source checkout (the helpers are newer than
PyPI 2.11):

```sh
fixture=tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_calibration_v1_20261114
PYTHONPATH=. python examples/runpod_serving_calibration.py "$fixture/requests.jsonl.gz" \
  --measurement-profile runpod-qwen3-8b-a40-timestamps-calibration-v1-20261114 \
  --plan "$fixture/plan.json" --summary "$fixture/summary.json" --output report.json
python docs/figures/plot_runpod_calibration.py report.json --output-dir /tmp/figures
```

`manifest.json` preserves the source hashes frozen before measurement, including
the observational wrapper sources, and separately records the analysis sources
used by the current checkout. It hashes every fixture file except itself and
records both decompressed trace hashes. From this fixture directory, verify
the packaged checksums and reproduce the first synthetic prompt:

```sh
python - <<'PYTHON'
import gzip, hashlib, json, pathlib, random
p = pathlib.Path('.')
m = json.loads((p / 'manifest.json').read_text())
for name, expected in m['sha256'].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
for name, key in [('requests', 'uncompressed_requests'),
                  ('server_timestamps', 'uncompressed_server_timestamps')]:
    raw = gzip.decompress((p / (name + '.jsonl.gz')).read_bytes())
    assert len(raw) == m[key + '_bytes']
    assert hashlib.sha256(raw).hexdigest() == m[key + '_sha256']
rows = [json.loads(line) for line in gzip.decompress((p / 'requests.jsonl.gz').read_bytes()).splitlines()]
r = rows[0]
g = random.Random(r['seed'])
ids = [g.randint(1000, 20000) for _ in range(r['input_tokens'])]
assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == r['prompt_sha256']
assert len(rows) == 721 and all(row['success'] for row in rows)
print(len(rows), 'verified records; first prompt reproduced')
PYTHON
```
