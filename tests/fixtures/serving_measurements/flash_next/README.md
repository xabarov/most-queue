# Flash-Next serving measurement fixtures

Nine original client result files from [MarcoPizeta/flash-next-rtxpro6000-bench](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench), pinned to commit `fcdded55d1415d2a9b9256e58666d7c993cb5c2a`. Redistribution follows the included upstream MIT [LICENSE](LICENSE). Exact source paths, pinned download URLs, byte sizes, SHA256 checksums, and the shared `measurement_profile` are in [manifest.json](manifest.json).

These measurements use one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB), vLLM image `vllm/vllm-openai:qwen38-flash-next` built 2026-08-26, and vLLM `0.1.dev20073+g8e685d198`. The model is `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8`, with INT4 PLE CPU offload and MTP OFF. The client is **SGLang benchmark serving** (`sglang.benchmark.serving`). All files use concurrency 1, nominal input lengths 1024/8192/32768 and nominal output length 512; actual request lengths remain in the original arrays.

The pinned README reports an AMD Ryzen Threadripper 9960X (24C/48T), with maximum clock capped at 4.5 GHz from v3 onward, and 64 GB DDR5-4800 ECC RDIMM host RAM (2 × 32 GB, dual channel; 60.9 GiB usable). The GPU is rated at 600 W, with driver 595.84 and PCIe 5.0 x16. The host runs Ubuntu 26.04.1 LTS, kernel 7.0.0-30 and Docker 29.7.2. The model revision is `adf14c6b`, and PyTorch is `2.13.0+cu130`. The server image digest is reported only as prefix `sha256:fc120ece`; its full digest is unknown. The benchmark client runs from `lmsysorg/sglang:dev-cu13`; an immutable client revision is unavailable.

Only the current `results/` distinct-seed runs are included. The older fixed-seed `results-v1-seed42/` files are excluded. Runs r1/r2 form the train split; r3 forms the heldout split. Each file contains six request records, for 54 records total.

Even the current runs use two warm-up requests on the first prompt, so request index 0 has understated TTFT. The calibration policy excludes index 0 from each run: 30 train and 15 heldout records. SLO evaluation includes all six requests per file (54 total). This policy is metadata only: no records were deleted or modified.

The files provide no server queue timings and no per-request full HTTP end-to-end latencies (aggregate end-to-end summaries are present). `TTFT + sum(ITL)` represents the streaming span, not HTTP end-to-end latency. The upstream author already replaced `generated_texts` with a publication placeholder. Original bytes are preserved, including the non-standard JSON token `Infinity` in `request_rate`; use a compatible parser.
