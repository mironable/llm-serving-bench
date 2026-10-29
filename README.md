# llm-serving-bench

Serves Qwen2.5-7B-Instruct with vLLM and with SGLang on Modal, then measures what matters for a serving system under load: time to first token, inter-token latency, throughput, queue depth and KV-cache usage.

Latency is measured client-side from the streamed response, so the numbers are what a caller sees. Queue depth and KV usage come from each server's Prometheus `/metrics` endpoint, sampled during each concurrency level.

[docs/architecture.md](docs/architecture.md) walks through the request pipeline, shows where each metric is measured, and explains why these two servers are the ones worth comparing.

## Results

<!-- results:start -->
| workload | server | concurrency | ttft_p50_ms | ttft_p95_ms | itl_mean_ms | throughput_tok_s | queue_depth | kv_usage | errors |
|---|---|---|---|---|---|---|---|---|---|
| shared-prefix | sglang | 1 | 661.1 | 920.9 | 32.8 | 25.3 | 0.0 | 0.2 | 0.0 |
| shared-prefix | sglang | 4 | 702.9 | 1685.3 | 33.6 | 96.2 | 0.0 | 0.2 | 1.0 |
| shared-prefix | sglang | 16 | 897.0 | 1170.6 | 36.4 | 281.3 | 0.0 | 0.2 | 0.0 |
| shared-prefix | sglang | 64 | 1252.6 | 4386.5 | 43.7 | 861.4 | 4.5 | 0.4 | 0.0 |
| shared-prefix | sglang | 128 | 7245.8 | 10204.1 | 47.6 | 1030.1 | 9.8 | 0.5 | 0.0 |
| shared-prefix | vllm | 1 | 242.2 | 492.9 | 32.1 | 27.6 | 0.0 | 0.0 | 1.0 |
| shared-prefix | vllm | 4 | 510.4 | 551.0 | 32.8 | 108.1 | 0.0 | 0.0 | 1.0 |
| shared-prefix | vllm | 16 | 566.8 | 651.2 | 35.2 | 308.8 | 0.0 | 0.0 | 0.0 |
| shared-prefix | vllm | 64 | 866.3 | 1170.4 | 45.1 | 989.3 | 0.0 | 0.1 | 0.0 |
| shared-prefix | vllm | 128 | 1751.9 | 6271.6 | 55.8 | 1237.0 | 0.0 | 0.1 | 0.0 |
| short | sglang | 1 | 677.2 | 874.3 | 32.4 | 28.7 | 0.0 | 0.0 | 0.0 |
| short | sglang | 4 | 730.6 | 1224.9 | 32.9 | 107.8 | 0.0 | 0.0 | 1.0 |
| short | sglang | 16 | 885.3 | 1002.8 | 33.8 | 428.2 | 0.0 | 0.2 | 0.0 |
| short | sglang | 64 | 1197.7 | 9706.5 | 37.4 | 1101.2 | 7.6 | 0.4 | 0.0 |
| short | sglang | 128 | 10649.3 | 20009.6 | 42.3 | 1126.2 | 5.4 | 0.3 | 0.0 |
| short | vllm | 1 | 467.3 | 500.1 | 32.5 | 29.0 | 0.0 | 0.0 | 1.0 |
| short | vllm | 4 | 519.8 | 561.8 | 32.7 | 115.3 | 0.0 | 0.0 | 0.0 |
| short | vllm | 16 | 572.1 | 663.9 | 33.9 | 442.8 | 0.0 | 0.0 | 0.0 |
| short | vllm | 64 | 838.5 | 1168.5 | 36.5 | 1594.1 | 0.0 | 0.1 | 0.0 |
| short | vllm | 128 | 2489.1 | 13182.0 | 41.8 | 1774.7 | 0.0 | 0.1 | 0.0 |
<!-- results:end -->

Hardware and model are set by the `GPU` and `MODEL` env vars at deploy time, default A10G and Qwen/Qwen2.5-7B-Instruct.

Both servers ran with their defaults except a 4096 context length. `errors` counts streams the client lost during a level; the remaining requests still count. TTFT includes the round trip from Berlin to Modal's us-east region, which is most of the floor at concurrency 1, and that floor moved by up to 200 ms between runs made at different times, so compare TTFT within a run rather than across runs.

Two workloads. `short` sends one of five short user prompts with no system prompt. `shared-prefix` prepends the same system prompt of about 1500 tokens to every request, the shape of a production chat product, and is the workload prefix caching is designed for.

### Experiment 1: short prompts

Decode speed is the same engine to engine. ITL sits at 32 to 34 ms up to 16 concurrent streams on both, climbs to 37 to 38 ms at 64 and about 42 ms at 128. For a 7B model on one A10G both engines run the same kernels at the same batch sizes, so this is expected.

vLLM's knee is at 64 concurrent streams. Throughput is near linear up to 64, then 128 adds 11 percent more tokens per second while median TTFT triples and p95 TTFT reaches 13 s. Queue depth stays at zero because vLLM admits every request and lets prefill contend with decode instead.

SGLang's knee is lower, around 45 running requests, and it shows up as queueing. At 64 its queue depth averages 7.6 and p95 TTFT hits 9.7 s, and at 128 median TTFT is 10.6 s. Its scheduler log at that point reports 45 running requests at 99 percent token usage. The cause is a default: SGLang reserves 0.684 of GPU memory for weights plus KV pool, which after 15 GB of bf16 weights on a 24 GB card leaves roughly 11k tokens of KV capacity. vLLM defaults to 0.9. This is a configuration gap rather than an engine gap, and `--mem-fraction-static 0.9` on SGLang is the first thing to try before reading more into the throughput difference.

SGLang's TTFT is 150 to 300 ms higher than vLLM's at every level, including concurrency 1 where nothing is queued. That part is not explained by the memory fraction and is not yet understood. Candidates are its separate tokenizer process, its overlap scheduler delaying the first step, or its streaming interval.

Both endpoints dropped a stream once or twice per sweep at low concurrency, always as a client read error with a transfer-encoding complaint in Modal's proxy log. The flake sits between the engine's HTTP layer and Modal's ingress and did not recur under load.

### Experiment 2: shared system prompt

Both prefix caches work. Adding 1500 tokens of shared system prompt left TTFT where it was: SGLang 661 ms against 677 ms at concurrency 1 and 1253 against 1198 at 64, vLLM 567 against 572 at 16 and 866 against 839 at 64. Only the first request pays the prefill; every later one hits the cache. vLLM's automatic prefix caching by block hash matched SGLang's radix tree on this workload, so the headline SGLang advantage did not show up as a TTFT win.

Prefix caching saves prefill, not decode. ITL at 64 rose from 36 to 45 ms on vLLM and from 37 to 44 ms on SGLang, and throughput at 64 fell 38 percent on vLLM and 22 percent on SGLang. Every decode step still attends over the full context, cached or not, so a long shared prompt makes each step heavier and fills the KV pool faster. Capacity planning has to count the system prompt against every concurrent user.

SGLang degraded more gracefully at 128 than in experiment 1: median TTFT 7.2 s instead of 10.6 s, ITL 47.6 ms against vLLM's 55.8 ms, and its throughput gap to vLLM narrowed from 37 percent to 17 percent. The likely reason is that the radix tree stores the shared prefix once, so its smaller KV pool holds more sequences, but vLLM also shares identical blocks, so this is a hypothesis rather than a measured cause.

SGLang's extra 150 to 400 ms of TTFT at every level is still there and still unexplained.

### Not measured here

SGLang with `--mem-fraction-static 0.9`, which would test how much of its throughput deficit is the default KV pool size. Structured output. Long contexts that are not shared. Other GPUs. Quantised weights.

## Run it

```bash
uv sync
make deploy-vllm
make deploy-sglang
```

Each deploy prints a URL. Then, per server:

```bash
make bench URL=https://<vllm-url> SERVER=vllm
make bench URL=https://<sglang-url> SERVER=sglang
make bench URL=https://<vllm-url> SERVER=vllm WORKLOAD=shared-prefix
make bench URL=https://<sglang-url> SERVER=sglang WORKLOAD=shared-prefix
make report
```

`make report` rewrites the results table above from `results/*.parquet`.

Sweep settings: `python -m bench.load --help`. Defaults are the short workload, concurrency 1, 4, 16, 64, 128, 256 max tokens, and at least 32 requests or two full waves per level, whichever is larger.

## Checks

```bash
make check
```

Tests cover the stream statistics, the Prometheus parser and the report, and need no GPU.

## Notes

If the SGLang image build fails on flashinfer, add the wheel index for your CUDA and torch versions to `pip_install` in `serve/sglang_app.py`, as in the SGLang install docs.
