# llm-serving-bench

Serves Qwen2.5-7B-Instruct with vLLM and with SGLang on Modal, plus Qwen3-8B on vLLM for the reasoning experiment, then measures what matters for a serving system under load: time to first token, inter-token latency, throughput, queue depth and KV-cache usage.

Latency is measured client-side from the streamed response, so the numbers are what a caller sees. Queue depth and KV usage come from each server's Prometheus `/metrics` endpoint, sampled during each concurrency level.

[docs/architecture.md](docs/architecture.md) walks through the request pipeline, shows where each metric is measured, and explains why these two servers are the ones worth comparing.

## Results

<!-- results:start -->
| workload | model | server | concurrency | ttft_p50_ms | ttft_p95_ms | itl_mean_ms | throughput_tok_s | queue_depth | kv_usage | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| shared-prefix | Qwen2.5-7B-Instruct | sglang | 1 | 661.1 | 920.9 | 32.8 | 25.3 | 0.0 | 0.2 | 0.0 |
| shared-prefix | Qwen2.5-7B-Instruct | sglang | 4 | 702.9 | 1685.3 | 33.6 | 96.2 | 0.0 | 0.2 | 1.0 |
| shared-prefix | Qwen2.5-7B-Instruct | sglang | 16 | 897.0 | 1170.6 | 36.4 | 281.3 | 0.0 | 0.2 | 0.0 |
| shared-prefix | Qwen2.5-7B-Instruct | sglang | 64 | 1252.6 | 4386.5 | 43.7 | 861.4 | 4.5 | 0.4 | 0.0 |
| shared-prefix | Qwen2.5-7B-Instruct | sglang | 128 | 7245.8 | 10204.1 | 47.6 | 1030.1 | 9.8 | 0.5 | 0.0 |
| shared-prefix | Qwen2.5-7B-Instruct | vllm | 1 | 242.2 | 492.9 | 32.1 | 27.6 | 0.0 | 0.0 | 1.0 |
| shared-prefix | Qwen2.5-7B-Instruct | vllm | 4 | 510.4 | 551.0 | 32.8 | 108.1 | 0.0 | 0.0 | 1.0 |
| shared-prefix | Qwen2.5-7B-Instruct | vllm | 16 | 566.8 | 651.2 | 35.2 | 308.8 | 0.0 | 0.0 | 0.0 |
| shared-prefix | Qwen2.5-7B-Instruct | vllm | 64 | 866.3 | 1170.4 | 45.1 | 989.3 | 0.0 | 0.1 | 0.0 |
| shared-prefix | Qwen2.5-7B-Instruct | vllm | 128 | 1751.9 | 6271.6 | 55.8 | 1237.0 | 0.0 | 0.1 | 0.0 |
| short | Qwen2.5-7B-Instruct | sglang | 1 | 677.2 | 874.3 | 32.4 | 28.7 | 0.0 | 0.0 | 0.0 |
| short | Qwen2.5-7B-Instruct | sglang | 4 | 730.6 | 1224.9 | 32.9 | 107.8 | 0.0 | 0.0 | 1.0 |
| short | Qwen2.5-7B-Instruct | sglang | 16 | 885.3 | 1002.8 | 33.8 | 428.2 | 0.0 | 0.2 | 0.0 |
| short | Qwen2.5-7B-Instruct | sglang | 64 | 1197.7 | 9706.5 | 37.4 | 1101.2 | 7.6 | 0.4 | 0.0 |
| short | Qwen2.5-7B-Instruct | sglang | 128 | 10649.3 | 20009.6 | 42.3 | 1126.2 | 5.4 | 0.3 | 0.0 |
| short | Qwen2.5-7B-Instruct | vllm | 1 | 467.3 | 500.1 | 32.5 | 29.0 | 0.0 | 0.0 | 1.0 |
| short | Qwen2.5-7B-Instruct | vllm | 4 | 519.8 | 561.8 | 32.7 | 115.3 | 0.0 | 0.0 | 0.0 |
| short | Qwen2.5-7B-Instruct | vllm | 16 | 572.1 | 663.9 | 33.9 | 442.8 | 0.0 | 0.0 | 0.0 |
| short | Qwen2.5-7B-Instruct | vllm | 64 | 838.5 | 1168.5 | 36.5 | 1594.1 | 0.0 | 0.1 | 0.0 |
| short | Qwen2.5-7B-Instruct | vllm | 128 | 2489.1 | 13182.0 | 41.8 | 1774.7 | 0.0 | 0.1 | 0.0 |
| short | Qwen3-8B | vllm | 1 | 396.8 | 766.6 | 34.1 | 28.6 | 0.0 | 0.0 | 0.0 |
| short | Qwen3-8B | vllm | 4 | 709.6 | 823.1 | 36.4 | 106.2 | 0.0 | 0.1 | 0.0 |
| short | Qwen3-8B | vllm | 16 | 869.9 | 987.6 | 39.2 | 316.6 | 0.0 | 0.2 | 0.0 |
| short | Qwen3-8B | vllm | 64 | 1064.7 | 8017.4 | 56.8 | 789.9 | 5.4 | 0.6 | 0.0 |
| short | Qwen3-8B | vllm | 128 | 14628.0 | 50058.7 | 76.1 | 1044.0 | 13.8 | 0.8 | 0.0 |
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

### Experiment 3: reasoning effort

Qwen3-8B on vLLM with `--reasoning-parser qwen3`, short workload, `max_tokens` 2048, three modes per request: thinking off, thinking with a 512-token budget, thinking unbounded. `ttfvt` is time to first visible token, the number a user feels. `answered_pct` is the share of requests that produced any visible answer before `max_tokens`. Chunk counts approximate tokens.

<!-- reasoning:start -->
| model | reasoning | concurrency | ttft_p50_ms | ttfvt_p50_ms | ttfvt_p95_ms | reasoning_chunks_mean | visible_chunks_mean | answered_pct | visible_tok_s | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | budget | 1 | 459.3 | 17961.2 | 18358.4 | 508.3 | 611.0 | 100.0 | 12.8 | 4.0 |
| Qwen3-8B | budget | 4 | 333.5 | 19408.2 | 19758.1 | 508.5 | 693.8 | 100.0 | 55.5 | 0.0 |
| Qwen3-8B | budget | 16 | 664.1 | 20318.6 | 20949.2 | 508.5 | 579.6 | 100.0 | 177.1 | 0.0 |
| Qwen3-8B | budget | 64 | 1334.0 | 44149.2 | 66642.1 | 509.3 | 589.7 | 100.0 | 404.2 | 0.0 |
| Qwen3-8B | budget | 128 | 31828.9 | 105742.7 | 126800.3 | 509.9 | 568.5 | 100.0 | 448.0 | 0.0 |
| Qwen3-8B | off | 1 | 396.8 | 396.8 | 766.6 | 0.0 | 697.7 | 100.0 | 28.3 | 0.0 |
| Qwen3-8B | off | 4 | 709.6 | 709.6 | 823.1 | 0.0 | 716.8 | 100.0 | 105.2 | 0.0 |
| Qwen3-8B | off | 16 | 869.9 | 869.9 | 987.6 | 0.0 | 719.4 | 100.0 | 313.4 | 0.0 |
| Qwen3-8B | off | 64 | 1064.7 | 1064.7 | 8017.4 | 0.0 | 728.1 | 100.0 | 782.4 | 0.0 |
| Qwen3-8B | off | 128 | 14628.0 | 14628.0 | 50058.7 | 0.0 | 721.1 | 100.0 | 1033.9 | 0.0 |
| Qwen3-8B | on | 1 | 282.2 | 43455.3 | 58339.2 | 1189.2 | 575.9 | 100.0 | 9.6 | 0.0 |
| Qwen3-8B | on | 4 | 330.4 | 49251.1 | 70712.1 | 1327.8 | 439.1 | 87.5 | 25.0 | 0.0 |
| Qwen3-8B | on | 16 | 702.0 | 46615.1 | 78226.2 | 1239.0 | 443.8 | 93.8 | 85.0 | 0.0 |
| Qwen3-8B | on | 64 | 6361.7 | 82012.0 | 167329.8 | 1217.4 | 454.8 | 85.6 | 145.0 | 31.0 |
| Qwen3-8B | on | 128 | 26572.8 | 151990.4 | 222476.5 | 1285.5 | 415.8 | 92.3 | 118.4 | 152.0 |
<!-- reasoning:end -->

Thinking tokens cost exactly what answer tokens cost. ITL at concurrency 1 is 34 ms in all three modes and tracks the same curve up through 128, so the engine does not distinguish the two. The budget is enforced to the token: every budget row shows 508 to 510 reasoning chunks, and unbounded thinking used 1190 to 1330 on the same five prompts.

TTFT becomes a vanity metric. With thinking on, median TTFT stays at 0.3 to 0.7 s up to 16 streams while the first visible token arrives after 43 to 49 s unbounded and 18 to 20 s with the 512 budget. The budget number is just the budget times the ITL. A user-facing SLO on a reasoning model has to be on TTFVT, and the queue and KV alarms fire long before TTFT moves.

Useful throughput falls by more than the thinking share suggests. At 128 streams the server produced 1034 visible tokens per second with thinking off, 448 with the budget and 118 unbounded. Thinking off is also not the Qwen2.5 baseline: Qwen3-8B writes about 720 visible tokens per answer against the 256 cap in experiments 1 and 2, so each request holds three times the KV and the pool fills at 64 streams (queue depth 5.4, KV 62 percent) where Qwen2.5 showed no queue. Thinking makes that worse, 80 and 85 percent KV at 64 for budget and unbounded, because every in-flight request carries its thinking tokens until it finishes.

Unbounded thinking did not answer reliably. Between 6 and 14 percent of requests hit `max_tokens` 2048 inside or right after the thinking block and returned no visible answer at all, and the budget mode answered 100 percent in every row. Above 16 streams the unbounded sweep also lost 31 of 128 streams at 64 and 152 of 256 at 128 to the same mid-stream disconnect seen in experiment 1, now at scale: surviving requests ran two to four minutes and the dropped ones were presumably the longer tail, so those two rows are optimistic. Streams that long are not something this ingress path serves reliably, and the table reports survivors only.

The practical reading is the one the primer gives: thinking off for simple turns, a budget for most, unbounded only where the task needs it and the client can wait a minute, and capacity planned on visible tokens per second rather than total.

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

For the reasoning experiment, deploy Qwen3-8B as a second vLLM app and sweep each mode:

```bash
make deploy-qwen3
make bench-reasoning URL=https://<qwen3-url> REASONING=off
make bench-reasoning URL=https://<qwen3-url> REASONING=budget
make bench-reasoning URL=https://<qwen3-url> REASONING=on
make report
```

`make report` rewrites both tables above from `results/*.parquet`.

Sweep settings: `python -m bench.load --help`. Defaults are the short workload, concurrency 1, 4, 16, 64, 128, 256 max tokens, and at least 32 requests or two full waves per level, whichever is larger.

## Checks

```bash
make check
```

Tests cover the stream statistics, the Prometheus parser and the report, and need no GPU.

## Notes

If the SGLang image build fails on flashinfer, add the wheel index for your CUDA and torch versions to `pip_install` in `serve/sglang_app.py`, as in the SGLang install docs.
