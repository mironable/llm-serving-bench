# LLM serving primer

A plain-language guide to the terms in this repo. It is the notebook of an ML engineer who has shipped conventional models to production and is learning LLM serving by doing it: every concept here was met while deploying and measuring something, and the explanations say what the measurement showed. The last section before the glossary works through one analogy in detail, a speech recognition endpoint, for readers who come from a similar stack. It pairs with the [README](../README.md), which holds the measured results, and [architecture.md](architecture.md), which has the pipeline diagrams. The repo and its results are public at https://github.com/mironable/llm-serving-bench.

## The four layers

An LLM deployment has the same shape as any model serving system. Only the runtime in the middle is new.

| Layer | What it is | In this repo |
|---|---|---|
| Weights | The trained model files, a few GB to hundreds of GB | Qwen2.5-7B-Instruct and the other models, cached on a Modal volume |
| Inference engine | The process that loads weights onto a GPU and schedules requests through the model | vLLM or SGLang |
| API server | The HTTP contract clients speak | OpenAI-compatible `/v1/chat/completions` plus Prometheus `/metrics` |
| Hosting | Whatever supplies GPUs with an HTTPS front door, scaling and logs | Modal |

The client is a fifth piece: here it is the load generator, in production a service behind a gateway.

## The engine, and why it is the interesting layer

A conventional model does one forward pass per request. An LLM does one forward pass per generated token, and every pass has to see the attention state of everything generated so far. That state is the KV cache, and it lives in GPU memory next to the weights. The engine's job is scheduling many such sequences through one GPU at once. Two ideas make that efficient:

- Continuous batching. New requests join the running batch at any decode step rather than waiting for the batch to drain. Throughput scales with concurrency because of this.
- Paged KV cache. Attention state is stored in fixed-size blocks allocated on demand, so memory is not reserved for the maximum context up front. This sets how many sequences fit on one GPU.

vLLM and SGLang are the two most used open-source engines that implement both. They are Python libraries you install. There is no vLLM platform to send anything to. You package the library in a container and run it wherever there is a GPU.

Where they differ:

| | vLLM | SGLang |
|---|---|---|
| KV cache | PagedAttention, with prefix caching by block hash | RadixAttention, a radix tree over token prefixes so any shared prefix is reused |
| Scheduler | Chunked prefill mixes prompt processing and decoding in one step | Overlaps CPU scheduling with the GPU step |
| Structured output | Guided decoding through pluggable grammar backends | Constrained decoding as a core feature |
| Coverage | Broadest model, hardware and quantisation support | Narrower, tuned for common architectures |

The measured consequence in this repo: decode speed was identical, SGLang's defaults reserved less memory for the KV cache and so queued earlier, and SGLang's time to first token was consistently higher for reasons not yet pinned down. See the results in the [README](../README.md#results).

## Containers and who provides what

Three ways the engine container comes into existence:

1. Build it yourself. A Dockerfile from an `nvidia/cuda` devel base, install Python and the engine, set the serve command as the entrypoint. The two `serve/*_app.py` files do exactly this, written as a `modal.Image` chain that Modal builds. The first deploy here failed because a slim base lacked the CUDA toolkit the engine needed to JIT-compile kernels. That was a Dockerfile bug.
2. Use the project's official image. Both engines publish ready images on Docker Hub with CUDA, torch and the engine already matched.
3. Use the cloud's bundled image. AWS ships Large Model Inference containers for SageMaker that wrap vLLM behind the `/invocations` contract. Bring-your-own-container works the same way as for any other model: build from the official image, push to ECR, point the endpoint at it.

Hosting is a separate decision from the engine. The same container runs on Modal, SageMaker, EKS, an EC2 GPU instance or a workstation.

## The metrics

Three numbers describe how a streamed response feels, two describe why.

Time to first token, TTFT. The gap between sending the request and the first word arriving. Paid once per request. It is queueing (waiting to be admitted) plus prefill (one forward pass over the whole prompt) plus the network round trip. Long prompts and a full server raise it.

Inter-token latency, ITL. Once words are flowing, the average gap between one token and the next. It is the time of one decode step for the whole running batch. At 33 ms a user sees about 30 tokens per second, faster than most people read. It rises as the batch grows.

Throughput. Tokens per second across all users. Roughly concurrency divided by ITL. It climbs with concurrency until the GPU or the KV cache saturates, then flattens while TTFT grows instead.

Queue depth. Requests waiting to be admitted. Zero means the server keeps up. Above zero, new users pay extra TTFT.

KV usage. How full the memory reserved for attention state is. At 100 percent the server stops admitting and the queue grows.

The relationship to keep: TTFT is what a user feels at the start, ITL during, throughput is what you pay for. A bigger batch buys throughput at the cost of ITL; what cannot fit in the batch shows up as queue depth and therefore TTFT. The knee in the throughput curve is the concurrency where throughput stops climbing and TTFT starts, and it is the number of users one GPU serves before you need another.

## What the GPU changes

At low concurrency ITL is set by memory bandwidth: each decode step reads the full weights once. A 7B model in bf16 is about 15 GB; the A10G reads about 600 GB/s; one step is about 25 ms of memory traffic plus overhead, which is the 32 ms measured here. The engine barely matters at that point, which is why both engines matched.

| AWS instance | GPU | Memory | Bandwidth | Expected ITL, 7B bf16 |
|---|---|---|---|---|
| g5.xlarge | A10G | 24 GB | ~600 GB/s | ~32 ms (measured) |
| g6.xlarge | L4 | 24 GB | ~300 GB/s | ~55 to 60 ms |
| g6e.xlarge | L40S | 48 GB | ~860 GB/s | ~22 to 25 ms |
| p4d, p5 | A100, H100 | 40 to 80 GB | 1.5 to 3.3 TB/s | ~8 to 15 ms |

The L4 is newer and slower for decode because it is a low-power card with half the bandwidth. TTFT scales with compute rather than bandwidth. The knee scales with memory left over for KV cache after the weights, so a 48 GB card holds four to five times more concurrent sequences than a 24 GB one, which is often the bigger win.

Model size works the same way on the same card: ITL is weight bytes over bandwidth, so Qwen2.5-1.5B decodes at 8 ms where the 7B takes 32, and one A10G served over 128 concurrent users of it without reaching a knee. A mixture-of-experts model such as gpt-oss-20b is the exception: at batch size 1 a token only reads its active experts, so it decoded at 10 ms despite 21B parameters, but under load every expert is read each step and ITL climbed four-fold to the dense figure.

Quantisation changes ITL without changing hardware. Measured on the 7B: int8 and fp8 weights halved the step time at low concurrency, 32.5 ms to 16.9, and int4 took it to 10.6. At full load all formats converged to 33 to 42 ms because the dequantised kernels become compute-bound, so the lasting gain is memory, with the KV pool one-third to one-tenth as full at the same concurrency. The cost is accuracy, which has to be measured on the task.

## Reasoning models and thinking effort

Some models (Qwen3, DeepSeek-R1, gpt-oss) write a hidden chain of thought before the visible answer. The engine serves those thinking tokens exactly like answer tokens: each one is a decode step and costs the same GPU time and KV memory. "Reasoning effort" is the knob that controls how much of that hidden work happens, and it is set at serve time and per request.

The three settings that matter when serving such a model:

- Thinking on or off. For Qwen3 it is a chat-template switch, `enable_thinking`, sent per request. Off makes the model answer directly like a normal chat model.
- A thinking budget. vLLM accepts `thinking_token_budget` per request when the server was started with `--reasoning-parser`. When the budget runs out the server forces the closing think marker so the model has to answer.
- A reasoning parser. Started with `--reasoning-parser qwen3`, the server splits the stream into `reasoning` and `content` fields so clients can hide or show the thinking.

What reasoning does to the metrics in this repo:

- Time to first token stays small and becomes misleading, because the first token is the start of the thinking block. The number a user feels is time to first visible token, TTFVT, which can be seconds. The report has a separate column for it.
- Throughput of useful tokens drops by whatever share of the output was thinking. The report counts reasoning and visible chunks separately and reports visible tokens per second.
- Some requests never produce a visible answer when `max_tokens` runs out inside the thinking block. The report shows this as the answered percentage. A budget prevents it at the cost of answer quality on hard prompts.
- Every in-flight request holds its thinking tokens in the KV cache until it finishes, so a thinking model fills the pool faster than a chat model at the same concurrency.

So "effort" is a trade of answer quality against latency and GPU cost. The practical pattern is to run thinking off for simple turns, a small budget for most, and unbounded only where the task needs it, and to alarm on TTFVT rather than TTFT.

## How a production team would structure this

First decision: managed API or self-hosted. Managed (Bedrock, OpenAI) means no engine, no capacity planning, pay per token, no control over latency under load. Self-hosted wins when token volume makes per-token pricing expensive, when a fine-tuned or domain model is needed, when p95 latency must be controlled, or when data residency rules out the managed API.

Self-hosted on AWS looks the same as hosting any other GPU model behind a SageMaker endpoint. The pieces, in the order a request meets them:

1. Model artifact. The weights live in an S3 bucket as versioned files. A new model version is uploaded to a staging bucket first, tested, then copied to production. Nothing in the serving stack holds weights itself; the container downloads them at start-up.
2. Container image. A Docker image built from the engine's official image, with the serve command and the model location baked in. It is pushed to ECR, AWS's container registry, and pinned by tag so a deploy always refers to one exact image.
3. Endpoint. A SageMaker real-time endpoint is a managed HTTPS service that runs that image on GPU instances you choose, for example g5 for A10G or g6e for L40S. SageMaker starts the container, health-checks it, routes requests to it, and replaces instances that fail. The endpoint, its instance type and count, and its autoscaling policy are declared in CloudFormation so the whole thing is reproducible and reviewable as code.
4. Autoscaling. For CPU services you scale on CPU utilisation. For an LLM endpoint that signal is useless because the GPU is busy long before the CPU is. The scaling metric is concurrent requests per instance, with the target set from the knee measured in a benchmark like this one.
5. Observability. The engine's `/metrics` endpoint is scraped into CloudWatch. Alarms go on queue depth (requests are waiting, add capacity), KV usage (the memory pool is nearly full, the next thing that happens is queueing) and p95 TTFT (what users feel). SageMaker adds its own ModelLatency and OverheadLatency metrics for free.
6. Client. A service or Lambda calls the endpoint with the OpenAI-style request, streams the response back to the user, and sits behind an API gateway that handles authentication and rate limiting.

None of these six pieces is specific to LLMs. A team that already runs one GPU model on SageMaker has the pipeline, the IaC, the registry and the alarms; the LLM only changes which image goes in step 2 and which metrics matter in step 5.

Second decision: capacity. The benchmark in this repo is the capacity planning exercise. The knee per engine sets the autoscaling target and the instance count for a traffic forecast.

## One analogy worked through: a speech recognition endpoint

The production stack I knew before this repo is speech recognition on SageMaker: a real-time endpoint with two model variants, a Whisper-style encoder-decoder and a CTC model served from a NeMo container built the bring-your-own-container way, weights in S3 promoted from staging to production, the endpoint and its per-variant CloudWatch alarms declared in CloudFormation, and a Lambda in front that calls the endpoint, aligns the transcript against the expected text and scores it. Every piece of this repo maps onto one of those, and the places where the mapping breaks are exactly the new concepts.

What maps directly:

| Speech endpoint | Here |
|---|---|
| Checkpoints in S3, promoted staging to production | Weights on a Modal volume, pulled from Hugging Face at first start |
| NeMo or faster-whisper inside the container | vLLM or SGLang inside the container |
| The `/invocations` contract | `/v1/chat/completions`, plus `/metrics` |
| The SageMaker endpoint and its instance type | The Modal app and its `gpu` argument |
| The Dockerfile pushed to ECR | The `modal.Image` chain in `serve/*_app.py` |
| The Lambda in front | The load generator here, a service behind a gateway in production |
| A second production variant | A second Modal app with a different model, mode or weight format |

Where it breaks, in the order I met each one:

1. One forward pass becomes many. A speech request is one pass per utterance and its cost is fixed by the audio length, so ModelLatency is the whole story. An LLM request is a prefill pass over the prompt and then one decode pass per generated token, streamed. TTFT is the closest thing to ModelLatency plus OverheadLatency of one utterance. ITL has no counterpart in batch recognition; the partial-result interval of a streaming recogniser is the nearest idea. Experiment 1 measured both and found decode speed identical across engines and the engines differing only in how they fail under load.
2. Memory grows with the conversation. A speech batch's memory is fixed by batch size times audio length. Here every token in flight adds to the KV cache, so capacity is a running quantity, KV usage, instead of a batch size set at deploy time, and the number of concurrent users one instance holds depends on how long their conversations are. Experiment 2 showed a 1500-token shared prompt costing nothing in prefill thanks to the cache and still cutting throughput by a quarter to a third because it occupies KV for every user.
3. The runtime owns admission. In the speech container the runtime runs the batch it is given. Here the engine decides at every decode step which requests join and which wait, which is why the vLLM versus SGLang choice mattered more than NeMo versus faster-whisper did, and why queue depth, a gauge the engine exposes, has no SageMaker counterpart: on an endpoint the same waiting hides inside OverheadLatency.
4. The cost of a request becomes a request parameter. With two speech variants, each had a fixed cost per utterance and the choice between them was made at deploy time. A reasoning model moves that choice into the request: thinking off, a budget, or unbounded, and experiment 3 measured the first visible token moving from under a second to 18 s to 45 s on the same model and GPU. The SLO has to be on the visible part of the output, and the per-variant alarm on ModelLatency becomes an alarm on time to first visible token.
5. Model size is the instance-sizing decision. For a dense model the per-token cost is weight bytes over the card's bandwidth, so experiment 4 found a 1.5B model serving four times the users of the 7B at a quarter of the latency on the same A10G, the way a smaller acoustic model lets one instance take more concurrent streams. A mixture-of-experts model has no speech counterpart: it decodes like a small model when idle and like its full size under load, so it is sized on the loaded number.
6. Quantisation is the same trade. Whisper in int8 through CTranslate2 is the speech version: same model, smaller weights, faster per utterance, an accuracy cost to measure, and more streams per instance because the weights leave memory free. Experiment 5 found exactly that on the 7B, with the latency gain largest on an idle server and the memory gain the one that lasts under load.
7. Autoscaling and alarms. The scaling signal for a speech endpoint is invocations per instance; for an LLM endpoint it is concurrent requests per instance, with the target set from the knee measured in a sweep like this one. The alarms move from ModelLatency and invocation errors to queue depth, KV usage, p95 TTFT and, for reasoning models, TTFVT.

What does not change: the artifact promotion, the container registry, the IaC for the endpoint, the gateway in front, and the habit of measuring capacity before setting an autoscaling target. A team that runs one GPU model this way already has most of the pipeline, and the LLM changes the image and the metrics.

## Glossary

- Prefill: the forward pass over the prompt that produces the first token. Compute-bound.
- Decode: the per-token forward passes after prefill. Memory-bandwidth-bound.
- KV cache: attention keys and values for every token in flight, kept in GPU memory.
- Continuous batching: admitting new requests into a running batch between decode steps.
- Paged attention: KV cache stored in fixed-size blocks allocated on demand.
- Prefix caching: reusing KV blocks for a prompt prefix seen before, such as a shared system prompt.
- Chunked prefill: splitting a long prompt's prefill across several steps so decode of other requests is not stalled.
- CUDA graph: a captured GPU execution plan replayed without CPU launch overhead, usually only for batch sizes captured at startup.
- Memory fraction: the share of GPU memory the engine reserves for weights plus KV cache. vLLM defaults to 0.9, SGLang to about 0.68.
- Streaming: returning tokens as server-sent events as they are generated rather than waiting for the full response.
- OpenAI-compatible API: the request and response shapes of `/v1/chat/completions`, adopted by every open-source engine so clients can switch backends without code changes.
- Thinking tokens: the hidden chain of thought a reasoning model emits before its answer, served and billed like any other tokens.
- Thinking budget: a per-request cap on thinking tokens; when hit, the server forces the model to close the thinking block and answer.
- Reasoning parser: the engine component that splits a stream into reasoning and content so clients can hide the thinking.
- TTFVT, time to first visible token: the delay a user feels with a reasoning model, from request to the first answer token after the thinking block.
- Mixture of experts, MoE: a model whose feed-forward layers are split into experts of which only a few are used per token, so the active parameters, and the bytes read at batch size 1, are a fraction of the total.
- Quantisation: storing weights in fewer bits than bf16. GPTQ and AWQ are int8 and int4 formats computed offline from the model; fp8 can be applied at load time; mxfp4 is the 4-bit format gpt-oss ships in. Fewer bytes per decode step and more memory for KV, at some cost in accuracy.
