# LLM serving primer

A plain-language guide to the terms in this repo, written for someone who runs ML systems but is new to LLM serving. It pairs with the [README](../README.md), which holds the measured results, and [architecture.md](architecture.md), which has the pipeline diagrams. The repo and its results are public at https://github.com/mironable/llm-serving-bench.

## The four layers

An LLM deployment has the same shape as any model serving system. Only the runtime in the middle is new.

| Layer | What it is | In this repo | Equivalent in a speech or vision stack on AWS |
|---|---|---|---|
| Weights | The trained model files, a few GB to hundreds of GB | Qwen2.5-7B-Instruct, cached on a Modal volume | Checkpoints in S3, promoted staging to production |
| Inference engine | The process that loads weights onto a GPU and schedules requests through the model | vLLM or SGLang | NeMo or faster-whisper inside a container |
| API server | The HTTP contract clients speak | OpenAI-compatible `/v1/chat/completions` plus Prometheus `/metrics` | SageMaker's `/invocations` |
| Hosting | Whatever supplies GPUs with an HTTPS front door, scaling and logs | Modal | SageMaker real-time endpoint, EKS with GPU nodes, or a managed API such as Bedrock |

The client is a fifth piece: here it is the load generator, in production a service or a Lambda behind a gateway.

## The engine, and why it is the interesting layer

A speech model does one forward pass per utterance. An LLM does one forward pass per generated token, and every pass has to see the attention state of everything generated so far. That state is the KV cache, and it lives in GPU memory next to the weights. The engine's job is scheduling many such sequences through one GPU at once. Two ideas make that efficient:

- Continuous batching. New requests join the running batch at any decode step rather than waiting for the batch to drain. Throughput scales with concurrency because of this.
- Paged KV cache. Attention state is stored in fixed-size blocks allocated on demand, so memory is not reserved for the maximum context up front. This sets how many sequences fit on one GPU.

vLLM and SGLang are the two most used open-source engines that implement both. They are Python libraries you install, in the same way NeMo is a library. There is no vLLM platform to send anything to. You package the library in a container and run it wherever there is a GPU.

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

In SageMaker terms, TTFT plus decode time is ModelLatency, network and ingress is OverheadLatency, and throughput is invocations per instance scaled by tokens.

## What the GPU changes

At low concurrency ITL is set by memory bandwidth: each decode step reads the full weights once. A 7B model in bf16 is about 15 GB; the A10G reads about 600 GB/s; one step is about 25 ms of memory traffic plus overhead, which is the 32 ms measured here. The engine barely matters at that point, which is why both engines matched.

| AWS instance | GPU | Memory | Bandwidth | Expected ITL, 7B bf16 |
|---|---|---|---|---|
| g5.xlarge | A10G | 24 GB | ~600 GB/s | ~32 ms (measured) |
| g6.xlarge | L4 | 24 GB | ~300 GB/s | ~55 to 60 ms |
| g6e.xlarge | L40S | 48 GB | ~860 GB/s | ~22 to 25 ms |
| p4d, p5 | A100, H100 | 40 to 80 GB | 1.5 to 3.3 TB/s | ~8 to 15 ms |

The L4 is newer and slower for decode because it is a low-power card with half the bandwidth. TTFT scales with compute rather than bandwidth. The knee scales with memory left over for KV cache after the weights, so a 48 GB card holds four to five times more concurrent sequences than a 24 GB one, which is often the bigger win.

Quantisation changes ITL without changing hardware: int8 or fp8 weights halve the bytes read per step and double the KV headroom.

## How a production team would structure this

First decision: managed API or self-hosted. Managed (Bedrock, OpenAI) means no engine, no capacity planning, pay per token, no control over latency under load. Self-hosted wins when token volume makes per-token pricing expensive, when a fine-tuned or domain model is needed, when p95 latency must be controlled, or when data residency rules out the managed API.

Self-hosted on AWS looks the same as hosting any other GPU model, for example a speech recognition model behind a SageMaker endpoint. The pieces, in the order a request meets them:

1. Model artifact. The weights live in an S3 bucket as versioned files. A new model version is uploaded to a staging bucket first, tested, then copied to production. Nothing in the serving stack holds weights itself; the container downloads them at start-up.
2. Container image. A Docker image built from the engine's official image, with the serve command and the model location baked in. It is pushed to ECR, AWS's container registry, and pinned by tag so a deploy always refers to one exact image.
3. Endpoint. A SageMaker real-time endpoint is a managed HTTPS service that runs that image on GPU instances you choose, for example g5 for A10G or g6e for L40S. SageMaker starts the container, health-checks it, routes requests to it, and replaces instances that fail. The endpoint, its instance type and count, and its autoscaling policy are declared in CloudFormation so the whole thing is reproducible and reviewable as code.
4. Autoscaling. For CPU services you scale on CPU utilisation. For an LLM endpoint that signal is useless because the GPU is busy long before the CPU is. The scaling metric is concurrent requests per instance, with the target set from the knee measured in a benchmark like this one.
5. Observability. The engine's `/metrics` endpoint is scraped into CloudWatch. Alarms go on queue depth (requests are waiting, add capacity), KV usage (the memory pool is nearly full, the next thing that happens is queueing) and p95 TTFT (what users feel). SageMaker adds its own ModelLatency and OverheadLatency metrics for free.
6. Client. A service or Lambda calls the endpoint with the OpenAI-style request, streams the response back to the user, and sits behind an API gateway that handles authentication and rate limiting.

The point of the comparison to a speech endpoint is that none of these six pieces is specific to LLMs. A team that already runs one GPU model on SageMaker has the pipeline, the IaC, the registry and the alarms; the LLM only changes which image goes in step 2 and which metrics matter in step 5.

Second decision: capacity. The benchmark in this repo is the capacity planning exercise. The knee per engine sets the autoscaling target and the instance count for a traffic forecast.

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
