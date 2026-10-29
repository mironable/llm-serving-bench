# llm-serving-bench

Serves Qwen2.5-7B-Instruct with vLLM and with SGLang on Modal, then measures what matters for a serving system under load: time to first token, inter-token latency, throughput, queue depth and KV-cache usage.

Latency is measured client-side from the streamed response, so the numbers are what a caller sees. Queue depth and KV usage come from each server's Prometheus `/metrics` endpoint, sampled during each concurrency level.

## Results

<!-- results:start -->
Not run yet.
<!-- results:end -->

Hardware and model are set by the `GPU` and `MODEL` env vars at deploy time, default A10G and Qwen/Qwen2.5-7B-Instruct.

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
make report
```

`make report` rewrites the results table above from `results/*.parquet`.

Sweep settings: `python -m bench.load --help`. Defaults are concurrency 1, 4, 16, 64 with 32 requests per level and 256 max tokens.

## Checks

```bash
make check
```

Tests cover the stream statistics, the Prometheus parser and the report, and need no GPU.

## Notes

If the SGLang image build fails on flashinfer, add the wheel index for your CUDA and torch versions to `pip_install` in `serve/sglang_app.py`, as in the SGLang install docs.
