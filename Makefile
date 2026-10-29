URL ?=
SERVER ?= vllm
WORKLOAD ?= short
REASONING ?= on

deploy-vllm:
	uv run modal deploy serve/vllm_app.py

deploy-sglang:
	uv run modal deploy serve/sglang_app.py

deploy-qwen3:
	MODEL=Qwen/Qwen3-8B APP_SUFFIX=qwen3 VLLM_EXTRA_ARGS="--reasoning-parser qwen3" uv run modal deploy serve/vllm_app.py

bench:
	uv run python -m bench.load --base-url $(URL) --server-name $(SERVER) --workload $(WORKLOAD) --out results/$(SERVER)-$(WORKLOAD).parquet

bench-reasoning:
	uv run python -m bench.load --base-url $(URL) --server-name vllm --model-name Qwen3-8B --reasoning $(REASONING) --max-tokens 2048 --out results/qwen3-8b-$(REASONING).parquet

report:
	uv run python -m bench.report results/*.parquet --write-readme

check:
	uv run pytest
	uv run ruff check .
	uv run pyright

.PHONY: deploy-vllm deploy-sglang deploy-qwen3 bench bench-reasoning report check
