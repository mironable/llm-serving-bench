URL ?=
SERVER ?= vllm
WORKLOAD ?= short
REASONING ?= on
# Any HF model id vLLM can serve; SUFFIX names the Modal app, EXTRA adds `vllm serve` flags.
MODEL ?= Qwen/Qwen2.5-7B-Instruct
SUFFIX ?=
EXTRA ?=
MAX_TOKENS ?= 256
EFFORT ?=
NAME = $(notdir $(MODEL))
SLUG = $(shell echo $(NAME) | tr A-Z a-z)

deploy-vllm:
	uv run modal deploy serve/vllm_app.py

deploy-sglang:
	uv run modal deploy serve/sglang_app.py

deploy:
	MODEL=$(MODEL) APP_SUFFIX=$(SUFFIX) VLLM_EXTRA_ARGS="$(EXTRA)" uv run modal deploy serve/vllm_app.py

deploy-qwen3:
	$(MAKE) deploy MODEL=Qwen/Qwen3-8B SUFFIX=qwen3 EXTRA="--reasoning-parser qwen3"

bench:
	uv run python -m bench.load --base-url $(URL) --server-name $(SERVER) --workload $(WORKLOAD) --out results/$(SERVER)-$(WORKLOAD).parquet

bench-reasoning:
	uv run python -m bench.load --base-url $(URL) --server-name vllm --model-name Qwen3-8B --reasoning $(REASONING) --max-tokens 2048 --out results/qwen3-8b-$(REASONING).parquet

# One model on the short workload; EFFORT=low|medium|high for gpt-oss style models.
bench-model:
	uv run python -m bench.load --base-url $(URL) --server-name vllm --model-name $(NAME) --max-tokens $(MAX_TOKENS) $(if $(EFFORT),--reasoning-effort $(EFFORT),) --out results/vllm-$(SLUG)$(if $(EFFORT),-$(EFFORT),).parquet

report:
	uv run python -m bench.report results/*.parquet --write-readme

check:
	uv run pytest
	uv run ruff check .
	uv run pyright

.PHONY: deploy-vllm deploy-sglang deploy deploy-qwen3 bench bench-reasoning bench-model report check
