URL ?=
SERVER ?= vllm

deploy-vllm:
	uv run modal deploy serve/vllm_app.py

deploy-sglang:
	uv run modal deploy serve/sglang_app.py

bench:
	uv run python -m bench.load --base-url $(URL) --server-name $(SERVER) --out results/$(SERVER).parquet

report:
	uv run python -m bench.report results/*.parquet --write-readme

check:
	uv run pytest
	uv run ruff check .
	uv run pyright

.PHONY: deploy-vllm deploy-sglang bench report check
