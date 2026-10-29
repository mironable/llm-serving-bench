"""Serve an open-weights model with vLLM on Modal, OpenAI-compatible API plus /metrics."""

import os
import subprocess

import modal

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("GPU", "A10G")
PORT = 8000
MINUTES = 60

volume = modal.Volume.from_name("llm-serving-bench-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.32.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models"})
)
app = modal.App("llm-serving-bench-vllm")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/models": volume},
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=15 * MINUTES)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        "model",
        "--max-model-len",
        "4096",
    ]
    subprocess.Popen(cmd)
