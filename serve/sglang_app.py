"""Serve the same model with SGLang on Modal, OpenAI-compatible API plus /metrics."""

import os
import subprocess

import modal

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("GPU", "A10G")
PORT = 8000
MINUTES = 60

volume = modal.Volume.from_name("llm-serving-bench-models", create_if_missing=True)
# Same reason as the vLLM app: flashinfer JIT needs nvcc, and the torch pin is a CUDA 13 build.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("sglang[all]==0.5.22", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models"})
)
app = modal.App("llm-serving-bench-sglang")


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
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        "model",
        "--context-length",
        "4096",
        "--enable-metrics",
    ]
    subprocess.Popen(cmd)
