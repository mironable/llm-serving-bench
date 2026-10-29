"""Serve an open-weights model with vLLM on Modal, OpenAI-compatible API plus /metrics."""

import os
import shlex
import subprocess

import modal

# These are read at deploy time on the laptop and baked into the image below, because
# Modal re-imports this module inside the container where the shell env is not present.
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("GPU", "A10G")
# Extra `vllm serve` flags, e.g. "--reasoning-parser qwen3" for a thinking model.
EXTRA_ARGS_RAW = os.environ.get("VLLM_EXTRA_ARGS", "")
# Lets a second model deploy beside the default one, e.g. APP_SUFFIX=qwen3.
APP_NAME = "llm-serving-bench-vllm" + (f"-{s}" if (s := os.environ.get("APP_SUFFIX")) else "")
PORT = 8000
MINUTES = 60

volume = modal.Volume.from_name("llm-serving-bench-models", create_if_missing=True)
# flashinfer JIT-compiles sampling kernels at first use, so nvcc must be in the image.
# Torch 2.13 on PyPI is a CUDA 13 build; Modal hosts expose CUDA driver 13.3.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.32.0", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/models",
            "MODEL": MODEL,
            "VLLM_EXTRA_ARGS": EXTRA_ARGS_RAW,
        }
    )
)
app = modal.App(APP_NAME)


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
        *shlex.split(os.environ["VLLM_EXTRA_ARGS"]),
    ]
    subprocess.Popen(cmd)
