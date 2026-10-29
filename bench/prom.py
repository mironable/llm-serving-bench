"""Minimal Prometheus text-exposition parser and per-server metric selection."""

# Server-specific metric names for queue depth and KV-cache usage, first match wins.
QUEUE_METRICS = {
    "vllm": ["vllm:num_requests_waiting"],
    "sglang": ["sglang:num_queue_reqs"],
}
KV_METRICS = {
    "vllm": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "sglang": ["sglang:token_usage"],
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Flatten samples to {name: value}; labels are dropped and same-name samples summed."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            # Label values may contain spaces, so split on the closing brace instead.
            name_part, _, tail = line.partition("{")
            _, _, rest = tail.rpartition("}")
        else:
            name_part, _, rest = line.partition(" ")
        fields = rest.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        out[name_part] = out.get(name_part, 0.0) + value
    return out


def pick(metrics: dict[str, float], candidates: list[str]) -> float | None:
    for name in candidates:
        if name in metrics:
            return metrics[name]
    return None


def server_gauges(metrics: dict[str, float], server: str) -> dict[str, float | None]:
    return {
        "queue_depth": pick(metrics, QUEUE_METRICS.get(server, [])),
        "kv_usage": pick(metrics, KV_METRICS.get(server, [])),
    }
