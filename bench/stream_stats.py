"""Client-side latency statistics for one streamed completion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestStats:
    ttft_s: float
    itl_mean_s: float | None
    tokens: int
    total_s: float


def summarize_stream(start: float, chunk_times: list[float], tokens: int) -> RequestStats:
    """chunk_times are wall-clock arrival times of content-bearing chunks, in order."""
    if not chunk_times:
        raise ValueError("stream produced no content chunks")
    ttft = chunk_times[0] - start
    total = chunk_times[-1] - start
    itl = (total - ttft) / (tokens - 1) if tokens > 1 else None
    return RequestStats(ttft_s=ttft, itl_mean_s=itl, tokens=tokens, total_s=total)
