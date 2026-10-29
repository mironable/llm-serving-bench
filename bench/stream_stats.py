"""Client-side latency statistics for one streamed completion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestStats:
    ttft_s: float
    itl_mean_s: float | None
    tokens: int
    total_s: float
    # Reasoning models: first visible (non-thinking) chunk, and the chunk split.
    ttfvt_s: float | None = None
    reasoning_chunks: int = 0
    visible_chunks: int = 0


def summarize_stream(
    start: float,
    chunk_times: list[float],
    tokens: int,
    first_visible: float | None = None,
    reasoning_chunks: int = 0,
) -> RequestStats:
    """chunk_times are wall-clock arrival times of content-bearing chunks, in order."""
    if not chunk_times:
        raise ValueError("stream produced no content chunks")
    ttft = chunk_times[0] - start
    total = chunk_times[-1] - start
    itl = (total - ttft) / (tokens - 1) if tokens > 1 else None
    return RequestStats(
        ttft_s=ttft,
        itl_mean_s=itl,
        tokens=tokens,
        total_s=total,
        ttfvt_s=None if first_visible is None else first_visible - start,
        reasoning_chunks=reasoning_chunks,
        visible_chunks=len(chunk_times) - reasoning_chunks,
    )
