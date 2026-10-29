"""Async load generator: sweeps concurrency and records per-request TTFT, ITL and tokens."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from bench.prom import parse_prometheus, server_gauges
from bench.stream_stats import RequestStats, summarize_stream

PROMPTS = [
    "Explain the difference between paged attention and continuous batching.",
    "Write a Python function that merges two sorted lists without using sort().",
    "Summarise the trade-offs between int8 and fp8 quantisation for inference.",
    "Describe how a KV cache grows with sequence length and why that matters.",
    "List five failure modes of an LLM serving system under burst traffic.",
]

WORKLOADS = ["short", "shared-prefix"]

_RULES = [
    "Answer in the learner's target language first, then give a one-line gloss in English.",
    "Correct at most two mistakes per reply and explain each with a short example sentence.",
    "Prefer everyday vocabulary over formal registers unless the learner asks for formality.",
    "When the learner writes a sentence, echo it back corrected before adding anything new.",
    "Keep replies under eighty words so the learner reads the whole message on a phone.",
    "Introduce one new grammar point at a time and reuse it in the next three exchanges.",
    "Never switch topic yourself; follow the learner's thread even if it is off the lesson.",
    "If the learner mixes languages, respond entirely in the target language.",
    "Avoid idioms the learner has not seen in the course unless they ask for slang.",
    "Praise specific improvements rather than giving generic encouragement.",
    "When pronunciation comes up, describe mouth position in plain words, not IPA.",
    "Close every reply with a single short question that invites the next turn.",
]

_LANGUAGES = ["Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Swedish", "Polish"]


def shared_system_prompt(target_tokens: int = 1500) -> str:
    """Deterministic ~1500-token system prompt shared by every shared-prefix request."""
    lines = ["You are a language tutor inside a mobile app. Follow these rules exactly."]
    i = 0
    # Roughly 1.3 tokens per word for this kind of English prose.
    while sum(len(line.split()) for line in lines) * 1.3 < target_tokens:
        lang = _LANGUAGES[i % len(_LANGUAGES)]
        rule = _RULES[i % len(_RULES)]
        lines.append(f"Rule {i + 1} ({lang} track): {rule}")
        i += 1
    return "\n".join(lines)


def messages_for(workload: str, i: int) -> list[dict[str, str]]:
    user = {"role": "user", "content": PROMPTS[i % len(PROMPTS)]}
    if workload == "shared-prefix":
        return [{"role": "system", "content": shared_system_prompt()}, user]
    return [user]


async def stream_one(
    client: httpx.AsyncClient, messages: list[dict[str, str]], max_tokens: int
) -> RequestStats:
    body = {
        "model": "model",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    chunk_times: list[float] = []
    usage_tokens: int | None = None
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event: dict[str, Any] = json.loads(payload)
            if usage := event.get("usage"):
                usage_tokens = usage.get("completion_tokens")
            choices = event.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                chunk_times.append(time.perf_counter())
    tokens = usage_tokens if usage_tokens is not None else len(chunk_times)
    return summarize_stream(start, chunk_times, tokens)


async def sample_metrics(
    client: httpx.AsyncClient, server: str, stop: asyncio.Event, interval_s: float = 0.5
) -> dict[str, float | None]:
    samples: list[dict[str, float | None]] = []
    while not stop.is_set():
        try:
            resp = await client.get("/metrics")
            samples.append(server_gauges(parse_prometheus(resp.text), server))
        except httpx.HTTPError:
            pass
        await asyncio.sleep(interval_s)
    out: dict[str, float | None] = {}
    for key in ("queue_depth", "kv_usage"):
        vals = [v for s in samples if (v := s[key]) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


async def run_level(
    client: httpx.AsyncClient,
    server: str,
    workload: str,
    concurrency: int,
    n_requests: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_metrics(client, server, stop))

    async def guarded(i: int) -> RequestStats | None:
        async with sem:
            try:
                return await stream_one(client, messages_for(workload, i), max_tokens)
            except (httpx.HTTPError, ValueError) as exc:
                # Dropped streams are a property of the system under test; count, don't abort.
                print(f"  request {i} failed: {type(exc).__name__}: {exc}")
                return None

    wall_start = time.perf_counter()
    stats = await asyncio.gather(*(guarded(i) for i in range(n_requests)))
    wall = time.perf_counter() - wall_start
    stop.set()
    gauges = await sampler
    errors = sum(s is None for s in stats)
    return [
        {
            "server": server,
            "workload": workload,
            "concurrency": concurrency,
            "request_idx": i,
            "ttft_s": s.ttft_s,
            "itl_mean_s": s.itl_mean_s,
            "tokens": s.tokens,
            "total_s": s.total_s,
            "level_wall_s": wall,
            "level_errors": errors,
            **gauges,
        }
        for i, s in enumerate(stats)
        if s is not None
    ]


async def main_async(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        await stream_one(client, messages_for(args.workload, 0), 16)  # warm up
        for c in args.concurrency:
            # At least two full waves, otherwise high levels never reach their concurrency.
            n_requests = max(args.requests_per_level, 2 * c)
            level = await run_level(
                client, args.server_name, args.workload, c, n_requests, args.max_tokens
            )
            rows.extend(level)
            if not level:
                print(f"{args.server_name} c={c:>3} every request failed")
                continue
            df = pd.DataFrame(level)
            tput = df["tokens"].sum() / df["level_wall_s"].iloc[0]
            print(
                f"{args.server_name} {args.workload} c={c:>3} "
                f"ttft_p50={df['ttft_s'].median() * 1000:.0f}ms "
                f"itl={df['itl_mean_s'].mean() * 1000:.1f}ms tput={tput:.0f} tok/s "
                f"errors={df['level_errors'].iloc[0]}"
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--server-name", required=True, choices=["vllm", "sglang"])
    p.add_argument("--workload", choices=WORKLOADS, default="short")
    p.add_argument(
        "--concurrency", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 16, 64, 128]
    )
    p.add_argument("--requests-per-level", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
