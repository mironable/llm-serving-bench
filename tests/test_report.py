import pandas as pd
import pytest

from bench.report import (
    COLUMNS,
    KEY,
    MARKERS,
    REASONING_COLUMNS,
    REASONING_KEY,
    splice_readme,
    summarize,
    summarize_reasoning,
    to_markdown,
)


def _frame(workload: str | None = "short", **extra) -> pd.DataFrame:
    rows = []
    for i, ttft in enumerate([0.1, 0.2, 0.3, 0.4]):
        row = {
            "server": "vllm",
            "concurrency": 4,
            "request_idx": i,
            "ttft_s": ttft,
            "itl_mean_s": 0.02,
            "tokens": 100,
            "total_s": 2.0,
            "level_wall_s": 8.0,
            "level_errors": 2,
            "queue_depth": 1.5,
            "kv_usage": None,
            **extra,
        }
        if workload is not None:
            row["workload"] = workload
        rows.append(row)
    return pd.DataFrame(rows)


def _reasoning_frame(mode: str) -> pd.DataFrame:
    # Two answered requests, two that ran out of budget inside thinking.
    rows = []
    for i in range(4):
        answered = i < 2
        rows.append(
            {
                "server": "vllm",
                "model": "Qwen3-8B",
                "workload": "short",
                "reasoning": mode,
                "concurrency": 4,
                "request_idx": i,
                "ttft_s": 0.1,
                "ttfvt_s": 1.0 if answered else None,
                "itl_mean_s": 0.03,
                "tokens": 300,
                "reasoning_chunks": 200,
                "visible_chunks": 100 if answered else 0,
                "total_s": 9.0,
                "level_wall_s": 20.0,
                "level_errors": 0,
                "queue_depth": 0.0,
                "kv_usage": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_summary_metrics():
    s = summarize(_frame())
    assert len(s) == 1
    row = s.iloc[0]
    assert row["workload"] == "short"
    assert row["model"] == "Qwen2.5-7B-Instruct"
    assert row["ttft_p50_ms"] == pytest.approx(250)
    assert row["itl_mean_ms"] == pytest.approx(20)
    assert row["throughput_tok_s"] == pytest.approx(400 / 8)
    assert row["queue_depth"] == 1.5
    assert pd.isna(row["kv_usage"])
    assert row["errors"] == 2


def test_runs_without_workload_column_default_to_short():
    s = summarize(_frame(workload=None))
    assert list(s["workload"]) == ["short"]


def test_workloads_are_separate_rows():
    df = pd.concat([_frame("short"), _frame("shared-prefix")], ignore_index=True)
    s = summarize(df)
    assert list(s["workload"]) == ["shared-prefix", "short"]


def test_thinking_runs_stay_out_of_main_table():
    df = pd.concat([_frame(), _reasoning_frame("on")], ignore_index=True)
    assert len(summarize(df)) == 1


def test_reasoning_summary():
    df = pd.concat([_reasoning_frame("on"), _reasoning_frame("budget")], ignore_index=True)
    s = summarize_reasoning(df)
    assert list(s["reasoning"]) == ["budget", "on"]
    row = s.iloc[0]
    assert row["ttfvt_p50_ms"] == pytest.approx(1000)
    assert row["answered_pct"] == pytest.approx(50)
    assert row["reasoning_chunks_mean"] == pytest.approx(200)
    assert row["visible_tok_s"] == pytest.approx(200 / 20)


def test_reasoning_summary_is_empty_without_thinking_runs():
    assert summarize_reasoning(_frame()).empty


def test_markdown_table_handles_missing_values():
    table = to_markdown(summarize(_frame()), COLUMNS, KEY)
    assert table.splitlines()[0].startswith("| workload | model | server | concurrency |")
    assert "| short | Qwen2.5-7B-Instruct | vllm | 4 | 250.0 |" in table
    assert table.rstrip().endswith("| n/a | 2.0 |")


def test_reasoning_markdown_header():
    table = to_markdown(
        summarize_reasoning(_reasoning_frame("on")), REASONING_COLUMNS, REASONING_KEY
    )
    assert table.splitlines()[0].startswith("| model | reasoning | concurrency |")


def test_splice_readme_replaces_between_markers():
    start, end = MARKERS["results"]
    readme = f"# T\n\n{start}\nold\n{end}\n\ntail\n"
    out = splice_readme(readme, "NEW")
    assert "old" not in out
    assert f"{start}\nNEW\n{end}\n\ntail" in out


def test_splice_readme_without_markers_raises():
    with pytest.raises(ValueError):
        splice_readme("no markers", "x")
