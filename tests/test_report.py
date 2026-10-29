import pandas as pd
import pytest

from bench.report import README_END, README_START, splice_readme, summarize, to_markdown


def _frame(workload: str | None = "short") -> pd.DataFrame:
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
        }
        if workload is not None:
            row["workload"] = workload
        rows.append(row)
    return pd.DataFrame(rows)


def test_summary_metrics():
    s = summarize(_frame())
    assert len(s) == 1
    row = s.iloc[0]
    assert row["workload"] == "short"
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


def test_markdown_table_handles_missing_values():
    table = to_markdown(summarize(_frame()))
    assert table.splitlines()[0].startswith("| workload | server | concurrency |")
    assert "| short | vllm | 4 | 250.0 |" in table
    assert table.rstrip().endswith("| n/a | 2.0 |")


def test_splice_readme_replaces_between_markers():
    readme = f"# T\n\n{README_START}\nold\n{README_END}\n\ntail\n"
    out = splice_readme(readme, "NEW")
    assert "old" not in out
    assert f"{README_START}\nNEW\n{README_END}\n\ntail" in out


def test_splice_readme_without_markers_raises():
    with pytest.raises(ValueError):
        splice_readme("no markers", "x")
