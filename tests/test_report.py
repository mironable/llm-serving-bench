import pandas as pd
import pytest

from bench.report import README_END, README_START, splice_readme, summarize, to_markdown


def _frame() -> pd.DataFrame:
    rows = []
    for i, ttft in enumerate([0.1, 0.2, 0.3, 0.4]):
        rows.append(
            {
                "server": "vllm",
                "concurrency": 4,
                "request_idx": i,
                "ttft_s": ttft,
                "itl_mean_s": 0.02,
                "tokens": 100,
                "total_s": 2.0,
                "level_wall_s": 8.0,
                "queue_depth": 1.5,
                "kv_usage": None,
            }
        )
    return pd.DataFrame(rows)


def test_summary_metrics():
    s = summarize(_frame())
    assert len(s) == 1
    row = s.iloc[0]
    assert row["ttft_p50_ms"] == pytest.approx(250)
    assert row["itl_mean_ms"] == pytest.approx(20)
    assert row["throughput_tok_s"] == pytest.approx(400 / 8)
    assert row["queue_depth"] == 1.5
    assert pd.isna(row["kv_usage"])


def test_markdown_table_handles_missing_values():
    table = to_markdown(summarize(_frame()))
    assert table.splitlines()[0].startswith("| server | concurrency |")
    assert "| vllm | 4 | 250.0 |" in table
    assert table.rstrip().endswith("n/a |")


def test_splice_readme_replaces_between_markers():
    readme = f"# T\n\n{README_START}\nold\n{README_END}\n\ntail\n"
    out = splice_readme(readme, "NEW")
    assert "old" not in out
    assert f"{README_START}\nNEW\n{README_END}\n\ntail" in out


def test_splice_readme_without_markers_raises():
    with pytest.raises(ValueError):
        splice_readme("no markers", "x")
